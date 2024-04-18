import os
import time
import numpy as np
import random
import warnings
import wandb
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from imagenet_ipc import ImageFolderIPC
from utils import cifar_mixup
from utils import wandb_init

def parse_config():

    parser = argparse.ArgumentParser(
        description="PyTorch Post-Training")
    
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="resume from checkpoint")
    parser.add_argument("--output-dir", default="./save", type=str, help="output directory")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
    parser.add_argument("--check-ckpt", default=None, type=str, help="checkpoint path")
    parser.add_argument("--batch-size", default=128, type=int, help="batch size")
    
    parser.add_argument("--dataset", default="cifar100", type=str, help="type of dataset, [cifar100, in1k]")
    parser.add_argument("--data-path", default="./data", type=str, help="path to dataset")
    parser.add_argument("--syn-data-path", default="", type=str)
    parser.add_argument("--teacher-path", default="", type=str)

    parser.add_argument("--mixup-alpha", default=0.8, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--ipc", default=50, type=int)
    parser.add_argument('--temperature', type=int, default=30, help='temperature')
    
    parser.add_argument('--wandb-project', type=str,
                        default='rn18_fkd', help='wandb project name')
    # parser.add_argument('--wandb-api-key', type=str,
    #                     default=None, help='wandb api key')
    parser.add_argument('--wandb-name', type=str,
                        default="db_name", help='name')
    args = parser.parse_args()
    return args

args = parse_config()
if args.check_ckpt:
    checkpoint = torch.load(args.check_ckpt)
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    print(f"==> test ckp: {args.check_ckpt}, acc: {best_acc}, epoch: {start_epoch}")
    exit()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test_cifar100 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_dir = args.syn_data_path
val_dir = args.data_path

print("=> Using IPC setting of ", args.ipc)
train_dataset = ImageFolderIPC(
    root = train_dir, transform = transform_train, ipc = args.ipc)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size = 128, shuffle = True, num_workers = 2)
print("=> train data loaded")

if args.dataset == "cifar100":
    val_dataset = datasets.CIFAR100(
        root = val_dir, train = False, download = True, transform = transform_test_cifar100)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 4)
    print("=> Using CIFAR100 dataset")

elif args.dataset == "imagenet":
    valdir = os.path.join(args.data_path, "imagenet/val")
    val_dataset = datasets.ImageFolder(
        root = valdir, transform = transform_test_cifar100
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size = args.batch_size, shuffle = False, 
        num_workers = 4, pin_memory = True
    )
    print("=> Using ImageNet dataset")
print("=> val data loaded")

# Model
print("==> Building model..")

model = torchvision.models.get_model("resnet18", num_classes=100)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = nn.Identity()

model_student = model.to(device)
if device == "cuda":
    model_student = torch.nn.DataParallel(model_student)
    cudnn.benchmark = True

model_teacher = torchvision.models.get_model("resnet18", num_classes=100)
model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model_teacher.maxpool = nn.Identity()

model_teacher = nn.DataParallel(model_teacher).cuda()
# teacher model load from checkpoint
checkpoint = torch.load(args.teacher_path)
model_teacher.load_state_dict(checkpoint["state_dict"])

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    model_student.load_state_dict(checkpoint["state_dict"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model_student.parameters(), lr=0.001, weight_decay=0.01)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Train
def train(epoch) -> None:
# def train(epoch, wandb_metrics) -> None:
    temperature = args.temperature
    model_student.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_function_kl = nn.KLDivLoss(reduction="batchmean")
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = cifar_mixup(inputs, args.mixup_alpha)

        optimizer.zero_grad()
        outputs = model_student(inputs)
        outputs_ = F.log_softmax(outputs / temperature, dim=1)

        # teacher model pretrained, params load from checkpoint (saved in squeeze phase)
        soft_label = model_teacher(inputs).detach()
        soft_label_ = F.softmax(soft_label / temperature, dim=1)

        # crucial to make synthetic data and labels more aligned
        loss = loss_function_kl(outputs_, soft_label_)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f"Epoch: [{epoch}], Acc@1 {100.*correct/total:.3f}, Loss {train_loss/(batch_idx+1):.6f}")
    metrics = {
        "train/loss": float(f"{train_loss/(batch_idx+1):.6f}"),
        "train/Top1": float(f"{100.*correct/total:.3f}"),
        "train/epoch": epoch,
    }
    # wandb_metrics.update(metrics)
    wandb.log(metrics)

# Val
def test(epoch) -> None:
# def test(epoch, wandb_metrics) -> None:
    global best_acc
    model_student.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_student(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"Test: Acc@1 {100.*correct/total:.3f}, Loss {test_loss/(batch_idx+1):.6f}")

    metrics = {
        'val/loss': float(f"{test_loss/(batch_idx+1):.6f}"),
        'val/top1': float(f"{100.*correct/total:.3f}"),
        'val/epoch': epoch,
    }

    # wandb_metrics.update(metrics)
    # wandb.log(wandb_metrics)
    wandb.log(metrics)

    # Save checkpoint.
    acc = 100.0 * correct / total
    # if acc > best_acc:
    # save last checkpoint
    if True:
        state = {
            "state_dict": model_student.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')

        path = os.path.join(args.output_dir, "./ckpt.pth")
        torch.save(state, path)
        best_acc = acc

def main():
    args = parse_config()
    start_time = time.time()
    wandb_init(args)

    for epoch in range(start_epoch, start_epoch + args.epochs):

        # wandb_metrics = {}

        train(epoch)
        # fast test
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            test(epoch)
        scheduler.step()

    end_time = time.time()
    print(f"total time: {end_time - start_time} s")
    wandb.finish()

if __name__ == "__main__":
    main()