import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from imagenet_ipc import ImageFolderIPC
from utils import AverageMeter, accuracy, get_parameters
from utils_fkd import mix_aug

def get_args():
    parser = argparse.ArgumentParser("KD Training on ImageNet-1K")

    """data path flags"""
    parser.add_argument('--train-dir', type=str, default=None,
                        help='path to training dataset')
    parser.add_argument('--output-dir', type=str, default='./save', 
                        help='path to output dir')
    
    """training flags"""
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, 
                        help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int, default=0, 
                        help='start epoch')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='total epoch')
    parser.add_argument('-j', '--workers', default=16, type=int,
                        help='number of data loading workers')

    """optimization flags"""
    parser.add_argument('-T', '--temperature', type=float, default=3.0, 
                        help='temperature for distillation loss')
    parser.add_argument('--cos', action='store_true', 
                        help='cosine lr scheduler')
    parser.add_argument('--adamw-lr', type=float, default=0.001, 
                        help='adamw learning rate')
    parser.add_argument('--adamw-weight-decay', type=float, default=0.01, 
                        help='adamw weight decay')
    
    """model flags"""
    parser.add_argument('--model', type=str, default='resnet18', 
                        help='student model name')
    parser.add_argument('--teacher-model', type=str, default='resnet18',
                        help='teacher model name')
    parser.add_argument('--weights', type=str, default='ResNet18_Weights.DEFAULT', 
                        help='path to pretrained weights')
    # """wandb flags"""
    # parser.add_argument('--wandb-project', type=str, default='Temperature', 
    #                     help='wandb project name')
    # parser.add_argument('--wandb-api-key', type=str, default=None, 
    #                     help='wandb api key')
    # parser.add_argument('--wandb-name', type=str, default="db_name", 
    #                     help='name')
    # parser.add_argument('--wandb-group', type=str, default="syn_it2k",  
    #                     help='group name')
    # parser.add_argument('--wandb-job-type', type=str, default="val_KD", 
    #                     choices=['recover', 'relabel', 'val_KD', 'valKD_awp', 'val_FKD'],
    #                     help="job type in certain group")
    
    """mix augmentation flags"""
    parser.add_argument('--mix-type', default=None, type=str, 
                        choices=['mixup', 'cutmix', None], 
                        help='mixup or cutmix or None')
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--mode', type=str, default='fkd_save', choices=['fkd_save', 'fkd_load'],
                        help='training mode')
    parser.add_argument('--ipc', default=50, type=int, 
                        help='number of images per class')
    parser.add_argument('--save-soft-labels', action='store_true',
                        help='save soft labels')
    args = parser.parse_args()


    return args

def train(args):
    model = args.model
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    
    # load teacher model
    print("=> loading teacher model '{}'".format(args.teacher_model))
    # cross-model validation
    if args.weights is not None:
        print(f"load teacher model weights from {args.weights}")
        weights = args.weights.split('.')
        teacher_model = models.__dict__[args.teacher_model](weights = weights)
    else:
        raise ValueError(f"teacher model {args.teacher_model} not supported")
    
    # teacher_model = models.__dict__[args.teacher_model](pretrained=True)
    teacher_model = nn.DataParallel(teacher_model).cuda()
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    args.teacher_model = teacher_model
    args.teacher_model.eval()

    train_dataset = ImageFolderIPC(
        args.train_dir,
        transform=train_transforms,
        image_number=args.ipc
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    
    soft_labels = {}
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)
        data, target = data.cuda(), target.cuda()
        images, _, _, _ = mix_aug(data, args)

        soft_label = args.teacher_model(images).detach()
        soft_label = F.softmax(soft_label/args.temperature, dim=1).cpu().numpy()
        # print(soft_label)
        # append corresponding soft label to the list according its target
        for i, t in enumerate(target):
            if t.item() not in soft_labels:
                soft_labels[t.item()] = []
            soft_labels[t.item()].append(soft_label[i])
        # print(len(soft_labels[0]))
        print(f"batch {batch_idx} done")
        
    if args.save_soft_labels:
        print("=> save soft labels")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        # save soft labels dict
        np.save(os.path.join(args.output_dir, 'soft_labels.npy'), soft_labels)

if __name__ == '__main__':
    args = get_args()
    train(args)

