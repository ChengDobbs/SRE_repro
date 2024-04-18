import argparse
import math
import os
import shutil
import sys
import time
import wandb
import tqdm
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.optim.lr_scheduler import LambdaLR

from utils import ImageFolder_FKD_MIX
from utils import (AverageMeter, 
                   accuracy, 
                   get_parameters
                   )
from utils import (ComposeWithCoords,
                   RandomHorizontalFlipWithRes, 
                   RandomResizedCropWithCoords,
                   )
from utils import MixAugmentation
from utils import wandb_init

def parse_args():
    parser = argparse.ArgumentParser("FKD Training on ImageNet-1K")

    """Optimization related flags"""
    parser.add_argument('-b', '--batch-size', type=int, default=256, 
                        help='batch size')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--grad-accum-steps', type=int, default=1, 
                        help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--cos', default=False, action='store_true', 
                        help='cosine lr scheduler')
    parser.add_argument('--sgd', default=False, action='store_true', 
                        help='sgd optimizer')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1.024, 
                        help='sgd init learning rate')  # checked
    parser.add_argument('--momentum', type=float, default=0.875, 
                        help='sgd momentum')  # checked
    parser.add_argument('--weight-decay', type=float, default=3e-5, 
                        help='sgd weight decay')  # checked
    parser.add_argument('--adamw-lr', type=float, default=0.001, 
                        help='adamw learning rate')
    parser.add_argument('--adamw-weight-decay', type=float, default=0.01, 
                        help='adamw weight decay')
    parser.add_argument('--start-epoch', type=int, default=0, 
                        help='start epoch for resuming')
    parser.add_argument('--epochs', type=int, default=300,
                        help='total epoch')
    
    """distributed training flags"""
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    
    """Data paths flags"""
    parser.add_argument('--syn-data-dir', type=str, default='./syn_data',
                        help='path to synthesized data in recover stage')
    parser.add_argument('--val-dir', type=str, default='/path/to/imagenet/val', 
                        help='path to validation dataset')
    parser.add_argument('--output-dir', type=str, default='./save', 
                        help='path to output dir')

    """Label flags"""
    parser.add_argument('--model', type=str,
                        default='resnet18', help='student model name')
    parser.add_argument('--keep-topk', type=int, default=1000,
                        help='keep topk logits for kd loss')
    parser.add_argument('-T', '--temperature', type=float, default=20.0, 
                        help='temperature for distillation loss')
    parser.add_argument('--use-fp16', dest='use_fp16', action='store_true',
                        help='save soft labels as `fp16`')
    parser.add_argument('--mode', type=str, default='fkd_load', 
                        help='fkd_save or fkd_load')
    parser.add_argument('--fkd-path', type=str, default=None, 
                        help='path to fkd label')
    parser.add_argument('--fkd-seed', default=42, type=int,
                        help='seed for batch loading sampler')
    
    """WandB related flags"""
    parser.add_argument('--wandb-project', type=str, default='val_in1k', 
                        help='wandb project name')
    parser.add_argument('--wandb-api-key', type=str, default=None, 
                        help='wandb api key')
    parser.add_argument('--wandb-name', type=str, default="db_name", 
                        help='name')
    
    """Mix augmentation"""
    parser.add_argument("--min-scale-crops", type=float, default=0.08,
                        help="argument in RandomResizedCrop")
    parser.add_argument("--max-scale-crops", type=float, default=1.,
                        help="argument in RandomResizedCrop")
    parser.add_argument('--mix-type', default=None, type=str, choices=['mixup', 'cutmix', None], 
                        help='mixup or cutmix or None')
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    wandb_init(args)

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")

    assert os.path.exists(args.train_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Data loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode=args.mode,
        args_epoch=args.epochs,
        args_bs=args.batch_size,
        root=args.train_dir,
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(size=224,
                                        scale=(0.08, 1),
                                        interpolation=transforms.InterpolationMode.BILINEAR),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            normalize,
        ]))

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)

    # only main process, no worker process
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
    #     num_workers=0, pin_memory=True,
    #     prefetch_factor=None)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.workers, pin_memory=True)

    # load validation data
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=int(args.batch_size/4), shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print('load data successfully')


    # load student model
    print("=> loading student model '{}'".format(args.model))
    model = torchvision.models.__dict__[args.model](pretrained=False)
    model = nn.DataParallel(model).cuda()
    model.train()

    if args.sgd:
        optimizer = torch.optim.SGD(get_parameters(model),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(get_parameters(model),
                                      lr=args.adamw_lr,
                                      weight_decay=args.adamw_weight_decay)

    if args.cos == True:
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (1. + math.cos(math.pi * step / args.epochs)) if step <= args.epochs else 0, last_epoch=-1)
    else:
        scheduler = LambdaLR(optimizer,
                             lambda step: (1.0-step/args.epochs) if step <= args.epochs else 0, last_epoch=-1)


    args.best_acc1=0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch: {epoch}")

        global wandb_metrics
        wandb_metrics = {}

        train(model, args, epoch)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            top1 = validate(model, args, epoch)
        else:
            top1 = 0

        wandb.log(wandb_metrics)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = top1 > args.best_acc1
        args.best_acc1 = max(top1, args.best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': args.best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
        }, is_best, output_dir=args.output_dir)

def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def train(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='batchmean')

    model.train()
    t1 = time.time()
    args.train_loader.dataset.set_epoch(epoch)
    for batch_idx, batch_data in enumerate(args.train_loader):
        images, target, flip_status, coords_status = batch_data[0]
        mix_index, mix_lam, mix_bbox, soft_label = batch_data[1:]

        images = images.cuda()
        target = target.cuda()
        soft_label = soft_label.cuda().float()  # convert to float32
        images, _, _, _ = MixAugmentation.mix_aug(images, args, mix_index, mix_lam, mix_bbox)

        optimizer.zero_grad()
        assert args.batch_size % args.grad_accum_steps == 0
        small_bs = args.batch_size // args.grad_accum_steps

        # images.shape[0] is not equal to args.batch_size in the last batch, usually
        if batch_idx == len(args.train_loader) - 1:
            accum_step = math.ceil(images.shape[0] / small_bs)
        else:
            accum_step = args.grad_accum_steps

        for accum_id in range(accum_step):
            partial_images = images[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_target = target[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_soft_label = soft_label[accum_id * small_bs: (accum_id + 1) * small_bs]

            output = model(partial_images)
            prec1, prec5 = accuracy(output, partial_target, topk=(1, 5))

            output = F.log_softmax(output/args.temperature, dim=1)
            partial_soft_label = F.softmax(partial_soft_label/args.temperature, dim=1)
            loss = loss_function_kl(output, partial_soft_label)
            # loss = loss * args.temperature * args.temperature
            loss = loss / args.grad_accum_steps
            loss.backward()

            n = partial_images.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        optimizer.step()


        # output = model(images)
        # prec1, prec5 = accuracy(output, target, topk=(1, 5))
        # output = F.log_softmax(output/args.temperature, dim=1)
        # soft_label = F.softmax(soft_label/args.temperature, dim=1)

        # loss = loss_function_kl(output, soft_label)
        # # loss = loss * args.temperature * args.temperature

        # n = images.size(0)
        # objs.update(loss.item(), n)
        # top1.update(prec1.item(), n)
        # top5.update(prec5.item(), n)

        # if batch_idx == 0:
        #     optimizer.zero_grad()

        # # do not support accumulate gradient, batch_size is fixed to 1024
        # assert args.grad_accum_steps == 1
        # if args.grad_accum_steps > 1:
        #     loss = loss / args.grad_accum_steps

        # loss.backward()

        # if (batch_idx + 1) % args.grad_accum_steps == 0 or batch_idx == len(args.train_loader) - 1:
        #     optimizer.step()
        #     optimizer.zero_grad()

    metrics = {
        "train/loss": objs.avg,
        "train/Top1": top1.avg,
        "train/Top5": top5.avg,
        "train/lr": scheduler.get_last_lr()[0],
        "train/epoch": epoch,}
    wandb_metrics.update(metrics)


    printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                'train_time = {:.6f}'.format((time.time() - t1))
    print(printInfo)
    t1 = time.time()


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1  = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
              'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    print(logInfo)

    metrics = {
        'val/loss': objs.avg,
        'val/top1': top1.avg,
        'val/top5': top5.avg,
        'val/epoch': epoch,
    }
    wandb_metrics.update(metrics)

    return top1.avg

def save_checkpoint(state, is_best, output_dir=None,epoch=None):
    if epoch is None:
        path = output_dir + '/' + 'checkpoint.pth.tar'
    else:
        path = output_dir + f'/checkpoint_{epoch}.pth.tar'
    torch.save(state, path)

    if is_best:
        path_best = output_dir + '/' + 'model_best.pth.tar'
        shutil.copyfile(path, path_best)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()
    wandb.finish()