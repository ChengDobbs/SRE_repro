import os
import random
import collections
import numpy as np
import wandb
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from utils import save_images, validate
from utils import ImageProcessor
from utils import BNFeatureHook
from utils import lr_cosine_policy
from utils import wandb_init
from utils import SAM

wandb_metrics = {}

def parse_args():
    parser = argparse.ArgumentParser(
        "SRe2L: recover data from pre-trained model")
    
    '''data path flags'''
    parser.add_argument('--exp-name', type=str, default='in1k_rn18_it2k_4091',
                        help='name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--dataset', type=str, default='imagenet', 
                        help='type of dataset', choices=['cifar100', 'imagenet', 'tinyIN'])
    parser.add_argument('--select-num', type=int, default=50,
                        help='number of images selected for each class')
    parser.add_argument('--data-select-path', type=str, default='/raid/data_dujw/imagenet/select/select_dataset',
                        help='path to selected images')
    parser.add_argument('--syn-data-path', type=str, default='./syn_data', 
                        help='where to store synthetic data')
    parser.add_argument('--store-best-images', action='store_true',
                        help='whether to store best images')
    
    '''optimization flags'''
    parser.add_argument('--batch-size', type=int, default=100, 
                        help='number of images to optimize at the same time')
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--jitter', default=32, type=int, 
                        help='random shift on the synthetic data')
    parser.add_argument('--r-bn', type=float, default=0.01,
                        help='coefficient for BN feature distribution regularization')
    parser.add_argument('--first-bn-multiplier', type=float, default=10.,
                        help='additional multiplier on first bn layer of R_bn')
    parser.add_argument('--rho', type=float, default=0.015,
                        help='rho parameter for SAM optimizer')
    parser.add_argument('--sam-steps', type=int, default=15,
                        help='number of SAM steps')
    parser.add_argument('--tanh-space', action='store_true',
                        help='whether to use tanh space for optimization')
    
    '''model flags'''
    parser.add_argument('--arch-name', type=str, default='resnet18',
                        help='arch name from pretrained torchvision models')
    parser.add_argument('--arch-path', type=str, default='')
    parser.add_argument('--verifier', action='store_true', default=False,
                        help='whether to evaluate synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='mobilenet_v2',
                        help='arch name from torchvision models to act as a verifier')
    parser.add_argument('--ipc-start', default=0, type=int)
    parser.add_argument('--ipc-end', default=50, type=int)

    '''wandb flags'''
    parser.add_argument('--wandb-project', type=str, default='rn18_in1k_syn', 
                        help='wandb project name')
    parser.add_argument('--wandb-name', type=str, default='db_name', 
                        help='name')
    parser.add_argument('--wandb-group', type=str, default='syn_it3k', 
                        help='group name')
    parser.add_argument('--wandb-job-type', type=str, default='recover', help='job type in certain group',
                        choices=['recover', 'recover_awp', 'relabel', 'val_KD', 'val_FKD'])
    
    args = parser.parse_args()
    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    args.data_select_path = os.path.join(args.data_select_path + '_' + str(args.select_num))
    # class-wise alignment
    args.sorted_path = sorted(os.listdir(args.data_select_path))
    return args

def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    return preprocess(image)

def get_images(args, model_teacher, ipc_id):
    global wandb_metrics
    print("get_images call")
    batch_size = args.batch_size
    best_cost = 1e4
    dataset = args.dataset

    loss_r_feature_layers = []
    for module in model_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BNFeatureHook(module))

    # setup target labels
    # targets_all = torch.LongTensor(np.random.permutation(1000))
    targets_all = torch.LongTensor(np.arange(1000))

    for kk in range(0, 1000, batch_size):
        
        start_index = kk
        end_index = min(kk + batch_size, 1000)
        targets = targets_all[start_index:end_index].to('cuda')
        
        inputs = []
        for folder_index in range(start_index, end_index):
            # folder_path = os.path.join(args.data_select_path, os.listdir(args.data_select_path)[folder_index])
            folder_path = os.path.join(args.data_select_path, args.sorted_path[folder_index])
            image_path = os.path.join(folder_path, os.listdir(folder_path)[ipc_id])
            image = load_image(image_path)
            inputs.append(image)

        img_process = ImageProcessor(dataset)
        inputs = torch.stack(inputs).to('cuda')

        if args.tanh_space:
            inputs.data = img_process.denormalize(inputs.data)
            inputs.data = img_process.inverse_tanh_space(inputs.data)

        inputs = inputs.requires_grad_(True)

        iterations_per_layer = args.iteration
        lim_0, lim_1 = args.jitter , args.jitter

        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps = 1e-8)

        if args.sam_steps:
            paras = model_teacher.parameters()
            sam_optimizer = SAM(paras, optimizer, rho = args.rho / args.sam_steps)
        
        if args.tanh_space:
            inputs.data = img_process.tanh_space(inputs.data)
            inputs.data = img_process.normalize(inputs.data)

        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer) # 0 - do not use warmup
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)

            aug_func = transforms.Compose([
                transforms.RandomResizedCrop(224), # Crop Coord
                transforms.RandomHorizontalFlip(), # Flip Status
            ])

            inputs_jit = aug_func(inputs)
            # apply random jitter offsets
            off1 = random.randint(0, lim_0)
            off2 = random.randint(0, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

            # forward pass
            optimizer.zero_grad()

            if iteration ==0:
                for _ in range(args.sam_steps):
                    outputs = model_teacher(inputs_jit)
                    loss_ce = criterion(outputs, targets)
                    loss_ce.backward()
                    sam_optimizer.first_step(True)
            outputs = model_teacher(inputs_jit)

            # R_cross classification loss
            loss_ce = criterion(outputs, targets)

            # R_feature loss
            rescale = [args.first_bn_multiplier] + [1.]* (len(loss_r_feature_layers) - 1)

            # loss_r_bn_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])
            loss_r_bn_feature = [
                mod.r_feature.to(loss_ce.device) * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)
            ]
            loss_r_bn_feature = torch.stack(loss_r_bn_feature).sum()

            # combining losses
            # tv_l2 = l2_scale = 0, omitted
            loss_aux = args.r_bn * loss_r_bn_feature
            loss = loss_ce + loss_aux

            metrics = {
                'syn/loss_ce': loss_ce.item(),
                'syn/loss_aux': loss_aux.item(),
                'syn/loss_total': loss.item(),
                # 'syn/aux_weight': aux_weight,
                # 'image/acc_image': acc_image,
                # 'image/loss_image': loss_image,
                'syn/ipc_id': ipc_id,
            }

            wandb_metrics.update(metrics)
            wandb.log(wandb_metrics)

            save_every = args.batch_size
            if iteration % save_every==0:
                print("------------iteration {}----------".format(iteration))
                print("total loss", loss.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("main criterion", criterion(outputs, targets).item())
                
            # comment below line to speed up the training (no validation process)
            if args.verifier:
                model_verifier = models.__dict__[args.verifier_arch](pretrained=True)
                model_verifier = model_verifier.cuda()
                model_verifier.eval()
                for p in model_verifier.parameters():
                    p.requires_grad = False
                validate(inputs, targets, model_verifier)

            # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            img_process = ImageProcessor(dataset)
            inputs.data = img_process.clip(inputs.data)

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()

        if args.store_best_images:
            best_inputs = inputs.data.clone() # using multicrop, save the last one
            best_inputs = img_process.denormalize(best_inputs)
            save_images(args, best_inputs, targets, ipc_id)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
    torch.cuda.empty_cache()

def main_syn(args, ipc_id):

    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    # avoid higher version torchvision warning
    # model_teacher = models.__dict__[args.arch_name] \
    #                 (weights = models.resnet.ResNet18_Weights.DEFAULT)
    model_teacher = models.__dict__[args.arch_name](pretrained=True)
    # multi-GPUs prerequisite, pesudo parallelism
    model_teacher = nn.DataParallel(model_teacher).cuda()

    model_teacher.eval()
    get_images(args, model_teacher, ipc_id)


if __name__ == '__main__':

    args = parse_args()
    wandb_init(args)

    for ipc_id in range(args.ipc_start, args.ipc_end):
        print("ipc = ", ipc_id)
        main_syn(args, ipc_id)

    wandb.finish()