import argparse
import os
import random
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms

from utils import save_images, validate
from utils import ImageProcessor
from utils import BNFeatureHook
from utils import lr_cosine_policy
from utils import wandb_init

wandb_metrics = {}

def parse_args():
    parser = argparse.ArgumentParser(
        "recover data from pre-trained model")
    ''' Data save flags '''
    parser.add_argument('--exp-name', type=str, default='test',
                        help='name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--syn-data-path', type=str,
                        default='./syn_data', help='where to store synthetic data')
    parser.add_argument('--store-last-images', action='store_true',
                        help='whether to store best images')
    
    ''' Optimization flags '''
    parser.add_argument('--batch-size', type=int,
                        default=100, help='number of images to optimize at the same time')
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--jitter', default=4, type=int, help='random shift on the synthetic data')
    parser.add_argument('--r-bn', type=float, default=0.05,
                        help='coefficient for BN feature distribution regularization')
    parser.add_argument('--first-bn-multiplier', type=float, default=10.,
                        help='additional multiplier on first bn layer of R_bn')
    
    ''' Model flags '''
    parser.add_argument('--arch-name', type=str, default='resnet18',
                        help='arch name from pretrained torchvision models')
    parser.add_argument('--verifier', action='store_true',
                        help='whether to evaluate synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='mobilenet_v2',
                        help="arch name from torchvision models to act as a verifier")
    parser.add_argument('--arch-path', type=str, default='')
    parser.add_argument('--verifier-arch-path', type=str, default='')
    
    ''' Training Helpers '''
    parser.add_argument('--ipc-start', default=0, type=int)
    parser.add_argument('--ipc-end', default=50, type=int)

    '''wandb flags'''
    parser.add_argument('--wandb-project', type=str, default='SRe2L', 
                        help='wandb project name')
    parser.add_argument('--wandb-name', type=str, default='tiny', 
                        help='name')
    parser.add_argument('--wandb-group', type=str, default='syn_it3k', 
                        help='group name')
    parser.add_argument('--wandb-job-type', type=str, default='recover', help='job type in certain group',
                        choices=['recover', 'relabel', 'val_KD', 'val_FKD'])
    
    args = parser.parse_args()
    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    return args


def get_images(args, model_teacher, hook_for_display, ipc_id):
    global wandb_metrics
    print("generating one IPC images (200)")
    batch_size = args.batch_size

    loss_r_feature_layers = []
    for module in model_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BNFeatureHook(module))

    # setup target labels
    # targets_all = torch.LongTensor(np.random.permutation(200))
    targets_all = torch.LongTensor(np.arange(200))

    for kk in range(0, 200, batch_size):
                
        start_index = kk
        end_index = min(kk + batch_size, 200)
        targets = targets_all[start_index:end_index].to('cuda')

        data_type = torch.float
        inputs = torch.randn((targets.shape[0], 3, 64, 64), requires_grad=True, device='cuda',
                             dtype=data_type)

        iterations_per_layer = args.iteration

        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)  # 0 - do not use warmup
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)

            aug_function = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
            ])
            inputs_jit = aug_function(inputs)

            # apply random jitter offsets
            off1 = random.randint(-args.jitter, args.jitter)
            off2 = random.randint(-args.jitter, args.jitter)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

            # forward pass
            optimizer.zero_grad()
            outputs = model_teacher(inputs_jit)

            # R_cross classification loss
            loss_ce = criterion(outputs, targets)

            # R_feature loss
            rescale = [args.first_bn_multiplier] + [1.]* (len(loss_r_feature_layers) - 1)
            loss_r_bn_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

            # final loss
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
            if iteration % save_every == 0 and args.verifier:
                print("------------iteration {}----------".format(iteration))
                print("total loss", loss.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("main criterion", criterion(outputs, targets).item())
                # comment below line can speed up the training (no validation process)
                if hook_for_display is not None:
                    hook_for_display(inputs, targets)

            # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            img_process = ImageProcessor('tinyIN')
            inputs.data = img_process.clip(inputs.data)

        if args.store_last_images:
            best_inputs = inputs.data.clone()  # using multicrop, save the last one
            best_inputs = img_process.denormalize(best_inputs)
            save_images(args, best_inputs, targets, ipc_id)


def main_syn(ipc_id, args):

    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    model_teacher = models.__dict__[args.arch_name](num_classes=200)
    model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model_teacher.maxpool = nn.Identity()
    checkpoint = torch.load(args.arch_path, map_location="cpu")
    model_teacher.load_state_dict(checkpoint["model"])

    model_teacher = nn.DataParallel(model_teacher).cuda()
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    if args.verifier:
        model_verifier = models.__dict__[args.verifier_arch](num_classes=200)
        model_verifier.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model_verifier.maxpool = nn.Identity()
        checkpoint = torch.load(args.verifier_arch_path, map_location="cpu")
        model_verifier.load_state_dict(checkpoint["model"])

        model_verifier = model_verifier.cuda()
        model_verifier.eval()
        for p in model_verifier.parameters():
            p.requires_grad = False
        hook_for_display = lambda x, y: validate(x, y, model_verifier)
    else:
        hook_for_display = None

    get_images(args, model_teacher, hook_for_display, ipc_id)


if __name__ == '__main__':
    args = parse_args()
    wandb_init(args)

    for ipc_id in range(args.ipc_start, args.ipc_end):
        print(f'ipc = {ipc_id}')
        main_syn(ipc_id, args)

    wandb.finish()