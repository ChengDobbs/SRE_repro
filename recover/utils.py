import os
import numpy as np
import torch
from PIL import Image

import wandb
from datetime import datetime


class ImageProcessor:
    def __init__(self, dataset: str):
        if dataset == 'cifar100':
            self.mean = np.array([0.4914, 0.4822, 0.4465])
            self.std = np.array([0.2023, 0.1994, 0.2010])
        elif dataset == 'tinyIN':
            self.mean = np.array([0.4802, 0.4481, 0.3975])
            self.std = np.array([0.2302, 0.2265, 0.2262])
        elif dataset == 'imagenet':
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
    
    def clip(self, x):
        """
        clip the input image
        """
        lower_bound = (0 - self.mean) / self.std
        upper_bound = (1 - self.mean) / self.std
        
        for c in range(3):
            x[:, c] = torch.clamp(x[:, c], lower_bound[c], upper_bound[c])

        return x
    
    def normalize(self, x):
        """
        normalize the input image
        """
        for c in range(3):
            x[:, c] = (x[:, c] - self.mean[c]) / self.std[c]

        return x

    def denormalize(self, x):
        """
        denormalize back to original input
        """
        for c in range(3):
            x[:, c] = torch.clamp(x[:, c] * self.std[c] + self.mean[c], 0, 1)

        return x

    def inverse_tanh_space(self, x):
        """
        inverse tanh space
        """
        return self.atanh(torch.clamp(x * 2 - 1, -1, 1))
    
    def tanh_space(self, x):
        """
        tanh space
        """
        return (torch.tanh(x) + 1) / 2
    
    def atanh(self, x):
        """
        atanh transform
        """
        return torch.log((1 + x) / (1 - x)) / 2


class BNFeatureHook:
    def __init__(self, module, args):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.r_var = args.r_var

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) * self.r_var + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05,dropout =0, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.base_optimizer = base_optimizer
        for group in self.param_groups:
            group["rho"] = rho
        self.dropout = dropout

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                # __import__('pdb').set_trace()
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        1.0 * p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def save_images(args, images, targets, ipc_id):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store = dir_path + '/class{:03d}_id{:03d}.jpg'.format(class_id, ipc_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def validate(input, target, model):
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def wandb_init(args)->None:
    date_time = datetime.now().strftime("%m/%d, %H:%M:%S")
    wandb.init(
        project = args.wandb_project, 
        name = args.wandb_name + "_" + date_time,
        group = args.wandb_group,
        job_type = args.wandb_job_type
    )
