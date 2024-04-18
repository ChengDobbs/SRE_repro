import os
import numpy as np
from PIL import Image

import wandb
from datetime import datetime

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

def cifar_mixup(images, alpha=0.8):
    """
    Returns mixed inputs
    For image augment during training on CIFAR-100
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size()[0]
    rand_index = torch.randperm(batch_size).cuda()
    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images

class MixAugmentation():
    def __init__(self, args):
        self.args = args

    def rand_bbox(self, size, lam):
        W, H = size[-2:]
        cut_rat = np.sqrt(1. - lam)
        half_cut_w = int(W * cut_rat * .5)
        half_cut_h = int(H * cut_rat * .5)

        # randomly choose bbox center
        cx = np.random.randint(half_cut_w, W - half_cut_w)
        cy = np.random.randint(half_cut_h, H - half_cut_h)

        bbx1 = np.clip(cx - half_cut_w, 0, W)
        bbx2 = np.clip(cx + half_cut_w, 0, W)
        bby1 = np.clip(cy - half_cut_h, 0, H)
        bby2 = np.clip(cy + half_cut_h, 0, H)

        return bbx1, bby1, bbx2, bby2

    # raw code
    # def rand_bbox(size, lam):
    #     W = size[2]
    #     H = size[3]
    #     cut_rat = np.sqrt(1. - lam)
    #     cut_w = int(W * cut_rat)
    #     cut_h = int(H * cut_rat)

    #     # uniform
    #     cx = np.random.randint(W)
    #     cy = np.random.randint(H)

    #     bbx1 = np.clip(cx - cut_w // 2, 0, W)
    #     bby1 = np.clip(cy - cut_h // 2, 0, H)
    #     bbx2 = np.clip(cx + cut_w // 2, 0, W)
    #     bby2 = np.clip(cy + cut_h // 2, 0, H)

    #     return bbx1, bby1, bbx2, bby2

    
    def mixup(self, images, args, rand_index=None, lam=None):
        if args.mode == 'fkd_save':
            batch_size = images.size()[0]
            rand_index = torch.randperm(batch_size).cuda()
            lam = np.random.beta(args.mixup, args.mixup)
        elif args.mode == 'fkd_load':
            assert rand_index is not None and lam is not None
            rand_index = rand_index.cuda()
            lam = lam
        else: raise ValueError('mode should be fkd_save or fkd_load')
        mixed_images = lam * images + (1 - lam) * images[rand_index]
        return mixed_images


    def cutmix(self, images, args, rand_index=None, lam=None, bbox=None):
        if args.mode == 'fkd_save':
            rand_index = torch.randperm(images.size()[0]).cuda()
            lam = np.random.beta(args.cutmix, args.cutmix)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
        elif args.mode == 'fkd_load':
            assert rand_index is not None and lam is not None and bbox is not None
            rand_index = rand_index.cuda()
            lam = lam
            bbx1, bby1, bbx2, bby2 = bbox
        else:
            raise ValueError('mode should be fkd_save or fkd_load')
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        # return images, rand_index.cpu(), lam, [bbx1, bby1, bbx2, bby2]
        return images

    def mix_aug(self, images, args, rand_index=None, lam=None, bbox=None):

        if args.mix_type == 'mixup':
            return self.mixup(images, args, rand_index, lam)
        elif args.mix_type == 'cutmix':
            return self.cutmix(images, args, rand_index, lam, bbox)
        else:
            return images


class RandomResizedCropWithCoords(transforms.RandomResizedCrop):
    def __init__(self, **kwargs):
        super(RandomResizedCropWithCoords, self).__init__(**kwargs)

    def __call__(self, img, coords):
        try:
            reference = (coords.any())
        except:
            reference = False
        if not reference:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            coords = (i / img.size[1],
                      j / img.size[0],
                      h / img.size[1],
                      w / img.size[0])
            coords = torch.FloatTensor(coords)
        else:
            i = coords[0].item() * img.size[1]
            j = coords[1].item() * img.size[0]
            h = coords[2].item() * img.size[1]
            w = coords[3].item() * img.size[0]
        return F.resized_crop(img, i, j, h, w, self.size,
                                 self.interpolation), coords
    
class RandomHorizontalFlipWithRes(nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, status):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        if status is not None:
            if status == True:
                return F.hflip(img), status
            else:
                return img, status
        else:
            status = False
            if torch.rand(1) < self.p:
                status = True
                return F.hflip(img), status
            return img, status


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class ComposeWithCoords(transforms.Compose):
    def __init__(self, **kwargs):
        super(ComposeWithCoords, self).__init__(**kwargs)

    def __call__(self, img, coords, status):
        for t in self.transforms:
            if type(t).__name__ == 'RandomResizedCropWithCoords':
                img, coords = t(img, coords)
            elif type(t).__name__ == 'RandomCropWithCoords':
                img, coords = t(img, coords)
            elif type(t).__name__ == 'RandomHorizontalFlipWithRes':
                img, status = t(img, status)
            else:
                img = t(img)
        return img, status, coords

class ImageFolder_FKD_MIX(ImageFolder):
    def __init__(self, fkd_path, mode, args_epoch=None, args_bs=None, **kwargs):
        self.fkd_path = fkd_path
        self.mode = mode
        super(ImageFolder_FKD_MIX, self).__init__(**kwargs)
        self.batch_config = None  # [list(coords), list(flip_status)]
        self.batch_config_idx = 0  # index of processing image in this batch
        if self.mode == 'fkd_load':
            max_epoch, batch_size, num_img = self.get_FKD_info(self.fkd_path)
            if args_epoch > max_epoch:
                raise ValueError(f'`--epochs` should be no more than max epoch.')
            if args_bs != batch_size:
                raise ValueError('`--batch-size` should be same in both saving and loading phase. Please use `--gradient-accumulation-steps` to control batch size in model forward phase.')
            # self.img2batch_idx_list = torch.load('/path/to/img2batch_idx_list.tar')
            self.img2batch_idx_list = self.get_img2batch_idx_list(num_img=num_img, batch_size=batch_size, epochs=max_epoch)
            self.epoch = None

    def __getitem__(self, index):
        path, target = self.samples[index]

        if self.mode == 'fkd_save':
            coords_ = None
            flip_ = None
        elif self.mode == 'fkd_load':
            if self.batch_config == None:
                raise ValueError('config is not loaded')
            assert self.batch_config_idx <= len(self.batch_config[0])

            coords_ = self.batch_config[0][self.batch_config_idx]
            flip_ = self.batch_config[1][self.batch_config_idx]

            self.batch_config_idx += 1
        else:
            raise ValueError('mode should be fkd_save or fkd_load')

        sample = self.loader(path)

        if self.transform is not None:
            sample_new, flip_status, coords_status = self.transform(sample, coords_, flip_)
        else:
            flip_status = None
            coords_status = None

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample_new, target, flip_status, coords_status

    # def set_epoch(self, epoch):
    #     self.epoch = epoch

    def get_FKD_info(self, fkd_path):
        def custom_sort_key(s):
            # Extract numeric part from the string using regular expression
            numeric_part = int(s.split('_')[1].split('.tar')[0])
            return numeric_part

        max_epoch = len(os.listdir(fkd_path))
        batch_list = sorted(os.listdir(os.path.join(
            fkd_path + 'epoch_0')), key=custom_sort_key)
        batch_size = torch.load(os.path.join(
            fkd_path, 'epoch_0', batch_list[0]))[1].size()[0]
        last_batch_size = torch.load(os.path.join(
            fkd_path, 'epoch_0', batch_list[-1]))[1].size()[0]
        num_img = batch_size * (len(batch_list) - 1) + last_batch_size

        print('======= FKD: dataset info ======')
        print('path: {}'.format(fkd_path))
        print('num img: {}'.format(num_img))
        print('batch size: {}'.format(batch_size))
        print('max epoch: {}'.format(max_epoch))
        print('================================')
        return max_epoch, batch_size, num_img

    def load_batch_config(self, img_idx):
        """Use the `img_idx` to locate the `batch_idx`

        Args:
            img_idx: index of the first image in this batch
        """
        assert self.epoch != None
        batch_idx = self.img2batch_idx_list[self.epoch][img_idx]
        batch_config_path =  os.path.join(self.fkd_path, 'epoch_{}'.format(self.epoch), 'batch_{}.tar'.format(batch_idx))

        # [coords, flip_status, mix_index, mix_lam, mix_bbox, soft_label]
        config = torch.load(batch_config_path)
        self.batch_config_idx = 0
        self.batch_config = config[:2]
        return config[2:]

    def get_img2batch_idx_list(self, num_img = 50000, batch_size = 1024, seed=42, epochs=300):
        train_dataset = torch.utils.data.TensorDataset(torch.arange(num_img))
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

        img2batch_idx_list = []
        for epoch in range(epochs):
            img2batch_idx = {}
            for batch_idx, img_indices in enumerate(batch_sampler):
                img2batch_idx[img_indices[0]] = batch_idx

            img2batch_idx_list.append(img2batch_idx)
        return img2batch_idx_list

# keep top k largest values, and smooth others
def keep_top_k(p,k,n_classes=1000): # p is the softmax on label output
    if k == n_classes:
        return p

    values, indices = p.topk(k, dim=1)

    mask_topk = torch.zeros_like(p)
    mask_topk.scatter_(-1, indices, 1.0)
    top_p = mask_topk * p

    minor_value = (1 - torch.sum(values, dim=1)) / (n_classes-k)
    minor_value = minor_value.unsqueeze(1).expand(p.shape)
    mask_smooth = torch.ones_like(p)
    mask_smooth.scatter_(-1, indices, 0)
    smooth_p = mask_smooth * minor_value

    topk_smooth_p = top_p + smooth_p
    assert np.isclose(topk_smooth_p.sum().item(), p.shape[0]), f'{topk_smooth_p.sum().item()} not close to {p.shape[0]}'
    return topk_smooth_p

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

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


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(
        params=group_no_weight_decay, weight_decay=0.)]
    return groups

def wandb_init(args)->None:
    date_time = datetime.now().strftime("%m/%d, %H:%M:%S")
    wandb.init(
        project = args.wandb_project, 
        name = args.wandb_name + "_" + date_time,
        group = args.wandb_group,
        job_type = args.wandb_job_type
    )

# class download_IN1K_images():
#     def __init__(self, path):
#         self.path = path
#         self.download()

#     def download(self):
#         if not os.path.exists(self.path):
#             os.makedirs(self.path, exist_ok = True)
#         os.system("cd data")
#         os.system("mkdir imagenet")
#         os.system("mkdir imagenet/val")
#         os.system("cd imagenet/val")
#         os.system("wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate")
#         os.system("tar -xzvf ILSVRC2012_img_val.tar")
#         os.system("rm ILSVRC2012_img_val.tar")
#         os.system("find . -name "*.JPEG" | wc -l") # 50000
#         os.system("wget -qO- https://raw.githubusercontent.com/ChengDobbs/SRe2L_repro/master/utils/createINClassFolder.sh | bash")
#         os.system("rm createINClassFolder.sh")
#         os.system("cd ../../..")

#     def __call__(self):
#         return self.path