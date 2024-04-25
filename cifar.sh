CUDA_VISIBLE_DEVICES=0
folder_path="SRE2L3.0" 
ipcn=50

python 'pretrain/squeeze_cifar100.py' \
    --epochs 100 \
    --output-dir save/cifar100/resnet18_e100

python 'recover/recover_cifar100.py' \
    --wandb-project $folder_path \
    --wandb-group CF100 \
    --wandb-job-type recover \
    --wandb-name SRE2L_syn_cf100 \
    --exp-name cifar100_rn18_1k_mobile_ipc${ipcn} \
    --arch-path save/cifar100/resnet18_E200/ckpt.pth \
    --dataset cifar100 \
    --arch-name resnet18 \
    --batch-size 100 \
    --lr 0.25 \
    --iteration 1000 \
    --r-bn 0.01 \
    --store-best-images \
    --ipc-start 0 \
    --ipc-end 50

python 'relabel/relabel_cifar100.py' \
    --wandb-project $folder_path \
    --wandb-group CF100 \
    --wandb-job-type relabel \
    --wandb-name SRE2L_relabel_cf100 \
    --epochs 800 \
    --batch-size 1024 \
    --dataset cifar100 \
    --output-dir save/post_cifar100/ipc50/E800_4090 \
    --syn-data-path syn_data/cifar100_rn18_1k_mobile_ipc50 \
    --teacher-path save/cifar100/resnet18_E200/ckpt.pth \
    --ipc 50 \
    --wandb-project $folder_path \
    --wandb-name ${wandb_name}
