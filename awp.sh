iter="2k"
ipcn=10


# python 'recover/recover_cifar100_awp_multisteps' \
#     --wandb-project $folder_path \
#     --wandb-name SRE2L_syn_cifar100_${device} \
#     --wandb-group syn_cifar100 \
#     --wandb-job-type recover \
#     --exp-name cifar100_rn18_1k_mobile_lr.25_bne.2 \
#     --arch-path save/cifar100/resnet18_E200/ckpt.pth \
#     --dataset cifar100 \
#     --arch-name resnet18 \
#     --batch-size 100 \
#     --lr 0.25 \
#     --iteration 1000 \
#     --r-bn 0.01 \
#     --store-best-images \
#     --ipc-start 0 \
#     --ipc-end 50


python 'recover/recover_in1k_awp.py' \
    --wandb-project $folder_path \
    --wandb-group syn_2k3k \
    --wandb-job-type recover_awp \
    --wandb-name in1k_rn18_${iter}_ipc${ipcn} \
    --exp-name in1k_rn18_${iter}_ipc${ipcn} \
    --batch-size 100 \
    --lr 0.25 \
    --iteration 2000 \
    --r-bn 0.01 \
    --store-best-images \
    --ipc-start 0 \
    --ipc-end ${ipcn}

# iter="2k"
# python 'recover/recover_in1k.py' \
#     --wandb-project $folder_path \
#     --wandb-group syn_2k3k \
#     --wandb-job-type recover \
#     --wandb-name in1k_rn18_${iter}_ipc${ipcn} \
#     --exp-name in1k_rn18_${iter}_ipc${ipcn} \
#     --batch-size 100 \
#     --lr 0.25 \
#     --iteration 2000 \
#     --r-bn 0.01 \
#     --store-best-images \
#     --ipc-start 0 \
#     --ipc-end ${ipcn}

iter="2k"
ipcn=10
python 'recover/recover_in1k_awp.py' \
    --wandb-project $folder_path \
    --wandb-group syn_2k3k \
    --wandb-job-type recover_awp \
    --wandb-name in1k_rn18_${iter}_ipc${ipcn} \
    --exp-name in1k_rn18_${iter}_ipc${ipcn} \
    --batch-size 100 \
    --lr 0.25 \
    --iteration 2000 \
    --r-bn 0.01 \
    --store-best-images \
    --ipc-start 0 \
    --ipc-end ${ipcn}

# python 'relabel/relabel_cifar100.py' \
#     --epochs 800 \
#     --batch-size 1024 \
#     --dataset 'cifar100' \
#     --output-dir 'save/post_cifar100/ipc50/E800_4090' \
#     --syn-data-path 'syn_data/cifar100_rn18_1k_mobile_ipc50' \
#     --teacher-path 'save/cifar100/resnet18_E200/ckpt.pth' \
#     --ipc 50 \
#     --wandb-project $folder_path \
#     --wandb-name ${wandb_name}


# python 'relabel/relabel_in1k.py' \
#     --epochs 300 \
#     --batch-size 1024 \
#     --dataset 'imagenet' \
#     --output-dir 'save/post_in1k/ipc50/E800_4090' \
#     --syn-data-path 'syn_data/in1k_rn18_4k_ipc50_4090' \
#     --teacher-path 'save/cifar100/resnet18_E200/ckpt.pth' \
#     --ipc 50 \
#     --wandb-project $folder_path \
#     --wandb-name "${wandb_name}"

# python 'validate/train_KD.py' \
#     --wandb-project $folder_path \
#     --wandb-name SRE2L_val_KD_${iter}_ipc${ipcn} \
#     --wandb-group syn_2k3k \
#     --wandb-job-type val_KD \
#     --batch-size 128 \
#     --gradient-accumulation-steps 1 \
#     --teacher-model resnet18 \
#     --model resnet18 \
#     --cos -T 20 -j 4 \
#     --ipc ${ipcn} \
#     --mix-type 'cutmix' \
#     --train-dir syn_data/in1k_rn18_${iter}_ipc${ipcn} \
#     --val-dir data/imagenet/val \
#     --output-dir save/val_kd_in1k/in1k_rn18_${iter}_ipc${ipcn}

# python 'validate/train_FKD.py' \
#     --wandb-project $folder_path \
#     --wandb-name ${wandb_name} \
#     --epochs 300 \
#     --batch-size 1024 \
#     --gradient-accumulation-steps 4 \
#     --model resnet18 \
#     --cos -T 20 -j 4 \
#     --mix-type 'cutmix' \
#     --train-dir syn_data/in1k_rn18_4k_ipc50 \
#     --val-dir data/imagenet/val \
#     --fkd-path data/FKD_cutmix_fp16/ \
#     --output-dir save/val_kd_in1k/r18_ipc50_it4k_3090
