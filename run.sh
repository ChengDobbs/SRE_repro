folder_path="SRE2L3.0" 

ipcn=10
iter="2k"


CUDA_VISIBLE_DEVICES=0 \
python 'recover/recover_in1k.py' \
    --wandb-project $folder_path \
    --wandb-group syn_2k3k \
    --wandb-job-type recover \
    --wandb-name in1k_rn18_${iter}_ipc${ipcn} \
    --exp-name in1k_rn18_${iter}_ipc${ipcn} \
    --batch-size 100 \
    --lr 0.25 \
    --iteration 2000 \
    --r-bn 0.01 \
    --store-best-images \
    --ipc-start 0 \
    --ipc-end ${ipcn}


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


CUDA_VISIBLE_DEVICES=0 \
python 'validate/train_KD.py' \
    --wandb-project $folder_path \
    --wandb-group syn_2k3k \
    --wandb-job-type val_KD \
    --wandb-name in1k_val_KD_${iter}_ipc${ipcn} \
    --batch-size 128 \
    --gradient-accumulation-steps 1 \
    --teacher-model resnet18 \
    --model resnet18 \
    --cos -T 20 -j 4 \
    --ipc ${ipcn} \
    --mix-type 'cutmix' \
    --train-dir syn_data/in1k_rn18_${iter}_ipc${ipcn} \
    --val-dir data/imagenet/val \
    --output-dir save/val_kd_in1k/in1k_rn18_${iter}_ipc${ipcn}


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
