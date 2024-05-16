wandb disabled
folder_path="SRE2L3.0" 

iter="2k"
ipcn=10

# CUDA_VISIBLE_DEVICES=0 \
# python 'recover/recover_in1k_awp.py' \
#     --wandb-project $folder_path \
#     --wandb-group syn_2k3k_awp \
#     --wandb-job-type recover_awp \
#     --wandb-name in1k_rn18_awp_${iter}_ipc${ipcn} \
#     --exp-name in1k_rn18_awp_${iter}_ipc${ipcn} \
#     --data-select-path data/imagenet/select/select_dataset \
#     --select-num 50 \
#     --batch-size 100 \
#     --lr 0.25 \
#     --iteration 2000 \
#     --r-bn 0.01 \
#     --store-best-images \
#     --ipc-start 0 \
#     --ipc-end ${ipcn}


CUDA_VISIBLE_DEVICES=0 \
python 'validate/train_KD.py' \
    --wandb-project $folder_path \
    --wandb-name SRE2L_valKD_awp_${iter}_ipc${ipcn} \
    --wandb-group syn_2k3k_awp \
    --wandb-job-type valKD_awp \
    --batch-size 128 \
    --gradient-accumulation-steps 1 \
    --teacher-model resnet18 \
    --model resnet18 \
    --cos -T 20 -j 4 \
    --ipc ${ipcn} \
    --mix-type 'cutmix' \
    --train-dir syn_data/in1k_rn18_awp_${iter}_ipc${ipcn}_v7 \
    --val-dir data/imagenet/val \
    --output-dir save/in1k_rn18_awp_${iter}_ipc${ipcn} \
    --read-soft-labels