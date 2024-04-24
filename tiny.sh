
folder_path="SRE2L3.0" 

bs=256
ep=50
python pretrain/squeeze_tiny.py \
    --wandb-project $folder_path \
    --wandb-group tiny_awp \
    --wandb-job-type squeeze \
    --wandb-name tiny_rn18_awp_bs${bs}_ep${ep} \
    --model resnet18 \
    --batch-size $bs \
    --epochs $ep \
    --opt sgd \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr \
    --lr-warmup-epochs 5 \
    --lr-warmup-method linear \
    --lr-warmup-decay 0.01 \
    --output-dir save/tiny_rn18_ep${ep}


# ipcn=50
# iter=4k
# python recover/recover_tiny_awp.py \
#     --wandb-project $folder_path \
#     --wandb-group tiny_awp \
#     --wandb-job-type recover_awp \
#     --wandb-name tiny_rn18_awp_${iter}_ipc${ipcn} \
#     --arch-name resnet18 \
#     --arch-path save/tiny_rn18_ep${ep}/checkpoint.pth \
#     --exp-name tiny_rn18_4k_ipc${ipcn} \
#     --syn-data-path ./syn_data \
#     --batch-size 100 \
#     --lr 0.1 \
#     --r-bn 1 \
#     --iteration 4000 \
#     --store-last-images \
#     --ipc-start 0 \
#     --ipc-end $ipcn


# python validate/train_kd.py \
#     --wandb-project $folder_path \
#     --wandb-group tiny_awp \
#     --wandb-job-type valKD_awp \
#     --wandb-name tiny_rn18_awp_${iter}_ipc${ipcn} \
#     --model resnet18 \
#     --teacher-model resnet18 \
#     --teacher-path /path/to/resnet18_E50/checkpoint.pth \
#     --batch-size 256 \
#     --epochs 100 \
#     --opt sgd \
#     --lr 0.2 \
#     --momentum 0.9 \
#     --weight-decay 1e-4 \
#     --lr-scheduler cosineannealinglr \
#     --lr-warmup-epochs 5 \
#     --lr-warmup-method linear \
#     --lr-warmup-decay 0.01 \
#     --syn-data-path syn_data/tiny_rn18_4k_ipc${ipcn} \
#     -T 20 \
#     --image-per-class $ipcn \
#     --output-dir save_kd/s18t18_t20_4k.ipc_$ipcn