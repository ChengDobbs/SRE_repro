wandb disabled
iter="2k"
ipcn=10
CUDA_VISIBLE_DEVICES=0 \
python 'validate/make_softlabels.py' \
    --train-dir syn_data/in1k_rn18_awp_${iter}_ipc${ipcn}_v7 \
    --batch-size 1024 \
    --teacher-model resnet18 \
    --cos -T 20 -j 4 \
    --ipc ${ipcn} \
    --mix-type cutmix \
    --output-dir save/in1k_rn18_awp_${iter}_ipc${ipcn} \
    --save-soft-labels