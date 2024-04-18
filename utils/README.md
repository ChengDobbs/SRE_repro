# SRe2L_repro


## New server cmds (48h usage once)
```{bash}
# for zh-cn env
curl -ksSL https://download.weakptr.com/linux/install.sh | bash

pigchacli

export http_proxy=http://127.0.0.1:15777 https_proxy=http://127.0.0.1:15777
curl google.com -v
```
### Huggingface Mirror 
```{bash}
pip install huggingface_hub[hf_transfer]
vim ~/.bashrc
# (for zh-cn env)
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT=https://hf-mirror.com
source ~/.bashrc
```
#### Upload syn_data via huggingface-cli
```{bash}

huggingface-cli upload --repo-type=dataset --commit-message 'Upload in1k_rn18_4k_ipc50_4090x2' Ortho_SRe2L files_to_upload_path/file_format_supported.pth
```

### OSError (libmpi.so.40)
```{bash}
apt purge hwloc-nox libhwloc-dev libhwloc-plugins
```

### github account init
```{bash}
git config --global user.name "
git config --global user.email "
```

### <s>github large file system (lfs)</s>
```{bash}
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git-lfs install

git clone https://github.com/ChengDobbs/SRe2L_repro.git

cd SRe2L_repro
# check again with ~/SRe2L_repro# prefix
git lfs install
git lfs track "*.pth"
git add .gitattributes
```
### Huggingface dataset upload/downLoad
```{bash}
pip install datasets huggingface-cli huggingface-hub[hf_transfer]

git config --global credential.helper store
huggingface-cli login

huggingface-cli upload --repo-type dataset --commit-message 'Upload xxx' Orxxxxxxx tobeupload.tar.gz

wget (*resolve* exclude'?download')
```


### WandB init
```{bash}
pip install wandb
wandb login
# go to https://wandb.ai/authorize to copy and paste
```

### load pretrained models
```{bash}
git lfs checkout save/cifar100/resnet18_E200/ckpt.pth
```

### Exec Method 

```{bash}
cat run.sh
sh run.sh
```

### mount perminent local storage
```{bash}
cd /opt/data/private
```

### metrics

cifar100

imagenet1K
