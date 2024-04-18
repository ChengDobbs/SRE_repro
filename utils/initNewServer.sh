apt purge libhwloc-dev libhwloc-plugins
pip install datasets huggingface-cli huggingface-hub[hf_transfer] wandb

git clone https://github.com/ChengDobbs/SRe2L_repro.git
git config --global credential.helper store
huggingface-cli login

wandb login