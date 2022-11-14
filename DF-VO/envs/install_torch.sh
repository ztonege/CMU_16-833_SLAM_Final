# Install torch with this command because we are using cuda 11.4
# https://discuss.pytorch.org/t/nvidia-a100-gpu-runtimeerror-cudnn-error-cudnn-status-mapping-error/121648/2
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Also checkout this issue and make proper modifications
# https://github.com/Huangying-Zhan/DF-VO/issues/13#issuecomment-702014660