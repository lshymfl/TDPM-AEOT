# TDPM-AEOT
PyTorch implementation of TDPM-AEOT   
* training AE-OT can refer to [AE-OT](https://github.com/icemiliang/pyvot?utm_source=catalyzex.com  )  
* training our model   
python main_DM_cifar-2    --train

# Download related the trained model and files


* Download precalculated statistic for dataset:  
[stats](https://drive.google.com/drive/folders/1_6dj0O20vXyW4rAAL97D-41rbfMy_BDd)

# Sample via this model
python main_DM_cifar-2    --sample  
python main_DM_celeba-2    --sample  
python main_DM_CelebAHQ-2    --sample

# This implementation is based on / inspired by
https://github.com/icemiliang/pyvot?utm_source=catalyzex.com  
https://github.com/w86763777/pytorch-ddpm
