# TDPM-AEOT
PyTorch implementation of TDPM-AEOT   
* training AE-OT can refer to [AE-OT](https://github.com/icemiliang/pyvot?utm_source=catalyzex.com)
**** 
## training our model   
python main_DM_cifar-2    --train
****
## Sample via this model
### **First, download pretrained model**
#### Download related the pre-trained model and files
* CIFAR10 pre-trained model  
[CIFAR10 Auto-Encoder and trained OT](https://drive.google.com/drive/folders/16d5L4ZWeDOZ49OMGFPnivnHsl26HjmlR)  
[CIFAR10 pre-trained truncated diffusion model](https://drive.google.com/drive/folders/1wmsSvvo_zl1AWCNa2WcHNRN9yOI0N8SC)  
Download the above pre-trained model to folder **cifar10-2**, **DMAE_cifar10-2-100**.
****    
* CelebA pre-trained model  
[CelebA Auto-Encoder and trained OT](https://drive.google.com/drive/folders/1gqoRFw6xBwMPBwbhim0wB4M9Skc1UJMg)  
[CelebA pre-trained truncated diffusion model](https://drive.google.com/drive/folders/1dSU-StVGXY0NUWTlvCqi6-k1f1yw3ImM)  
Download the above pre-trained model to folder **CelebA-2**, **DMAE_CelebA-2-100**.
****    
* CelebA-HQ pre-trained model  
[CelebA-HQ Auto-Encoder and trained OT](https://drive.google.com/drive/folders/1hwB5obWjquFOgw-YRDOMpxBxYC3GkomF)  
[CelebA-HQ pre-trained truncated diffusion model](https://drive.google.com/drive/folders/10ljbVmvXb_h3MOEPyaOrdQFP39VZdkXl)  
Download the above pre-trained model to folder **CelebA-HQ-2**, **DMAE_CelebA-HQ-150**.
****    
* Download precalculated statistic for dataset:  
[stats](https://drive.google.com/drive/folders/1_6dj0O20vXyW4rAAL97D-41rbfMy_BDd)
**** 
### **Secondly, sampling**  
python main_DM_cifar-2    --sample  
python main_DM_celeba-2    --sample  
python main_DM_CelebAHQ-2    --sample
****

## This implementation is based on / inspired by
https://github.com/icemiliang/pyvot?utm_source=catalyzex.com  
https://github.com/w86763777/pytorch-ddpm
