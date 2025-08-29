# LM-UMamba
<img width="3250" height="869" alt="Fig1" src="https://github.com/user-attachments/assets/b2be4d4a-3ea9-4d89-af81-c460df3a37c2" />
Overveiw of LM-UMamba:LM-UMamba consists of global-local feature encoding (GLFE) and efficient feature decoding (EFD) to achieve high-quality large-hole image inpainting. Specifically, we introduce the multi-branch selective scan module (MSSM) and lightweight multi-scale strip convolutions (LMSC) into GLFE. Here, MSSM combines a four-way scanning mechanism with multi-scale convolutions to model robust global features by effectively capturing global contexts. LMSC employs depth-wise separable strip convolutions to aggregate local features, thereby enhancing local modeling. Additionally, within EFD, we integrate the lightweight Convolutional Decoder (LCD) and the multi-hierarchical feature fusion block (MHFB) to generate more discriminative decoded features, which help our model produce inpainted results with consistent structures and coherent textures.

# Environment setup
Clone the repo:
```python
  git clone https://github.com/csust-yc/LM-UMamba
  cd LM-UMamba
  conda create -n LMUMamba python=3.8
  conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
  pip install -r requirements.txt
```


# Dataset
For the full CelebA-HQ dataset, please refer to http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

For the full Places2 dataset, please refer to http://places2.csail.mit.edu/download.html

For the full mask dataset we used are contained in coco_mask_list.txt and irregular_mask_list.txt,you can download them from https://github.com/ewrfcas/MST_inpainting.


# Pre-trained model
We released the pre-trained model [CelebA-HQ](https://pan.baidu.com/s/1yigVwq7HMo9n_OlLnDIlyw?pwd=xpqq) and [Places2](https://pan.baidu.com/s/1ZpiGWRvr9DPO7zCutXwwsg?pwd=sjsd)

# Getting Started
[Download pre-trained model] Download the pre-trained model to ./ckpt

[Data Preparation] Download the Datasets, and create a list file (.txt) for training and testing. Next, configure the respective parameters in the config.yml file.

run:
```python
  python -u train.py --nodes 1 --gpus 1 --GPU_ids '0' --path ./ckpt/celeba --config_file ./config_list/config.yml --lama --DDP
```

eval:
``` python
  python -u inference.py --nodes 1 --gpus 1 --GPU_ids '0' --path ./ckpt/celeba --config_file ./config_list/config.yml
```
  
# Acknowledgments
This repo is built upon [MambaIR](https://github.com/csguoh/MambaIR), [ZITS](https://github.com/DQiaole/ZITS_inpainting?tab=readme-ov-file)and [LaMa](https://github.com/advimman/lama).
