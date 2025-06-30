# Mamba-CLIP

This repository provides the implementation of Mamba-CLIP, which integrates the Mamba State Space Model (SSM) with CLIP for enhanced vision-language processing. The project focuses on medical image segmentation tasks, leveraging the efficiency of SSMs to capture long-range contextual dependencies with linear computational complexity.


## Abstract

Mamba-CLIP combines the strengths of State Space Models with CLIP's vision-language capabilities for advanced image processing tasks. By utilizing the efficient Mamba architecture, this approach maintains linear computational complexity while effectively modeling long-range interactions, making it particularly suitable for medical image segmentation applications.

## 0. Main Environments

```bash
conda create -n mamba_clip python=3.8
conda activate mamba_clip
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

The .whl files of causal_conv1d and mamba_ssm can be downloaded from [Baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k) or [GoogleDrive](https://drive.google.com/drive/folders/1tZGs1YFHiDrMa-MjYY8ZoEnCyy7m7Gaj?usp=sharing).

## 1. Prepare the dataset

### ISIC datasets

- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here [Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) or [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing).

- After downloading the datasets, place them into './data/isic17/' and './data/isic18/', following this file structure:
- ./data/isic17/
  - train/
    - images/
      - .png
    - masks/
    - .png
  - val/
  - images/
    - .png
  - masks/
  - .png

### Synapse datasets

- Download the Synapse dataset from [Baidu](https://pan.baidu.com/s/1JCXBfRL9y1cjfJUKtbEhiQ?pwd=9jti) or follow the instructions from [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet).

- After downloading, place the dataset into './data/Synapse/' with the following structure:
- ./data/Synapse/
  - lists/
    - list_Synapse/
    - all.lst
    - test_vol.txt
    - train.txt
  - test_vol_h5/
    - case.npy.h5
  - train_npz/
  - case_slice.npz

## 2. Prepare the pre-trained weights

- Download the pre-trained weights from [Baidu](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy) or [GoogleDrive](https://drive.google.com/drive/folders/1tZGs1YFHiDrMa-MjYY8ZoEnCyy7m7Gaj?usp=sharing).
- Store the pre-trained weights in './pretrained_weights/'.

## 3. Train the Mamba-CLIP

```bash
cd OOD-IMD
python train.py          # Train and test on ISIC17 or ISIC18 datasets
python train_synapse.py  # Train and test on Synapse dataset
```

### Inference Testing

For testing with a trained checkpoint and saving test images:

- **In `config_setting`**:
  - Set `only_test_and_save_figs` to `True`
  - Specify the trained checkpoint path in `best_ckpt_path`
  - Define the save path for test images in `img_save_path`

- **Execute**: Run `python train.py`

## 4. Obtain the outputs

After training, results will be saved in './results/'

## 5. Acknowledgments

We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba) and [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) for their open-source contributions.
