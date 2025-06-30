Mamba-CLIP
Introduction
Mamba-CLIP is a project that integrates the Mamba State Space Model (SSM) with CLIP for enhanced vision-language processing. This repository provides the necessary setup and configuration to utilize Mamba-CLIP for tasks such as image segmentation and related applications. Leveraging the efficiency of SSMs, Mamba-CLIP aims to capture long-range contextual dependencies with linear computational complexity, making it suitable for advanced vision tasks.
Setup and Configuration
1. Environment Setup
To set up the environment for Mamba-CLIP, follow these steps to create a virtual environment and install the required dependencies.
# Create a new conda environment with Python 3.8
conda create -n mamba_clip python=3.8
conda activate mamba_clip

# Install PyTorch and related packages
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# Install additional dependencies
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs

Note: The .whl files for causal_conv1d and mamba_ssm can be downloaded from Baidu or Google Drive.
2. Dataset Preparation
ISIC Datasets

Download the ISIC17 and ISIC18 datasets from Baidu or Google Drive.
Place the datasets in the following directory structure:
./data/isic17/
train/
images/
.png


masks/
.png




val/
images/
.png


masks/
.png






Similarly for ./data/isic18/.



Synapse Dataset

Download the Synapse dataset following the instructions from Swin-UNet or from Baidu.
Organize the dataset in the following structure:
./data/Synapse/
lists/list_Synapse/
all.lst
test_vol.txt
train.txt


test_vol_h5/
casexxxx.npy.h5


train_npz/
casexxxx_slicexxx.npz







3. Pre-trained Weights

Download the pre-trained weights for Mamba-CLIP from Baidu or Google Drive.
Store the weights in the ./pretrained_weights/ directory.

4. Training
To train the Mamba-CLIP model, navigate to the project directory and run the appropriate training script:
cd OOD-IMD
python train.py  # Train and test on ISIC17 or ISIC18 datasets
python train_synapse.py  # Train and test on Synapse dataset

Inference Testing:To use a trained checkpoint for inference and save test images:

In the config_setting file:
Set only_test_and_save_figs to True.
Specify the path to the trained checkpoint in best_ckpt_path.
Define the save path for test images in img_save_path.


Run the train.py script.

5. Output Results
After training, the results will be saved in the ./results/ directory.
Acknowledgments
This project builds upon open-source contributions from the VMamba and Swin-UNet repositories.
