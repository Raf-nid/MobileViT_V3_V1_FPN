# MobileViT_V3_V1_FPN

MobileViT-FPN for FMC Data Reconstruction
This repository contains a PyTorch implementation of a deep learning model designed for high-resolution reconstruction of Full Matrix Capture (FMC) data. The architecture combines the lightweight efficiency of MobileViT v3 with a Dynamic Feature Pyramid Network (FPN) and a PixelShuffle upsampling head to reconstruct amplitude signals from binary or sparse inputs.

The project is optimized for scientific data (e.g., ultrasound, NDT, or seismic data) stored in MATLAB formats, featuring custom loss functions like Normalized Cross-Correlation (NCC).
## Project structure
<br>
├── training_MobileUnet_V5.py # Main training script

├── mobilevit_v3_v1_Pixel2.py # Model architecture <br>

├── utils.py                  # Custom loss functions, metrics, and helper classes 

├── config_MobileUNET.py      # Configuration parameters 

└── README.md                 # Project documentation 

## Data Preparation
The model expects data in MATLAB (.mat) format.
Format: Each .mat file must contain:
  -FMC: The target Ground Truth (Float32).
  -Bin: The input binary mask or sparse data (Float32).
  
### Structure:
/path/to/dataset/ 

├── train/ 

│   ├── sample_001.mat 

│   ├── sample_002.mat 

│   └── ... 

└── valid/ 

    ├── sample_100.mat 
    
    └── ...

##Training
To start the training process, run the main script. Ensure you have updated the DATA_DIR paths in the configuration section of training_MobileUnet_V5.py.
by default : 
-batch_size: 2
-learning_rate: 2e-4
-image_size: (4096, 1024) - Adjust based on your input dimensions
-patch_size: (32, 32)
