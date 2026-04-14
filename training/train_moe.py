import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as func
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import config_MobileUNET as config

#from MobileUnet_V2 import MobileNetV2_unet
from mobilevit_v3_v1_MOE_Pixel2 import MobileViTv3_v1_dynamicFPN_MOE_Pixel2, MoELoss
from mobilevit_v3_v1_MOE3_Pixel2 import MobileViTv3_v1_dynamicFPN_MOE3_Pixel2, MoELoss2
from mobilevit_v3_v1_MOE4_Pixel2 import MobileViTv3_v1_dynamicFPN_MOE4_Pixel2
from mobilevit_v3_v1_MOEV2_Pixel2 import MobileViTv3_v1_dynamicFPN_MOEV2_Pixel2
import os
from os.path import dirname, join as pjoin
import scipy.io as sio
import mat73
from utils.utils import EarlyStopping, ncc, NCC_MSE_Loss, save_fmc_images, save_metrics_plots
import datetime
import h5py
from torch.utils.tensorboard import SummaryWriter
from matplotlib import cm
from mobilevit import MobileViT_dynamicFPN
from mobilevit_v2 import MobileViTv2_dynamicFPN


import glob
from torch.utils.data import Dataset, DataLoader
"""Mixture-of-Experts (MoE) training entry point with optional fixed seed."""


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set the seed for all GPUs
    # For deterministic results, set this flag (optional):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    torch.Generator(seed)
    
class MatDataset(Dataset):
    def __init__(self, directory, device='cuda:1'):
        self.files = sorted(glob.glob(os.path.join(directory, '*.mat')))
        self.filenames = [os.path.basename(f) for f in self.files] 
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        base = os.path.splitext(self.filenames[idx])[0]
        truncated = base[-8:] #+ '.mat'
        with h5py.File(filepath, 'r') as f:
            fmc = torch.tensor(f['FMC'][()].astype('float32'))
            bin = torch.tensor(f['Bin'][()].astype('float32'))
    
        # AJOUT
        fmc = fmc.permute(1, 0)  # corriger l'orientation
        bin = bin.permute(1, 0)
    
        fmc = fmc.unsqueeze(0).to(self.device)
        bin = bin.unsqueeze(0).to(self.device)
    
        return fmc, bin, truncated





def main():
    
    #seeds = [i for i in range(1,20)]
    # Set the seed for reproducibility
    #seed_value = 1000  # You can change this to try different seeds
    torch.backends.cudnn.benchmark = True  # cuDNN autotune for convolutions
    torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 matmul on Ampere+
    
    # Device (set after CUDA-related flags above)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    
    #print(f"Seed: {torch.initial_seed()}")
    # A few visual parameters for the figures
    size = 10
    font = {'family':'sans-serif',
            'sans-serif':'Times New Roman',
            'size' : size}
    plt.rc('font', **font)

    # Learning rate
    learning_rate=config.learning_rate
    learning_rate=np.multiply(learning_rate,1e-6)
    for iteration in range(1):
      # Date and time for the saving of the results
      now = datetime.datetime.now()
      date_str = now.strftime("%Y%m%d_%H%M%S")
    
      if iteration ==0:
          seed_value = 69738009
      else:
          seed_value = np.random.randint(1e6, int(1e9))
      #seed_value = 193157014
      set_seed(seed_value)
      print(seed_value)
  
      # Cycle thru all the selected learning rates
      
      #for seed_value in seeds:  # Iterate through all seeds
          #print(f"Training with seed: {seed_value}")
      #seed_value = 8296370483135186884
      #set_seed(seed_value)
      
      #scaler = torch.cuda.amp.GradScaler()
      for itrain in learning_rate:
  
          # Model and plot directories
          for truc in range(1):
              if truc == 0:
                  config.batch_size = 16
                  model_ici =f'MbViTPixel2_MOEV2_2E_XXSMALL4_noise00_FF8_Normal_MF_batchsize' + f'{config.batch_size}' +'_NW_W_BruitFixe_BruitDuet'
                  #UnetRed_NW_plus_Wedge_BruitFixe_BruitDuet_rd32prem_rd32last_50_50New_patch128
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Train_MF_FF8_Normal"
                  data_dir_valid = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Valid_MF_FF8_Normal"
                  reference = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Train/pyFMC_5MHz_NW_14_2.mat"
#                  data_dir3 = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/FMC_Bin_antoine_2927_3.mat"
              elif truc==1:
                  config.num_epochs = 200
                  config.batch_size = 2
                  model_ici =f'TEST_noise00_Loss2_MbViTPixel2_MOE_XXSMALL4_FF8_Normal_MF_batchsize' + f'{config.batch_size}' +'_NW_plus_Wedge_BruitFixe_BruitDuet'
                  #UnetRed_NW_plus_Wedge_BruitFixe_BruitDuet_rd32prem_rd32last_50_50New_patch128
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Train_test"
                  data_dir_valid = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Valid_test"
                  reference = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Train/pyFMC_5MHz_NW_14_2.mat"
#                  data_dir3 = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/FMC_Bin_antoine_2927_3.mat"           
              elif truc ==2:
                  config.batch_size = 16
                  config.num_epochs = 30
                  model_ici =f'UNETred_RR8_NewAmplitude' + f'{config.batch_size}' +'_NW_plus_Wedge_BruitFixe_BruitDuet_rd32prem_rd32last_50_50New'
                  #UnetRed_NW_plus_Wedge_BruitFixe_BruitDuet_rd32prem_rd32last_50_50New_patch128
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Train_RR8_New"
                  data_dir_valid = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Valid_RR8_New"
                  reference = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Train/pyFMC_5MHz_NW_14_2.mat"
#                  data_dir3 = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/FMC_Bin_antoine_2927_3.mat
              elif truc ==3:
                  config.batch_size = 16
                  config.num_epochs = 350
                  model_ici =f'MbViT4_Pixel2_FF8_normal' + f'{config.batch_size}' +'_NW_plus_Wedge_BruitFixe_BruitDuet_rd32prem_rd32last_50_50New'
                  #UnetRed_NW_plus_Wedge_BruitFixe_BruitDuet_rd32prem_rd32last_50_50New_patch128
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Train_FF8"
                  data_dir_valid = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Valid_FF8"
                  reference = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Train/pyFMC_5MHz_NW_14_2.mat"
              elif truc ==4:
                  model_ici ='Mobilevitv3V2_FPN_Bin_New_20x225_20x5_20x75_4real_NoShift'
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_Bin_New_20x225_20x5_20x75_4real_NoShift.mat')
              elif truc==5:
                  model_ici ='Mobilevitv3V1_FPN_batchsize2_Bin_New_20x225_20x5_20x75_4real_NoShift'
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_Bin_New_20x225_20x5_20x75_4real_NoShift.mat')              
#              elif truc ==6:
#                  model_ici ='Mobilevitv3V1_FPN_patch64_PWM_New_20x225_20x5_20x75_4real_NoShift'
#                  summary = f'./runs/'+model_ici
#                  writer = SummaryWriter(summary)
#                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
#                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
#                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
#                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
#                  # recuperation of FMC
#                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_PWM_New_20x225_20x5_20x75_4real_NoShift.mat')
#              elif truc ==7:
#                  model_ici ='Mobilevitv3V1_FPN_patch64_PWM_New_20x225_20x5_20x75_4real_Shift'
#                  summary = f'./runs/'+model_ici
#                  writer = SummaryWriter(summary)
#                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
#                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
#                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
#                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
#                  # recuperation of FMC
#                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_PWM_New_20x225_20x5_20x75_4real_Shift.mat')
#              elif truc ==8:
#                  model_ici ='Mobilevitv3V1_FPN_Bin_New_30x225_30x5_30x75_4real_NoShift'
#                  summary = f'./runs/'+model_ici
#                  writer = SummaryWriter(summary)
#                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
#                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
#                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
#                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
#                  # recuperation of FMC
#                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_Bin_New_30x225_30x5_30x75_4real_NoShift.mat')
#              elif truc==9:
#                  model_ici ='Mobilevitv3V1_FPN_Bin_New_30x225_30x5_30x75_4real_Shift'
#                  summary = f'./runs/'+model_ici
#                  writer = SummaryWriter(summary)
#                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
#                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
#                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
#                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
#                  # recuperation of FMC
#                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_Bin_New_30x225_30x5_30x75_4real_Shift.mat')              
#              elif truc ==10:
#                  model_ici ='Mobilevitv3V1_FPN_PWM_New_30x225_30x5_30x75_4real_NoShift'
#                  summary = f'./runs/'+model_ici
#                  writer = SummaryWriter(summary)
#                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
#                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
#                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
#                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
#                  # recuperation of FMC
#                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_PWM_New_30x225_30x5_30x75_4real_NoShift.mat')
#              elif truc ==11:
#                  model_ici ='Mobilevitv3V1_FPN_PWM_New_30x225_30x5_30x75_4real_Shift'
#                  summary = f'./runs/'+model_ici
#                  writer = SummaryWriter(summary)
#                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
#                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
#                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
#                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
#                  # recuperation of FMC
#                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_PWM_New_30x225_30x5_30x75_4real_Shift.mat')
              
              # Loading data
              
              
#              if config.Vmat == 7.3:
#                  FMC = mat73.loadmat(reference)['FMC']
#                  Bin = mat73.loadmat(reference)['Bin']
##                  FMC_v2 = mat73.loadmat(data_dir2)['FMC']
##                  Bin_v2 = mat73.loadmat(data_dir2)['Bin']
##                  FMC_v3 = mat73.loadmat(data_dir3)['FMC']
##                  Bin_v3 = mat73.loadmat(data_dir3)['Bin']
#              else :
#                  FMC = sio.loadmat(reference)['FMC']
#                  Bin = sio.loadmat(reference)['Bin']
#              
#              
##              FMC = np.concatenate([FMC, FMC_v2, FMC_v3], axis=2)  # (4096,1024,192)
##              Bin = np.concatenate([Bin, Bin_v2, Bin_v3], axis=2)
#              
##              if truc == 6 or truc == 7 :
##                  FMC = FMC[:6080,:,:]
##                  Bin = Bin[:6080,:,:]
#                 
#              # Reorganising the dimencion of the FMC and Binary matrices
#              FMC = np.transpose(FMC,(2,0,1))
#              Bin = np.transpose(Bin,(2,0,1))
#              
#              FMC = torch.from_numpy(FMC).float().to("cuda:1") 
#              FMC = torch.unsqueeze(FMC,1)    # optional: add channel dim at index 1
#              
#              Bin = torch.from_numpy(Bin).float().to("cuda:1")
#              Bin = torch.unsqueeze(Bin,1)    #idem
#              print(FMC.shape)
#              print(FMC.shape[2])
              
#              if config.Vmat == 7.3:
#                  FMC_valid = mat73.loadmat(data_dir_valid)['FMC']
#                  Bin_valid = mat73.loadmat(data_dir_valid)['Bin']
#              else :
#                  FMC_valid = sio.loadmat(data_dir_valid)['FMC']
#                  Bin_valid = sio.loadmat(data_dir_valid)['Bin']
#              FMC_valid = np.transpose(FMC,(2,0,1))
#              Bin_valid = np.transpose(Bin,(2,0,1))
#              
#              FMC_valid = torch.from_numpy(FMC).float().to("cuda:1") 
#              FMC_valid = torch.unsqueeze(FMC,1)    # optional: add channel dim at index 1
#              
#              Bin_valid = torch.from_numpy(Bin).float().to("cuda:1")
#              Bin_valid = torch.unsqueeze(Bin,1)    #idem
#              
             
              
              
              if truc == 0:
                Model = MobileViTv3_v1_dynamicFPN_MOEV2_Pixel2((4096, 64), 'xx_small4', 1000, (32,32),noise_std=0.1).to("cuda:1")
                Model = Model.to("cuda:1")
              #else :
                #Model = MobileViTv3_v1_dynamicFPN_MOE_Pixel2((4096, 64), 'x_small', 1000, (32,32),noise_std=0.0).to("cuda:1")
                #Model = Model.to("cuda:1")              
          
              
              #pretrained_path = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/MobileUnet_V2_modif/Test_End_seed_lr_MbViTPixel2_MOE_2e_XXSMALL4_FF8_Normal_MF_batchsize16_NW_plus_Wedge_BruitFixe_BruitDuet200e-6_20251115_112352_epochs_intermediaire_350_epochs_seed_69738009/epoch_75/Model.pth"
#              if os.path.exists(pretrained_path):
#                  Model.load_state_dict(torch.load(pretrained_path, map_location=device))
#                  print(f"Loaded pretrained weights from {pretrained_path}")
#              else:
#                  print(f"No pretrained checkpoint at {pretrained_path}; training from scratch.")
  
              
              # Creation of the Training and validation Dataset
#              Dataset=TensorDataset(FMC,Bin)
#              Dataset_valid = TensorDataset(FMC_valid,Bin_valid)
#              
#              Datsize = FMC.shape[0]       # Number of samples in the dataset
#              train_size = np.round(Datsize*(1-config.val_per/100)).astype('int') # Number of samples in the training dataset
#              val_size = Datsize-train_size.astype('int') # Number of samples in the validation dataset
              
            
              
              
#              DatasetTrain, DatasetVal=torch.utils.data.random_split(Dataset,[train_size, val_size])  # random train/val split
#              train_loader=DataLoader(dataset=DatasetTrain,batch_size=config.batch_size,shuffle=True,drop_last=True)
#              val_loader=DataLoader(dataset=DatasetVal,batch_size=config.batch_size,shuffle=True,drop_last=True)
              # Creation of the Loss function optimizer and scheduler
              
              train_dataset = MatDataset(directory=data_dir, device="cuda:1")
              val_dataset = MatDataset(directory=data_dir_valid, device="cuda:1")

              train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
              val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

              
              losscalc =  MoELoss() # Compute loss
              #losscalc = NCC_MSE_Loss()
               # Algo to actualize MM weight
              #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=100000, factor=0.5, verbose=True) # Allow to reduce the learning rate during the training (if patience is superior to config.num_epoch = disabled)
              optimizer = torch.optim.Adam(Model.parameters(),lr=itrain)
              #optimizer = torch.optim.AdamW(Model.parameters(), lr=itrain, weight_decay=5e-2)
              scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100000, factor=0.5)
      
              # A few parameters
              running_loss = []
              epoch_inc = []
              losses = []
              epoch_cnt = []
              avg_grad = []
              accuracy = []
              accuracy_loss = []
      
              val_losses = []
              val_running_loss = []
              val_accuracy_loss = []
              
              # MoE Loss tracking
              mse_losses = []
              balance_losses = []
              entropy_losses = []
              expert_usage = [[] for _ in range(3)]  # one list per expert (3 experts)
              
              val_mse_losses = []
              val_balance_losses = []
              val_entropy_losses = []
              val_expert_usage = [[] for _ in range(3)]
              
              # Separate MSE tracking (from losscalc return)
              mse_vals = []
              val_mse_vals = []
              
              # MoE Loss tracking per epoch (averages)
              mse_losses_epoch = []
              balance_losses_epoch = []
              entropy_losses_epoch = []
              expert_usage_epoch = [[] for _ in range(3)]
              
              val_mse_losses_epoch = []
              val_balance_losses_epoch = []
              val_entropy_losses_epoch = []
              val_expert_usage_epoch = [[] for _ in range(3)]
              
              # MSE per epoch (averages from losscalc return)
              mse_vals_epoch = []
              val_mse_vals_epoch = []
              
              # Early stopping definition and variables
              es = EarlyStopping(patience = config.patience)
              done = False
              done2 = False
              epoch = 0
      
              # Training
              pbar = tqdm(total = config.num_epochs, leave=True)
              #scaler = torch.cuda.amp.GradScaler(enabled=True)
              while epoch < config.num_epochs and not done:
                  
                  Rec = []
                  BinR = []
                  AmpR = []
      
                  Val = []
                  BinV = []
                  AmpV = []
      
                  epoch += 1
                  pbar.update(1)
                  
                  Model.train()
                  
                  losses = []
                  accuracy = []
                  
                  # Reset MoE tracking lists at start of each epoch
                  mse_losses = []
                  balance_losses = []
                  entropy_losses = []
                  expert_usage = [[] for _ in range(3)]
                  mse_vals = []
                  
                  cnt = 0
                 
                  loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True)
                  # Training Loop
                  for (amp, bin, fname) in loop:
                      amp, bin = amp.to("cuda:1"), bin.to("cuda:1")  # Deplacez les donnees sur le meme appareil
                      optimizer.zero_grad()
                      #with torch.cuda.amp.autocast(enabled=True):
                      recon, aux_stats = Model(bin)
                          # placeholder for pseudo-label / auxiliary loss if needed
                          #recon_resized = func.interpolate(recon, size=(amp.size(2),amp.size(3)), mode='bilinear', align_corners=False)
                          #print(amp.shape,recon.shape)     
                      loss, loss_stats, mse = losscalc(amp, recon, aux_stats)
      
                      loss.backward() 
                      optimizer.step()
                      #scaler.scale(loss).backward()
                      #scaler.step(optimizer)
                      #scaler.update()
      
                          
                      #if epoch==1:
                          #initloss=loss.item()
      
                      #losses = np.append(losses,loss.item()/initloss)
                      losses = np.append(losses,loss.item())
                      epoch_cnt = np.append(epoch_cnt,epoch)
                      acc_val = ncc(amp, recon).item()
                      accuracy.append(acc_val)
                      
                      # Track MoE loss components
                      mse_losses.append(loss_stats['mse'])
                      balance_losses.append(loss_stats['balance'])
                      entropy_losses.append(loss_stats['entropy'])
                      
                      # Track MSE from losscalc return
                      mse_vals.append(mse.item())
                      
                      # Track expert usage
                      for i in range(3):
                          if f'expert_{i}_usage' in loss_stats:
                              expert_usage[i].append(loss_stats[f'expert_{i}_usage'])
                  
                      #loop.set_description(f"Epoch {epoch} | File: {fname} | Loss: {loss.item():.4e} | NCC: {acc_val:.4f}")
                      loop.set_description(f"Epoch {epoch} | Loss: {loss.item():.4e} | MSE :{mse.item():.4e} | Balance :{loss_stats['balance']:.4e} | Entropy :{loss_stats['entropy']:.4e} | NCC: {acc_val:.4f}")
                      if cnt==0:
                          Rec = torch.squeeze(recon,1).detach().to('cpu').numpy()                 
                          AmpR = torch.squeeze(amp,1).detach().to('cpu').numpy()
                          BinR = torch.squeeze(bin,1).detach().to('cpu').numpy()                
                      
                      if cnt>0 and cnt<2:
                          Rec = np.append(Rec,torch.squeeze(recon,1).detach().to('cpu').numpy(),axis=0)
                          AmpR = np.append(AmpR,torch.squeeze(amp,1).detach().to('cpu').numpy(),axis=0)
                          BinR = np.append(BinR,torch.squeeze(bin,1).detach().to('cpu').numpy(),axis=0)
      
                      cnt += 1
                      avg_train_loss = np.mean(losses)
                      avg_train_accuracy = np.mean(accuracy)
          
                      # **TensorBoard: Log train loss & accuracy**
                      writer.add_scalar("Loss/Train", avg_train_loss, epoch)
                      writer.add_scalar("Accuracy/Train", avg_train_accuracy, epoch)
                      
                      # **TensorBoard: Log MoE loss components**
                      if mse_losses:
                          writer.add_scalar("MoE/MSE_Train", np.mean(mse_losses), epoch)
                          writer.add_scalar("MoE/Balance_Train", np.mean(balance_losses), epoch)
                          writer.add_scalar("MoE/Entropy_Train", np.mean(entropy_losses), epoch)
                          writer.add_scalar("MoE/MSE_Raw_Train", np.mean(mse_vals), epoch)
                          
                          # Log expert usage
                          for i in range(3):
                              if expert_usage[i]:
                                  writer.add_scalar(f"MoE/Expert_{i}_Usage_Train", np.mean(expert_usage[i]), epoch)
      
                  cnt = 0
      
                  # Validation Loop
                  with torch.no_grad():
                      
                      Model.eval()
                      val_losses = []
                      val_accuracy = []
                      
                      # Reset validation MoE tracking lists
                      val_mse_losses = []
                      val_balance_losses = []
                      val_entropy_losses = []
                      val_expert_usage = [[] for _ in range(3)]
                      val_mse_vals = []

      
                      for (ampV, binV,_) in val_loader:
                          ampV, binV = ampV.to("cuda:1"), binV.to("cuda:1")
                          #with torch.cuda.amp.autocast(enabled=True):
                          reconV, aux_stats_val = Model(binV)
                              #print(reconV)
                              #recon_resizedV = func.interpolate(reconV, size=(ampV.size(2),ampV.size(3)), mode='bilinear', align_corners=False)
                          loss_val, loss_stats_val, mse_val = losscalc(ampV, reconV, aux_stats_val)
      
                          #my_lr = scheduler.optimizer.param_groups[0]['lr']
                          my_lr = optimizer.param_groups[0]['lr']
      
                          #if epoch==1:
                              #initvalloss = loss_val.item()
      
                          #val_losses = np.append(val_losses,loss_val.item()/initvalloss)
                          val_losses = np.append(val_losses,loss_val.item())
                          val_accuracy.append(ncc(ampV, reconV).item())
                          
                          # Track validation MoE loss components
                          val_mse_losses.append(loss_stats_val['mse'])
                          val_balance_losses.append(loss_stats_val['balance'])
                          val_entropy_losses.append(loss_stats_val['entropy'])
                          
                          # Track validation MSE from losscalc return
                          val_mse_vals.append(mse_val.item())
                          
                          # Track validation expert usage
                          for i in range(3):
                              if f'expert_{i}_usage' in loss_stats_val:
                                  val_expert_usage[i].append(loss_stats_val[f'expert_{i}_usage'])
      
      
                          if cnt==0:
                              Val = torch.squeeze(reconV,1).detach().to('cpu').numpy()
                              AmpV = torch.squeeze(ampV,1).detach().to('cpu').numpy()
                              BinV = torch.squeeze(binV,1).detach().to('cpu').numpy()                
                      
                          if cnt>0 and cnt<2:
                              Val = np.append(Val,torch.squeeze(reconV,1).detach().to('cpu').numpy(),axis=0)
                              AmpV = np.append(AmpV,torch.squeeze(ampV,1).detach().to('cpu').numpy(),axis=0)
                              BinV = np.append(BinV,torch.squeeze(binV,1).detach().to('cpu').numpy(),axis=0)
                          avg_val_loss = np.mean(val_losses)
                          avg_val_accuracy = np.mean(val_accuracy)
              
                          # **TensorBoard: Log validation loss & accuracy**
                          writer.add_scalar("Loss/Val", avg_val_loss, epoch)
                          writer.add_scalar("Accuracy/Val", avg_val_accuracy, epoch)
                          
                          # **TensorBoard: Log validation MoE loss components**
                          if val_mse_losses:
                              writer.add_scalar("MoE/MSE_Val", np.mean(val_mse_losses), epoch)
                              writer.add_scalar("MoE/Balance_Val", np.mean(val_balance_losses), epoch)
                              writer.add_scalar("MoE/Entropy_Val", np.mean(val_entropy_losses), epoch)
                              writer.add_scalar("MoE/MSE_Raw_Val", np.mean(val_mse_vals), epoch)
                              
                              # Log validation expert usage
                              for i in range(3):
                                  if val_expert_usage[i]:
                                      writer.add_scalar(f"MoE/Expert_{i}_Usage_Val", np.mean(val_expert_usage[i]), epoch)
                          cnt += 1
                      

                          
                  SizeTrain = Rec.shape[0]
                  SizeVal = Val.shape[0]
       
       
                  
                  # Recording differents losses and managing the early stopping algo
                  if epoch % 1 == 0:
      
                      running_loss = np.append(running_loss,np.mean(losses))
                      val_running_loss = np.append(val_running_loss,np.mean(val_losses))  
                      accuracy_loss.append(np.mean(accuracy))
                      val_accuracy_loss.append(np.mean(val_accuracy))
                      
                      # Record MoE loss components (epoch averages)
                      if mse_losses:
                          mse_losses_epoch.append(np.mean(mse_losses))
                          balance_losses_epoch.append(np.mean(balance_losses))
                          entropy_losses_epoch.append(np.mean(entropy_losses))
                          
                          # Record expert usage averages
                          for i in range(3):
                              if expert_usage[i]:
                                  expert_usage_epoch[i].append(np.mean(expert_usage[i]))
                      
                      # Record MSE values from losscalc return
                      if mse_vals:
                          mse_vals_epoch.append(np.mean(mse_vals))
                      
                      if val_mse_losses:
                          val_mse_losses_epoch.append(np.mean(val_mse_losses))
                          val_balance_losses_epoch.append(np.mean(val_balance_losses))
                          val_entropy_losses_epoch.append(np.mean(val_entropy_losses))
                          
                          # Record validation expert usage averages
                          for i in range(3):
                              if val_expert_usage[i]:
                                  val_expert_usage_epoch[i].append(np.mean(val_expert_usage[i]))
                      
                      # Record validation MSE values from losscalc return
                      if val_mse_vals:
                          val_mse_vals_epoch.append(np.mean(val_mse_vals))

                      epoch_inc=np.append(epoch_inc,epoch)
                      losses = []
                      val_losses = []
      
                      if es(Model, val_running_loss[-1]):
                          
                          done = True
                          pbar.set_description(
                              f"Epoch: {epoch}, lr: {my_lr:.2E}, tloss: {running_loss[-1]:.4E}, vloss: {val_running_loss[-1]:.4E}, accuracy: {val_accuracy_loss[-1]+1:.3E},{es.status}" 
                          )
                          scheduler.step(val_running_loss[-1])

                          
                      else:
                          pbar.set_description(
                              f"Epoch: {epoch}, lr: {my_lr:.2E}, tloss: {running_loss[-1]:.4E}, vloss: {val_running_loss[-1]:.4E}, accuracy: {val_accuracy_loss[-1]:.3E}, {es.status}"
                          )
                          scheduler.step(val_running_loss[-1])
                          
                  if epoch==10 or epoch==2 or epoch %25 == 0:
                      epoch_dir = os.path.join(Plt_Dir_epoch, f"epoch_{epoch}")
                      os.makedirs(epoch_dir, exist_ok=True)
                      
                      
                      save_metrics_plots(epoch, running_loss, val_running_loss, accuracy_loss, val_accuracy_loss, os.path.join(epoch_dir, 'train_loss.png'), os.path.join(epoch_dir, 'train_accuracy.png'))
                      
                      # **Graphique MoE Loss Components**
                      if len(mse_losses_epoch) > 0:  # skip if empty
                          plt.figure(figsize=(15, 10))
                          
                          # Subplot 1: Loss components
                          plt.subplot(2, 3, 1)
                          plt.plot(np.arange(len(mse_losses_epoch)), mse_losses_epoch, 'b-', label='MSE Train', linewidth=2)
                          plt.plot(np.arange(len(val_mse_losses_epoch)), val_mse_losses_epoch, 'r-', label='MSE Val', linewidth=2)
                          plt.xlabel("Epoch")
                          plt.ylabel("MSE Loss")
                          plt.yscale("log")
                          plt.grid(True)
                          plt.legend()
                          plt.title("MSE Loss Evolution")
                          
                          # Subplot 2: Balance loss
                          plt.subplot(2, 3, 2)
                          plt.plot(np.arange(len(balance_losses_epoch)), balance_losses_epoch, 'g-', label='Balance Train', linewidth=2)
                          plt.plot(np.arange(len(val_balance_losses_epoch)), val_balance_losses_epoch, 'orange', label='Balance Val', linewidth=2)
                          plt.xlabel("Epoch")
                          plt.ylabel("Balance Loss")
                          plt.yscale("log")
                          plt.grid(True)
                          plt.legend()
                          plt.title("Load Balancing Loss")
                          
                          # Subplot 3: Entropy loss
                          plt.subplot(2, 3, 3)
                          plt.plot(np.arange(len(entropy_losses_epoch)), entropy_losses_epoch, 'purple', label='Entropy Train', linewidth=2)
                          plt.plot(np.arange(len(val_entropy_losses_epoch)), val_entropy_losses_epoch, 'brown', label='Entropy Val', linewidth=2)
                          plt.xlabel("Epoch")
                          plt.ylabel("Entropy Loss")
                          plt.yscale("log")
                          plt.grid(True)
                          plt.legend()
                          plt.title("Entropy Loss")
                          
                          # Subplot 4: Expert usage (Training)
                          plt.subplot(2, 3, 4)
                          colors = ['red', 'blue', 'green']
                          for i in range(3):
                              if expert_usage_epoch[i]:
                                  plt.plot(np.arange(len(expert_usage_epoch[i])), expert_usage_epoch[i], 
                                          color=colors[i], label=f'Expert {i}', linewidth=2)
                          plt.xlabel("Epoch")
                          plt.ylabel("Usage Fraction")
                          plt.grid(True)
                          plt.legend()
                          plt.title("Expert Usage (Training)")
                          
                          # Subplot 5: Expert usage (Validation)
                          plt.subplot(2, 3, 5)
                          for i in range(3):
                              if val_expert_usage_epoch[i]:
                                  plt.plot(np.arange(len(val_expert_usage_epoch[i])), val_expert_usage_epoch[i], 
                                          color=colors[i], label=f'Expert {i}', linewidth=2)
                          plt.xlabel("Epoch")
                          plt.ylabel("Usage Fraction")
                          plt.grid(True)
                          plt.legend()
                          plt.title("Expert Usage (Validation)")
                          
                          # Subplot 6: Expert usage comparison (latest epoch)
                          plt.subplot(2, 3, 6)
                          if expert_usage_epoch[0] and val_expert_usage_epoch[0]:
                              train_usage = [expert_usage_epoch[i][-1] if expert_usage_epoch[i] else 0 for i in range(3)]
                              val_usage = [val_expert_usage_epoch[i][-1] if val_expert_usage_epoch[i] else 0 for i in range(3)]
                              
                              x = np.arange(3)
                              width = 0.35
                              plt.bar(x - width/2, train_usage, width, label='Train', alpha=0.8)
                              plt.bar(x + width/2, val_usage, width, label='Val', alpha=0.8)
                              plt.xlabel("Expert")
                              plt.ylabel("Usage Fraction")
                              plt.title("Expert Usage Comparison (Latest)")
                              plt.xticks(x, [f'Expert {i}' for i in range(3)])
                              plt.legend()
                              plt.grid(True, alpha=0.3)
                          
                          plt.tight_layout()
                          plt.savefig(os.path.join(epoch_dir, f'moe_stats_epoch_{epoch}.png'), dpi=1200, bbox_inches='tight')
                          plt.close()
                      
                      torch.save(es.best_model, os.path.join(epoch_dir, "Model.pth"))
                      
                      
                      # Save metrics to a .mat file
                      mat_metrics_path = os.path.join(epoch_dir, f"metrics_epoch_{epoch}.mat")
                      
                      # Package MoE tensors for .mat export
                      moe_data = {}
                      if mse_losses_epoch:
                          moe_data.update({
                              "mse_losses_epoch": mse_losses_epoch,
                              "balance_losses_epoch": balance_losses_epoch,
                              "entropy_losses_epoch": entropy_losses_epoch,
                              "expert_usage_epoch": expert_usage_epoch,
                              "val_mse_losses_epoch": val_mse_losses_epoch,
                              "val_balance_losses_epoch": val_balance_losses_epoch,
                              "val_entropy_losses_epoch": val_entropy_losses_epoch,
                              "val_expert_usage_epoch": val_expert_usage_epoch
                          })
                      
                      # Append raw MSE values from losscalc
                      if mse_vals_epoch:
                          moe_data.update({
                              "mse_vals_epoch": mse_vals_epoch,
                              "val_mse_vals_epoch": val_mse_vals_epoch
                          })
                      
                      sio.savemat(mat_metrics_path, {
                          "epoch": epoch,
                          "running_loss": running_loss,
                          "val_running_loss": val_running_loss,
                          "accuracy_loss": accuracy_loss,
                          "val_accuracy_loss": val_accuracy_loss,
                          "epoch_inc": epoch_inc,
                          **moe_data
                      })

                      
                      if epoch%50==0 or epoch ==25 or epoch == 2:  
                          taille1 = min(SizeTrain,3)
                          taille2 = min(SizeVal,5)
                          for i in tqdm(range(taille1)):
                             
                              AmpR[i,:,:] = AmpR[i,:,:] / np.max(np.absolute(AmpR[i,:,:]))
                              Rec[i,:,:] = Rec[i,:,:] / np.max(np.absolute(Rec[i,:,:]))
                              
                              plt.figure(figsize=(3,5))
                              plt.imshow(AmpR[i,:,:])
                              plt.xlabel("Element axis")
                              plt.ylabel("Time Increment")
                              plt.set_cmap('seismic')
                              plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                              plt.xticks([0,512,1024])            
                              plt.clim(-1,1)
                              cbar=plt.colorbar()
                              cbar.set_label('Amplitude (linear)')
                              plt.savefig(os.path.join(epoch_dir, f'T_{i}_Amp.png'),dpi=1200,bbox_inches='tight')
                              plt.close()
                              
                              plt.figure(figsize=(3,5))
                              plt.imshow(Rec[i,:,:])
                              plt.xlabel("Element axis")
                              plt.ylabel("Time Increment")
                              plt.set_cmap('seismic')
                              plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                              plt.xticks([0,512,1024])            
                              plt.clim(-1,1)
                              cbar=plt.colorbar()
                              cbar.set_label('Amplitude (linear)')
                              plt.savefig(os.path.join(epoch_dir, f'T_{i}_Rec.png'),dpi=1200,bbox_inches='tight')
                              plt.close()
                  
                              Err=AmpR[i,:,:]-Rec[i,:,:]
                              Err=np.absolute(Err)
                  
                              plt.figure(figsize=(3,5))
                              plt.imshow(Err)
                              plt.xlabel("Element axis")
                              plt.ylabel("Time Increment")
                              plt.set_cmap('inferno')
                              plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                              plt.xticks([0,512,1024])            
                              plt.clim(0,1)
                              cbar=plt.colorbar()
                              cbar.set_label('Error (absolute)')
                              plt.savefig(os.path.join(epoch_dir, f'T_{i}_Error.png'),dpi=1200,bbox_inches='tight')
                              plt.close()
                  
                            #   plt.figure(figsize=(3,5))
                            #   plt.imshow(BinR[i,:,:])
                            #   plt.xlabel("Element axis")
                            #   plt.ylabel("Time Increment")
                            #   plt.set_cmap('seismic')
                            #   plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                            #   plt.xticks([0,512,1024])            
                            #   plt.clim(-1,1)
                            #   cbar=plt.colorbar()
                            #   cbar.set_label('Amplitude (binary)')
                            #   plt.savefig(os.path.join(epoch_dir, f'T_{i}_Bin.png'),dpi=1200,bbox_inches='tight')
                            #   plt.close()
                              
                          for i in tqdm(range(taille2)):
                  
                              AmpV[i,:,:] = AmpV[i,:,:] / np.max(np.absolute(AmpV[i,:,:]))
                              Val[i,:,:] = Val[i,:,:] / np.max(np.absolute(Val[i,:,:]))
                  
                              plt.figure(figsize=(3,5))
                              plt.imshow(AmpV[i,:,:])
                              plt.xlabel("Element axis")
                              plt.ylabel("Time Increment")
                              plt.set_cmap('seismic')
                              plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                              plt.xticks([0,512,1024])            
                              plt.clim(-1,1)
                              cbar=plt.colorbar()
                              cbar.set_label('Amplitude (linear)')
                              plt.savefig(os.path.join(epoch_dir, f'V_{i}_Amp.png'),dpi=1200,bbox_inches='tight')
                              plt.close()
                  
                              plt.figure(figsize=(3,5))
                              plt.imshow(Val[i,:,:])
                              plt.xlabel("Element axis")
                              plt.ylabel("Time Increment")
                              plt.set_cmap('seismic')
                              plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                              plt.xticks([0,512,1024])            
                              plt.clim(-1,1)
                              cbar=plt.colorbar()
                              cbar.set_label('Amplitude (linear)')
                              plt.savefig(os.path.join(epoch_dir, f'V_{i}_Rec.png'),dpi=1200,bbox_inches='tight')
                              plt.close()
                  
                              Err=AmpV[i,:,:]-Val[i,:,:]
                              Err=np.absolute(Err)
                  
                              plt.figure(figsize=(3,5))
                              plt.imshow(Err)
                              plt.xlabel("Element axis")
                              plt.ylabel("Time Increment")
                              plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                              plt.xticks([0,512,1024])            
                              plt.set_cmap('inferno')
                              plt.clim(0,1)
                              cbar=plt.colorbar()
                              cbar.set_label('Error (absolute)')
                              plt.savefig(os.path.join(epoch_dir, f'V_{i}_Error.png'),dpi=1200,bbox_inches='tight')
                              plt.close()
                  
                            #   plt.figure(figsize=(3,5))
                            #   plt.imshow(BinV[i,:,:])
                            #   plt.xlabel("Element axis")
                            #   plt.ylabel("Time Increment")
                            #   plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                            #   plt.xticks([0,512,1024])
                            #   plt.set_cmap('seismic')
                            #   plt.clim(-1,1)
                            #   cbar=plt.colorbar()
                            #   cbar.set_label('Amplitude (binary)')
                            #   plt.savefig(os.path.join(epoch_dir, f'V_{i}_Bin.png'),dpi=1200,bbox_inches='tight')
                            #   plt.close()
                              
                              
                          
      
                  #if epoch % 20 == 0 or epoch == 2:
                  
                      #writer.add_image("Train/Input", amp[0], epoch, dataformats='CHW')
                      #writer.add_image("Train/Recon", recon[0], epoch, dataformats='CHW')
                      #writer.add_image("Train/Error", torch.abs(amp - recon)[0], epoch, dataformats='CHW')
                      
                  for name, param in Model.named_parameters():
                      writer.add_histogram(f'{name}/weights', param, epoch)
                      if param.grad is not None:
                          writer.add_histogram(f'{name}/gradients', param.grad, epoch)
      
                      
                  plt.figure()
                  plt.plot(np.arange(epoch), running_loss,'b-', label='Training')
                  plt.plot(np.arange(epoch), val_running_loss,'g-', label='Validation')
                  plt.xlabel("Training step")
                  plt.ylabel("Loss")
                  #plt.yscale("log")
                  plt.legend()
                  plt.savefig('./RunningLoss/'+Plt_Dir+'.png',dpi=1200)
                  plt.close()
                      
                  colors = [
                          (1, 0, 0),        # Rouge
                          (0.5, 0, 0.13),   # Bordeaux
                          (0.58, 0, 0.83),  # Violet
                          (0.53, 0.81, 0.98), # Bleu clair
                          (0.5, 0.5, 0.5),  # Gris
                          (0, 0, 0),        # Noir
                          (1, 0.75, 0.8),   # Rose
                          (0.56, 0.93, 0.56), # Vert clair
                          (1, 1, 0),         # Jaune
                          (0, 1, 1)      #cyan
                      ]
                  plt.figure()
                  plt.plot(np.arange(epoch), accuracy_loss,'b-', label='Training')
                  plt.plot(np.arange(epoch), val_accuracy_loss,'g-', label='Validation')
                  plt.xlabel("Training step")
                  plt.ylabel("Accuracy")
                  plt.legend()
                  #plt.yscale("log")
                  plt.savefig('./RunningAccuracy/'+Plt_Dir+'.png',dpi=1200)
                  plt.close()   
                  
                  #writer.add_histogram(f'{name}/weights', param, epoch)
                  #writer.add_histogram(f'{name}/gradients', param.grad, epoch)    
                   
                  torch.cuda.empty_cache()
                  
#              dummy_input = torch.randn(1, 1, 1024, 4096).to("cuda:1")  # example dummy input for add_graph
#              writer.add_graph(Model, dummy_input)
#      
              SizeTrain = Rec.shape[0]
              SizeVal = Val.shape[0]

      
              os.mkdir(Plt_Dir)
              
              save_metrics_plots(epoch, running_loss, val_running_loss, accuracy_loss, val_accuracy_loss, os.path.join(Plt_Dir, 'train_loss.png'), os.path.join(Plt_Dir, 'train_accuracy.png'))
              
              
              plt.figure()
              plt.plot(np.arange(epoch), running_loss,'b-', label='Training')
              plt.plot(np.arange(epoch), min_train,'b--')
              plt.plot(np.arange(epoch), val_running_loss,'g-', label='Validation')
              plt.plot(np.arange(epoch), min_val,'g--')
              plt.xlabel("Training step")
              plt.ylabel("Loss")
              plt.yscale("log")
              plt.grid(True)
              plt.title(f'Minimal Train Loss : {np.min(running_loss):.4E} / Val Loss : {np.min(val_running_loss):.4E}')
              plt.legend()
              plt.savefig(os.path.join(Plt_Dir, f'train_loss.png'),dpi=1200)
              plt.close() 
              
              # **Graphique final MoE Loss Components**
              if len(mse_losses) > 1:  # need at least two points for trend
                  plt.figure(figsize=(15, 10))
                  
                  # Subplot 1: Loss components
                  plt.subplot(2, 3, 1)
                  plt.plot(np.arange(len(mse_losses)), mse_losses, 'b-', label='MSE Train', linewidth=2)
                  plt.plot(np.arange(len(val_mse_losses)), val_mse_losses, 'r-', label='MSE Val', linewidth=2)
                  plt.xlabel("Epoch")
                  plt.ylabel("MSE Loss")
                  plt.yscale("log")
                  plt.grid(True)
                  plt.legend()
                  plt.title("MSE Loss Evolution")
                  
                  # Subplot 2: Balance loss
                  plt.subplot(2, 3, 2)
                  plt.plot(np.arange(len(balance_losses)), balance_losses, 'g-', label='Balance Train', linewidth=2)
                  plt.plot(np.arange(len(val_balance_losses)), val_balance_losses, 'orange', label='Balance Val', linewidth=2)
                  plt.xlabel("Epoch")
                  plt.ylabel("Balance Loss")
                  plt.yscale("log")
                  plt.grid(True)
                  plt.legend()
                  plt.title("Load Balancing Loss")
                  
                  # Subplot 3: Entropy loss
                  plt.subplot(2, 3, 3)
                  plt.plot(np.arange(len(entropy_losses)), entropy_losses, 'purple', label='Entropy Train', linewidth=2)
                  plt.plot(np.arange(len(val_entropy_losses)), val_entropy_losses, 'brown', label='Entropy Val', linewidth=2)
                  plt.xlabel("Epoch")
                  plt.ylabel("Entropy Loss")
                  plt.yscale("log")
                  plt.grid(True)
                  plt.legend()
                  plt.title("Entropy Loss")
                  
                  # Subplot 4: Expert usage (Training)
                  plt.subplot(2, 3, 4)
                  colors = ['red', 'blue', 'green']
                  for i in range(3):
                      if expert_usage[i]:
                          plt.plot(np.arange(len(expert_usage[i])), expert_usage[i], 
                                  color=colors[i], label=f'Expert {i}', linewidth=2)
                  plt.xlabel("Epoch")
                  plt.ylabel("Usage Fraction")
                  plt.grid(True)
                  plt.legend()
                  plt.title("Expert Usage (Training)")
                  
                  # Subplot 5: Expert usage (Validation)
                  plt.subplot(2, 3, 5)
                  for i in range(3):
                      if val_expert_usage[i]:
                          plt.plot(np.arange(len(val_expert_usage[i])), val_expert_usage[i], 
                                  color=colors[i], label=f'Expert {i}', linewidth=2)
                  plt.xlabel("Epoch")
                  plt.ylabel("Usage Fraction")
                  plt.grid(True)
                  plt.legend()
                  plt.title("Expert Usage (Validation)")
                  
                  # Subplot 6: Expert usage comparison (latest epoch)
                  plt.subplot(2, 3, 6)
                  if expert_usage[0] and val_expert_usage[0]:
                      train_usage = [expert_usage[i][-1] if expert_usage[i] else 0 for i in range(3)]
                      val_usage = [val_expert_usage[i][-1] if val_expert_usage[i] else 0 for i in range(3)]
                      
                      x = np.arange(3)
                      width = 0.35
                      plt.bar(x - width/2, train_usage, width, label='Train', alpha=0.8)
                      plt.bar(x + width/2, val_usage, width, label='Val', alpha=0.8)
                      plt.xlabel("Expert")
                      plt.ylabel("Usage Fraction")
                      plt.title("Expert Usage Comparison (Latest)")
                      plt.xticks(x, [f'Expert {i}' for i in range(3)])
                      plt.legend()
                      plt.grid(True, alpha=0.3)
                  
                  plt.tight_layout()
                  plt.savefig(os.path.join(Plt_Dir, f'moe_stats_final.png'), dpi=1200, bbox_inches='tight')
                  plt.close()
              
              #plt.figure()
              #plt.plot(np.arange(epoch), running_loss,'b-', label='Training')
              #plt.plot(np.arange(epoch), min_train,'b--')
              #plt.plot(np.arange(epoch), val_running_loss,'g-', label='Validation')
              #plt.plot(np.arange(epoch), min_val,'g--')
              #plt.xlabel("Training step")
              #plt.ylabel("Loss")
              #plt.yscale("log")
              #plt.legend()
              #plt.savefig('./'+Plt_Dir+'/loss_notitle.png',dpi=1200)
              #plt.close()
              
              
      
              #np.savez('./'+Plt_Dir+'/loss.npz',epoch_inc=np.arange(epoch),running_loss=running_loss,val_running_loss=val_running_loss)
              #np.savez('./'+Plt_Dir+'/Train.npz',FMCT=AmpR,Rec=Rec,BinR=BinR)
              #np.savez('./'+Plt_Dir+'/Valid.npz',FMCV=AmpV,Val=Val,BinV=BinV)
      
              #torch.save(Model.state_dict(),'./'+Plt_Dir+'/Model.pth') 
              torch.save(Model.state_dict(), f'./{Plt_Dir}/Model.pth')
              #print(f"Finished training with seed {seed_value}, learning rate {itrain}")
              writer.close()
              
              # Final metrics save at last epoch
              mat_metrics_path_final = os.path.join(Plt_Dir, f"metrics_final_epoch_{epoch}.mat")
              
              # Package final MoE tensors for .mat export
              moe_data_final = {}
              if mse_losses:
                  moe_data_final.update({
                      "mse_losses": mse_losses,
                      "balance_losses": balance_losses,
                      "entropy_losses": entropy_losses,
                      "expert_usage": expert_usage,
                      "val_mse_losses": val_mse_losses,
                      "val_balance_losses": val_balance_losses,
                      "val_entropy_losses": val_entropy_losses,
                      "val_expert_usage": val_expert_usage
                  })
              
              # Append raw MSE values from losscalc
              if mse_vals:
                  moe_data_final.update({
                      "mse_vals": mse_vals,
                      "val_mse_vals": val_mse_vals
                  })
              
              sio.savemat(mat_metrics_path_final, {
                  "epoch": epoch,
                  "running_loss": running_loss,
                  "val_running_loss": val_running_loss,
                  "accuracy_loss": accuracy_loss,
                  "val_accuracy_loss": val_accuracy_loss,
                  "epoch_inc": epoch_inc,
                  **moe_data_final
              })

              
              
              for i in tqdm(range(SizeTrain//SizeTrain)):
      
                  AmpR[i,:,:] = AmpR[i,:,:] / np.max(np.absolute(AmpR[i,:,:]))
                  Rec[i,:,:] = Rec[i,:,:] / np.max(np.absolute(Rec[i,:,:]))
      
                  plt.figure(figsize=(3,5))
                  plt.imshow(AmpR[i,:,:])
                  plt.xlabel("Element axis")
                  plt.ylabel("Time Increment")
                  plt.set_cmap('seismic')
                  plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                  plt.xticks([0,512,1024])            
                  plt.clim(-1,1)
                  cbar=plt.colorbar()
                  cbar.set_label('Amplitude (linear)')
                  plt.savefig('./'+Plt_Dir+'/T_'+ f'{i}' +'_Amp.png',dpi=1200,bbox_inches='tight')
                  plt.close()
      
                  plt.figure(figsize=(3,5))
                  plt.imshow(Rec[i,:,:])
                  plt.xlabel("Element axis")
                  plt.ylabel("Time Increment")
                  plt.set_cmap('seismic')
                  plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                  plt.xticks([0,512,1024])            
                  plt.clim(-1,1)
                  cbar=plt.colorbar()
                  cbar.set_label('Amplitude (linear)')
                  plt.savefig('./'+Plt_Dir+'/T_'+ f'{i}' +'_Rec.png',dpi=1200,bbox_inches='tight')
                  plt.close()
      
                  Err=AmpR[i,:,:]-Rec[i,:,:]
                  Err=np.absolute(Err)
      
                  plt.figure(figsize=(3,5))
                  plt.imshow(Err)
                  plt.xlabel("Element axis")
                  plt.ylabel("Time Increment")
                  plt.set_cmap('inferno')
                  plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                  plt.xticks([0,512,1024])            
                  plt.clim(0,1)
                  cbar=plt.colorbar()
                  cbar.set_label('Error (absolute)')
                  plt.savefig('./'+Plt_Dir+'/T_'+ f'{i}' +'_Error.png',dpi=1200,bbox_inches='tight')
                  plt.close()
      
                #   plt.figure(figsize=(3,5))
                #   plt.imshow(BinR[i,:,:])
                #   plt.xlabel("Element axis")
                #   plt.ylabel("Time Increment")
                #   plt.set_cmap('seismic')
                #   plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                #   plt.xticks([0,512,1024])            
                #   plt.clim(-1,1)
                #   cbar=plt.colorbar()
                #   cbar.set_label('Amplitude (binary)')
                #   plt.savefig('./'+Plt_Dir+'/T_'+ f'{i}' +'_Bin.png',dpi=1200,bbox_inches='tight')
                #   plt.close()   
      
              for i in tqdm(range(SizeVal//SizeVal)):
      
                  AmpV[i,:,:] = AmpV[i,:,:] / np.max(np.absolute(AmpV[i,:,:]))
                  Val[i,:,:] = Val[i,:,:] / np.max(np.absolute(Val[i,:,:]))
      
                  plt.figure(figsize=(3,5))
                  plt.imshow(AmpV[i,:,:])
                  plt.xlabel("Element axis")
                  plt.ylabel("Time Increment")
                  plt.set_cmap('seismic')
                  plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                  plt.xticks([0,512,1024])            
                  plt.clim(-1,1)
                  cbar=plt.colorbar()
                  cbar.set_label('Amplitude (linear)')
                  plt.savefig('./'+Plt_Dir+'/V_'+ f'{i}' +'_Amp.png',dpi=1200,bbox_inches='tight')
                  plt.close()
      
                  plt.figure(figsize=(3,5))
                  plt.imshow(Val[i,:,:])
                  plt.xlabel("Element axis")
                  plt.ylabel("Time Increment")
                  plt.set_cmap('seismic')
                  plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                  plt.xticks([0,512,1024])            
                  plt.clim(-1,1)
                  cbar=plt.colorbar()
                  cbar.set_label('Amplitude (linear)')
                  plt.savefig('./'+Plt_Dir+'/V_'+ f'{i}' +'_Rec.png',dpi=1200,bbox_inches='tight')
                  plt.close()
      
                  Err=AmpV[i,:,:]-Val[i,:,:]
                  Err=np.absolute(Err)
      
                  plt.figure(figsize=(3,5))
                  plt.imshow(Err)
                  plt.xlabel("Element axis")
                  plt.ylabel("Time Increment")
                  plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                  plt.xticks([0,512,1024])            
                  plt.set_cmap('inferno')
                  plt.clim(0,1)
                  cbar=plt.colorbar()
                  cbar.set_label('Error (absolute)')
                  plt.savefig('./'+Plt_Dir+'/V_'+ f'{i}' +'_Error.png',dpi=1200,bbox_inches='tight')
                  plt.close()
      
                #   plt.figure(figsize=(3,5))
                #   plt.imshow(BinV[i,:,:])
                #   plt.xlabel("Element axis")
                #   plt.ylabel("Time Increment")
                #   plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                #   plt.xticks([0,512,1024])
                #   plt.set_cmap('seismic')
                #   plt.clim(-1,1)
                #   cbar=plt.colorbar()
                #   cbar.set_label('Amplitude (binary)')
                #   plt.savefig('./'+Plt_Dir+'/V_'+ f'{i}' +'_Bin.png',dpi=1200,bbox_inches='tight')
                #   plt.close()  
                  #tbcallback
              
      
              matlab_dir = f'./{Plt_Dir}/Matlab'+model_ici
              os.makedirs(matlab_dir, exist_ok=True)
              
              def save_in_parts(data, base_filename, num_parts):
                  num_fmc = num_parts
                  for fmc_idx in tqdm(range(num_fmc)):
                      fmc_data = data[fmc_idx, :, :]
                      fmc_filename = os.path.join(matlab_dir, f"{base_filename}_FMC{fmc_idx+1}.mat")
                      sio.savemat(fmc_filename, {f"{base_filename}_FMC{fmc_idx+1}": fmc_data})
                      #print(f'Saved {filename} with shape {data_part.shape}')
              

              
if __name__ == "__main__":
    main()