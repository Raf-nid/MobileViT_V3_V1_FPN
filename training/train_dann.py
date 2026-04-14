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
from MobileUnet_V2_CRM import MobileNetV2_dynamicFPN
from MobileNetV3 import MobileNetV3_dynamicFPN
from SETR import SETR_Naive_S,SETR_PUP_S
from MnasNet import MnasNet_dynamicFPN
from mobilevit_v3_v1 import MobileViTv3_v1_dynamicFPN
from mobilevit_v3_v1_DANN import MobileViTv3_v1_dynamicFPN_DANN
from mobilevit_v3_v2 import MobileViTv3_v2_dynamicFPN


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
"""Domain-adversarial training (DANN) entry point with optional fixed seed."""


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
    def __init__(self, directory, device='cuda:0'):
        self.files = sorted(glob.glob(os.path.join(directory, '*.mat')))
        self.filenames = [os.path.basename(f) for f in self.files] 
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        base = os.path.splitext(self.filenames[idx])[0]
        truncated = base[-5:] + '.mat'
        with h5py.File(filepath, 'r') as f:
            fmc = torch.tensor(f['FMC'][()].astype('float32'))
            bin = torch.tensor(f['Bin'][()].astype('float32'))
    
        # Align FMC / binary tensor layout (H, W)
        fmc = fmc.permute(1, 0)
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                  model_ici ='Mobilevitv3V1_FPN_Bin_BruitFixeDuet_NWonly_rd32prem_rd32last_50_50New_patch128'
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Train"
                  data_dir_valid = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Valid"
                  reference = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Train/pyFMC_5MHz_NW_14_2.mat"
#                  data_dir3 = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/FMC_Bin_antoine_2927_3.mat"
              elif truc==1:
                  model_ici ='Mobilevitv3V1_FPN_Bin_TFlearning_NoShiftBin'
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_Bin_New_20x225_20x5_20x75_4real_NoShift.mat')              
              elif truc ==2:
                  model_ici ='Mobilevitv2_FPN_Bin_New_20x225_20x5_20x75_4real_NoShift'
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_Bin_New_20x225_20x5_20x75_4real_NoShift.mat')
              elif truc ==3:
                  model_ici ='Mobilevitv1_FPN_Bin_New_20x225_20x5_20x75_4real_NoShift'
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_Bin_New_20x225_20x5_20x75_4real_NoShift.mat')
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

             
                
              if truc == 0 or truc ==1:
                  Model = MobileViTv3_v1_dynamicFPN_DANN((4096, 1024), 'xx_small', 1000, (128,128)).to("cuda:0")
              elif truc == 1:
                  Model = MobileViTv3_v1_dynamicFPN((4096, 1024), 'xx_small', 1000, (128,128)).to("cuda:0")
              elif truc == 2:
                  Model = MobileViT_dynamicFPN(image_size=(4096, 1024), mode='xx_small', num_classes=1000).to("cuda:0")
              elif truc == 3:
                  Model = MobileViTv2_dynamicFPN(image_size=(4096, 1024), width_multiplier=0.5, num_classes=1000).to("cuda:0")
              elif truc == 4:
                  Model = MobileViTv3_v2_dynamicFPN(image_size=(4096, 1024), width_multiplier=0.5, num_classes=1000).to("cuda:0")
              else:
                  Model = MobileViTv3_v1_dynamicFPN((4096, 1024), 'xx_small', 1000, (32,32)).to("cuda:0")
                  

              
              train_dataset = MatDataset(directory=data_dir, device="cuda:0")
              val_dataset = MatDataset(directory=data_dir_valid, device="cuda:0")

              train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
              val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

              target_data_dir = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Target"
              target_dataset = MatDataset(directory=target_data_dir, device="cuda:0")

              target_loader = DataLoader(dataset=target_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

              losscalc =  nn.MSELoss() # Compute loss
              domain_criterion = nn.CrossEntropyLoss()

              #losscalc = NCC_MSE_Loss()
               # Algo to actualize MM weight
              #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=100000, factor=0.5, verbose=True) # Allow to reduce the learning rate during the training (if patience is superior to config.num_epoch = disabled)
              optimizer = torch.optim.Adam(Model.parameters(),lr=itrain)
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
                  
                  cnt = 0
                 
                  loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True)
                  target_iter = iter(target_loader)
                  # Training Loop
                  for (amp, bin, fname) in loop:
                      
                      try:
                         
                         x_target, _, _ = next(target_iter)  # target domain: amplitude unused
                      except StopIteration:
                        target_iter = iter(target_loader)
                        x_target, _, _ = next(target_iter)

                      amp, bin = amp.to("cuda:0"), bin.to("cuda:0")
                      x_target = x_target.to("cuda:0")

                      optimizer.zero_grad()
                      #with torch.cuda.amp.autocast(enabled=True):
                      recon, domain_pred_src = Model(bin)
                      _, domain_pred_tgt = Model(x_target)
                          # placeholder for pseudo-label / auxiliary loss if needed
                          #recon_resized = func.interpolate(recon, size=(amp.size(2),amp.size(3)), mode='bilinear', align_corners=False)
                          #print(amp.shape,recon.shape)     
                      # Pertes
                      loss_task = losscalc(amp, recon)
                      loss_domain_src = domain_criterion(domain_pred_src, torch.zeros(recon.size(0)).long().to('cuda:0'))
                      loss_domain_tgt = domain_criterion(domain_pred_tgt, torch.ones(x_target.size(0)).long().to('cuda:0'))

                      loss = loss_task + (loss_domain_src + loss_domain_tgt)

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
                  
                      loop.set_description(f"Epoch {epoch} |File: {fname}|Loss:{loss.item():.4e}|NCC:{acc_val:.4f}|MSE:{loss_task.item():.4f}|domain_src:{loss_domain_src.item():.4f}|domain_tgt:{loss_domain_tgt.item():.4f}")
                      
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
      
                  cnt = 0
      
                  # Validation Loop
                  with torch.no_grad():
                      
                      Model.eval()
                      val_losses = []
                      val_accuracy = []

      
                      for (ampV, binV,_) in val_loader:
                          ampV, binV = ampV.to("cuda:0"), binV.to("cuda:0")
                          #with torch.cuda.amp.autocast(enabled=True):
                          reconV,_ = Model(binV)
                              #print(reconV)
                              #recon_resizedV = func.interpolate(reconV, size=(ampV.size(2),ampV.size(3)), mode='bilinear', align_corners=False)
                          loss_val = losscalc(ampV, reconV)
      
                          #my_lr = scheduler.optimizer.param_groups[0]['lr']
                          my_lr = optimizer.param_groups[0]['lr']
      
                          #if epoch==1:
                              #initvalloss = loss_val.item()
      
                          #val_losses = np.append(val_losses,loss_val.item()/initvalloss)
                          val_losses = np.append(val_losses,loss_val.item())
                          val_accuracy.append(ncc(ampV, reconV).item())
      
      
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
                          cnt += 1
                      

                          
                  SizeTrain = Rec.shape[0]
                  SizeVal = Val.shape[0]
       
                  
                  
                  # Recording differents losses and managing the early stopping algo
                  if epoch % 1 == 0:
      
                      running_loss = np.append(running_loss,np.mean(losses))
                      val_running_loss = np.append(val_running_loss,np.mean(val_losses))  
                      accuracy_loss.append(np.mean(accuracy))
                      val_accuracy_loss.append(np.mean(val_accuracy))

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
                      torch.save(es.best_model, os.path.join(epoch_dir, "Model.pth"))
                      
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
                  
#              dummy_input = torch.randn(1, 1, 1024, 4096).to("cuda:0")  # example dummy input for add_graph
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