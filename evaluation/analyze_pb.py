"""Legacy TensorBoard-oriented training prototype (old import paths). For validation, use ``evaluate_loop2.py``."""
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
import os
from os.path import dirname, join as pjoin
import scipy.io as sio
import mat73
from utils import EarlyStopping,ncc, NCC_MSE_Loss
import datetime
import h5py
from torch.utils.tensorboard import SummaryWriter




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





def main():
    #config.patience = 100
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
          for truc in range(5):
              if truc == 0:
                  model_ici ='ANALYSE_FPN_new'
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_5MHznew.mat')
              elif truc==1:
                  model_ici ='ANALYSE_FPN_new_clipping_1'
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_5MHznew.mat')
              elif truc ==2:
                  model_ici ='ANALYSE_FPN_new_clipping_0_7'
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_5MHznew.mat')
              elif truc ==3:
                  model_ici ='ANALYSE_FPN_new_clipping_0_5'
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_5MHznew.mat')
              elif truc ==4:
                  model_ici ='ANALYSE_FPN_new_clipping_2'
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC
                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_5MHznew.mat')
              elif truc ==5:
                  model_ici ='FPN_2real225-5_'
                  summary = f'./runs/'+model_ici
                  writer = SummaryWriter(summary)
                  #Plt_Dir='Test_End_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str +'_'+ f'{config.num_epochs}'+'_epochs'
                  Model_Name = './Model/' + f'MobileNetV2_unet_'+model_ici + config.Case + '_lr_' + f'{np.round(1e6*itrain).astype(int)}' + 'e-6.pth'
                  Plt_Dir = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  Plt_Dir_epoch = f'Test_End_seed_lr_'+model_ici + f'{np.round(1e6*itrain).astype(int)}' + 'e-6_' + date_str + '_epochs_intermediaire_' + f'{config.num_epochs}' + '_epochs_seed_'+f'{str(seed_value)}'
                  # recuperation of FMC

                  data_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC2_25-5MHz2Real.mat')

              # Loading data
              if config.Vmat == 7.3:
                  FMC = mat73.loadmat(data_dir)['FMC']
                  Bin = mat73.loadmat(data_dir)['Bin']
              else :
                  FMC = sio.loadmat(data_dir)['FMC']
                  Bin = sio.loadmat(data_dir)['Bin']


              # Reorganising the dimencion of the FMC and Binary matrices
              FMC = np.transpose(FMC,(2,0,1))
              Bin = np.transpose(Bin,(2,0,1))

              FMC = torch.from_numpy(FMC).float().to("cuda:0")
              FMC = torch.unsqueeze(FMC,1)    # optional: add channel dim

              Bin = torch.from_numpy(Bin).float().to("cuda:0")
              Bin = torch.unsqueeze(Bin,1)    #idem
              print(FMC.shape)



              # load validation FMC

              valid_dir = pjoin(os.getcwd(), 'Data', config.Case,'FMC_Duet_validation.mat')

              if config.Vmat == 7.3:
                  FMCreal = mat73.loadmat(valid_dir)['FMC']
                  Binreal = mat73.loadmat(valid_dir)['Bin']
              else :
                  FMCreal = sio.loadmat(valid_dir)['FMC']
                  Binreal = sio.loadmat(valid_dir)['Bin']

              FMCreal = torch.from_numpy(FMCreal).float().to("cuda:0")
              FMCreal = torch.unsqueeze(FMCreal,0)   # optional: add channel dim
              FMCreal = torch.unsqueeze(FMCreal,0)

              Binreal = torch.from_numpy(Binreal).float().to("cuda:0")
              Binreal = torch.unsqueeze(Binreal,0)  #idem
              Binreal = torch.unsqueeze(Binreal,0)
              print(FMCreal.shape)


              Model = MobileNetV2_dynamicFPN().to("cuda:0")

              pretrained_path = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/MobileUnet_V2_modif/Test_End_seed_lr_FPN_new200e-6_20250227_095315_2500_epochs_seed_69738009/Model.pth"
              if os.path.exists(pretrained_path):
                  Model.load_state_dict(torch.load(pretrained_path, map_location=device))
                  print(f"Loaded pretrained weights from {pretrained_path}")
              else:
                  print(f"No pretrained checkpoint at {pretrained_path}; training from scratch.")


              # Creation of the Training and validation Dataset
              Dataset=TensorDataset(FMC,Bin)

              Datsize = FMC.shape[0]       # Number of samples in the dataset
              train_size = np.round(Datsize*(1-config.val_per/100)).astype('int') # Number of samples in the training dataset
              val_size = Datsize-train_size.astype('int') # Number of samples in the validation dataset



              DatasetDuet = TensorDataset(FMCreal,Binreal)
              duet_size = FMCreal.shape[0]
              # indices for two groups
              # train_indices = [0, 2]
              # val_indices = [1, 3]

              # manual subsets
              #DatasetTrain = torch.utils.data.Subset(Dataset, train_indices)
              #DatasetVal = torch.utils.data.Subset(Dataset, val_indices)


              DatasetTrain, DatasetVal=torch.utils.data.random_split(Dataset,[train_size, val_size])  # random train/val split
              train_loader=DataLoader(dataset=DatasetTrain,batch_size=config.batch_size,shuffle=True,drop_last=True)
              val_loader=DataLoader(dataset=DatasetVal,batch_size=config.batch_size,shuffle=True,drop_last=True)
              # Creation of the Loss function optimizer and scheduler

              val_loader_duet = DataLoader(dataset=DatasetDuet, batch_size=config.batch_size, shuffle=False)
              losscalc =  nn.MSELoss() # Compute loss
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
              val_duet_loss = []
              # Early stopping definition and variables
              es = EarlyStopping()
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


                  # Training Loop
                  for (amp, bin) in train_loader:
                      amp, bin = amp.to("cuda:0"), bin.to("cuda:0")
                      optimizer.zero_grad()
                      #with torch.cuda.amp.autocast(enabled=True):
                      recon = Model(bin)
                          # placeholder for pseudo-label / auxiliary loss if needed
                          #recon_resized = func.interpolate(recon, size=(amp.size(2),amp.size(3)), mode='bilinear', align_corners=False)
                          #print(amp.shape,recon.shape)
                      loss = losscalc(amp, recon)

                      loss.backward()
                      if truc==1:
                          torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=1.0)
                      elif truc ==2:
                          torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=0.7)
                      elif truc ==3:
                          torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=0.5)
                      elif truc ==4:
                          torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=2.0)
                      optimizer.step()
                      #scaler.scale(loss).backward()
                      #scaler.step(optimizer)
                      #scaler.update()


                      if epoch==1:
                          initloss=loss.item()

                      losses = np.append(losses,loss.item()/initloss)
                      epoch_cnt = np.append(epoch_cnt,epoch)
                      accuracy.append(ncc(amp, recon).item())

                      if cnt==0:
                          Rec = torch.squeeze(recon,1).detach().to('cpu').numpy()
                          AmpR = torch.squeeze(amp,1).detach().to('cpu').numpy()
                          BinR = torch.squeeze(bin,1).detach().to('cpu').numpy()

                      if cnt>0:
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
                      val_duet = []

                      for (ampV, binV) in val_loader:
                          ampV, binV = ampV.to("cuda:0"), binV.to("cuda:0")
                          #with torch.cuda.amp.autocast(enabled=True):
                          reconV = Model(binV)
                              #print(reconV)
                              #recon_resizedV = func.interpolate(reconV, size=(ampV.size(2),ampV.size(3)), mode='bilinear', align_corners=False)
                          loss_val = losscalc(ampV, reconV)

                          #my_lr = scheduler.optimizer.param_groups[0]['lr']
                          my_lr = optimizer.param_groups[0]['lr']

                          if epoch==1:
                              initvalloss = loss_val.item()

                          val_losses = np.append(val_losses,loss_val.item()/initvalloss)
                          val_accuracy.append(ncc(ampV, reconV).item())


                          if cnt==0:
                              Val = torch.squeeze(reconV,1).detach().to('cpu').numpy()
                              AmpV = torch.squeeze(ampV,1).detach().to('cpu').numpy()
                              BinV = torch.squeeze(binV,1).detach().to('cpu').numpy()

                          if cnt>0:
                              Val = np.append(Val,torch.squeeze(reconV,1).detach().to('cpu').numpy(),axis=0)
                              AmpV = np.append(AmpV,torch.squeeze(ampV,1).detach().to('cpu').numpy(),axis=0)
                              BinV = np.append(BinV,torch.squeeze(binV,1).detach().to('cpu').numpy(),axis=0)
                          avg_val_loss = np.mean(val_losses)
                          avg_val_accuracy = np.mean(val_accuracy)

                          # **TensorBoard: Log validation loss & accuracy**
                          writer.add_scalar("Loss/Val", avg_val_loss, epoch)
                          writer.add_scalar("Accuracy/Val", avg_val_accuracy, epoch)
                          cnt += 1

                      # Evaluation de la FMC du duet jamais vu par l'ia
                      cnt2 = 0
                      for (ampD, binD) in val_loader_duet:
                          ampD, binD = ampD.to("cuda:0"), binD.to("cuda:0")
                          reconD = Model(binD)
                          val_duet.append(ncc(ampD, reconD).item())
                          if cnt2==0:
                              ValD = torch.squeeze(reconD,1).detach().to('cpu').numpy()
                              AmpD = torch.squeeze(ampD,1).detach().to('cpu').numpy()
                              BinD = torch.squeeze(binD,1).detach().to('cpu').numpy()

                          if cnt2>0:
                              ValD = np.append(ValD,torch.squeeze(reconD,1).detach().to('cpu').numpy(),axis=0)
                              AmpD = np.append(AmpD,torch.squeeze(ampD,1).detach().to('cpu').numpy(),axis=0)
                              BinD = np.append(BinD,torch.squeeze(binD,1).detach().to('cpu').numpy(),axis=0)

                          avg_val_duet = np.mean(val_duet)
                          cnt2 += 1


                  SizeTrain = Rec.shape[0]
                  SizeVal = Val.shape[0]
                  SizeDuet = ValD.shape[0]


                  # Recording differents losses and managing the early stopping algo
                  if epoch % 1 == 0:

                      running_loss = np.append(running_loss,np.mean(losses))
                      val_running_loss = np.append(val_running_loss,np.mean(val_losses))
                      accuracy_loss.append(np.mean(accuracy))
                      val_accuracy_loss.append(np.mean(val_accuracy))
                      val_duet_loss.append(np.mean(val_duet))
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

                  if epoch == 25 or epoch %250 == 0:
                      epoch_dir = os.path.join(Plt_Dir_epoch, f"epoch_{epoch}")
                      os.makedirs(epoch_dir, exist_ok=True)


                      min_train=np.ones_like(epoch_inc)*np.min(running_loss)
                      min_val=np.ones_like(epoch_inc)*np.min(val_running_loss)
                      min_train2 = np.ones_like(epoch_inc) * np.max(accuracy_loss)
                      min_val2 = np.ones_like(epoch_inc) * np.max(val_accuracy_loss)
                      min_val3 = np.ones_like(epoch_inc) * np.max(val_duet_loss)

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
                      plt.savefig(os.path.join(epoch_dir, f'train_loss.png'),dpi=1200)
                      plt.close()

                      plt.figure()
                      plt.plot(np.arange(len(accuracy_loss)), accuracy_loss, 'b-', label='Training')
                      plt.plot(np.arange(len(accuracy_loss)), min_train2, 'b--')
                      plt.plot(np.arange(len(val_accuracy_loss)), val_accuracy_loss, 'g-', label='Validation')
                      plt.plot(np.arange(len(val_accuracy_loss)), min_val2, 'g--')
                      plt.plot(np.arange(len(val_duet_loss)), val_duet_loss, 'r-', label='Duet')
                      plt.plot(np.arange(len(val_duet_loss)), min_val3, 'r--')
                      plt.xlabel("Training step")
                      plt.ylabel("Accuracy")
                      plt.yscale("log")
                      plt.grid(True)
                      plt.title(f'Minimal Train accuracy : {np.max(accuracy_loss):.4E} / Val Accuracy : {np.max(val_accuracy_loss):.4E} / Duet_Accuracy : {np.max(val_duet_loss):.4E}')
                      plt.legend()
                      plt.savefig(os.path.join(epoch_dir, f'train_accuracy.png'), dpi=1200)
                      plt.close()

                      if epoch%500==0 or epoch ==25:
                          for i in tqdm(range(SizeTrain)):

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

                              plt.figure(figsize=(3,5))
                              plt.imshow(BinR[i,:,:])
                              plt.xlabel("Element axis")
                              plt.ylabel("Time Increment")
                              plt.set_cmap('seismic')
                              plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                              plt.xticks([0,512,1024])
                              plt.clim(-1,1)
                              cbar=plt.colorbar()
                              cbar.set_label('Amplitude (binary)')
                              plt.savefig(os.path.join(epoch_dir, f'T_{i}_Bin.png'),dpi=1200,bbox_inches='tight')
                              plt.close()

                          for i in tqdm(range(SizeVal)):

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

                              plt.figure(figsize=(3,5))
                              plt.imshow(BinV[i,:,:])
                              plt.xlabel("Element axis")
                              plt.ylabel("Time Increment")
                              plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                              plt.xticks([0,512,1024])
                              plt.set_cmap('seismic')
                              plt.clim(-1,1)
                              cbar=plt.colorbar()
                              cbar.set_label('Amplitude (binary)')
                              plt.savefig(os.path.join(epoch_dir, f'V_{i}_Bin.png'),dpi=1200,bbox_inches='tight')
                              plt.close()


                          for i in tqdm(range(SizeDuet)):

                              AmpD[i,:,:] = AmpD[i,:,:] / np.max(np.absolute(AmpD[i,:,:]))
                              ValD[i,:,:] = ValD[i,:,:] / np.max(np.absolute(ValD[i,:,:]))

                              plt.figure(figsize=(3,5))
                              plt.imshow(AmpD[i,:,:])
                              plt.xlabel("Element axis")
                              plt.ylabel("Time Increment")
                              plt.set_cmap('seismic')
                              plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                              plt.xticks([0,512,1024])
                              plt.clim(-1,1)
                              cbar=plt.colorbar()
                              cbar.set_label('Amplitude (linear)')
                              plt.savefig(os.path.join(epoch_dir, f'D_{i}_Amp.png'),dpi=1200,bbox_inches='tight')
                              plt.close()

                              plt.figure(figsize=(3,5))
                              plt.imshow(ValD[i,:,:])
                              plt.xlabel("Element axis")
                              plt.ylabel("Time Increment")
                              plt.set_cmap('seismic')
                              plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                              plt.xticks([0,512,1024])
                              plt.clim(-1,1)
                              cbar=plt.colorbar()
                              cbar.set_label('Amplitude (linear)')
                              plt.savefig(os.path.join(epoch_dir, f'Duet_{i}_Rec.png'),dpi=1200,bbox_inches='tight')
                              plt.close()

                              Err=AmpD[i,:,:]-ValD[i,:,:]
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
                              plt.savefig(os.path.join(epoch_dir, f'Duet_{i}_Error.png'),dpi=1200,bbox_inches='tight')
                              plt.close()

                              plt.figure(figsize=(3,5))
                              plt.imshow(BinD[i,:,:])
                              plt.xlabel("Element axis")
                              plt.ylabel("Time Increment")
                              plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                              plt.xticks([0,512,1024])
                              plt.set_cmap('seismic')
                              plt.clim(0,1)
                              cbar=plt.colorbar()
                              cbar.set_label('Amplitude (binary)')
                              plt.savefig(os.path.join(epoch_dir, f'Duet_{i}_Bin.png'),dpi=1200,bbox_inches='tight')
                              plt.close()


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

                  plt.figure()
                  plt.plot(np.arange(epoch), accuracy_loss,'b-', label='Training')
                  plt.plot(np.arange(epoch), val_accuracy_loss,'g-', label='Validation')
                  plt.plot(np.arange(epoch), val_duet_loss,'r-', label='Duet')
                  plt.xlabel("Training step")
                  plt.ylabel("Accuracy")
                  plt.legend()
                  #plt.yscale("log")
                  plt.savefig('./RunningAccuracy/'+Plt_Dir+'.png',dpi=1200)
                  plt.close()

                  #writer.add_histogram(f'{name}/weights', param, epoch)
                  #writer.add_histogram(f'{name}/gradients', param.grad, epoch)

              dummy_input = torch.randn(1, 1, 1024, 4096).to("cuda:0")  # example dummy input for add_graph
              writer.add_graph(Model, dummy_input)

              SizeTrain = Rec.shape[0]
              SizeVal = Val.shape[0]
              SizeDuet = ValD.shape[0]

              os.mkdir(Plt_Dir)

              min_train=np.ones_like(epoch_inc)*np.min(running_loss)
              min_val=np.ones_like(epoch_inc)*np.min(val_running_loss)
              min_train2 = np.ones_like(epoch_inc) * np.max(accuracy_loss)
              min_val2 = np.ones_like(epoch_inc) * np.max(val_accuracy_loss)
              min_val3 = np.ones_like(epoch_inc) * np.max(val_duet_loss)

              plt.figure()
              plt.plot(np.arange(len(accuracy_loss)), accuracy_loss, 'b-', label='Training')
              plt.plot(np.arange(len(accuracy_loss)), min_train2, 'b--')
              plt.plot(np.arange(len(val_accuracy_loss)), val_accuracy_loss, 'g-', label='Validation')
              plt.plot(np.arange(len(val_accuracy_loss)), min_val2, 'g--')
              plt.plot(np.arange(len(val_duet_loss)), val_duet_loss, 'r-', label='Duet')
              plt.plot(np.arange(len(val_duet_loss)), min_val3, 'r--')
              plt.xlabel("Training step")
              plt.ylabel("Accuracy")
              plt.yscale("log")
              plt.grid(True)
              plt.title(f'Minimal Train accuracy : {np.max(accuracy_loss):.4E} / Val Accuracy : {np.max(val_accuracy_loss):.4E} / Duet_Accuracy : {np.max(val_duet_loss):.4E}')
              plt.legend()
              plt.savefig('./'+Plt_Dir+'/train_accuracy.png', dpi=1200)
              plt.close()


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
              plt.savefig('./'+Plt_Dir+'/loss_title.png',dpi=1200)
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



              np.savez('./'+Plt_Dir+'/loss.npz',epoch_inc=np.arange(epoch),running_loss=running_loss,val_running_loss=val_running_loss)
              np.savez('./'+Plt_Dir+'/Train.npz',FMCT=AmpR,Rec=Rec,BinR=BinR)
              np.savez('./'+Plt_Dir+'/Valid.npz',FMCV=AmpV,Val=Val,BinV=BinV)

              #torch.save(Model.state_dict(),'./'+Plt_Dir+'/Model.pth')
              torch.save(Model.state_dict(), f'./{Plt_Dir}/Model.pth')
              #print(f"Finished training with seed {seed_value}, learning rate {itrain}")
              writer.close()

              for i in tqdm(range(SizeTrain)):

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

                  plt.figure(figsize=(3,5))
                  plt.imshow(BinR[i,:,:])
                  plt.xlabel("Element axis")
                  plt.ylabel("Time Increment")
                  plt.set_cmap('seismic')
                  plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                  plt.xticks([0,512,1024])
                  plt.clim(-1,1)
                  cbar=plt.colorbar()
                  cbar.set_label('Amplitude (binary)')
                  plt.savefig('./'+Plt_Dir+'/T_'+ f'{i}' +'_Bin.png',dpi=1200,bbox_inches='tight')
                  plt.close()

              for i in tqdm(range(SizeVal)):

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

                  plt.figure(figsize=(3,5))
                  plt.imshow(BinV[i,:,:])
                  plt.xlabel("Element axis")
                  plt.ylabel("Time Increment")
                  plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                  plt.xticks([0,512,1024])
                  plt.set_cmap('seismic')
                  plt.clim(-1,1)
                  cbar=plt.colorbar()
                  cbar.set_label('Amplitude (binary)')
                  plt.savefig('./'+Plt_Dir+'/V_'+ f'{i}' +'_Bin.png',dpi=1200,bbox_inches='tight')
                  plt.close()
                  #tbcallback
              for i in tqdm(range(SizeDuet)):
                  AmpD[i,:,:] = AmpD[i,:,:] / np.max(np.absolute(AmpD[i,:,:]))
                  ValD[i,:,:] = ValD[i,:,:] / np.max(np.absolute(ValD[i,:,:]))

                  plt.figure(figsize=(3,5))
                  plt.imshow(AmpD[i,:,:])
                  plt.xlabel("Element axis")
                  plt.ylabel("Time Increment")
                  plt.set_cmap('seismic')
                  plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                  plt.xticks([0,512,1024])
                  plt.clim(-1,1)
                  cbar=plt.colorbar()
                  cbar.set_label('Amplitude (linear)')
                  plt.savefig('./'+Plt_Dir+'/Duet_'+ f'{i}' +'_Amp.png',dpi=1200,bbox_inches='tight')
                  plt.close()

                  plt.figure(figsize=(3,5))
                  plt.imshow(ValD[i,:,:])
                  plt.xlabel("Element axis")
                  plt.ylabel("Time Increment")
                  plt.set_cmap('seismic')
                  plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                  plt.xticks([0,512,1024])
                  plt.clim(-1,1)
                  cbar=plt.colorbar()
                  cbar.set_label('Amplitude (linear)')
                  plt.savefig('./'+Plt_Dir+'/Duet_'+ f'{i}' +'_Rec.png',dpi=1200,bbox_inches='tight')
                  plt.close()

                  Err=AmpD[i,:,:]-ValD[i,:,:]
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
                  plt.savefig('./'+Plt_Dir+'/Duet_'+ f'{i}' +'_Error.png',dpi=1200,bbox_inches='tight')
                  plt.close()

                  plt.figure(figsize=(3,5))
                  plt.imshow(BinD[i,:,:])
                  plt.xlabel("Element axis")
                  plt.ylabel("Time Increment")
                  plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
                  plt.xticks([0,512,1024])
                  plt.set_cmap('seismic')
                  plt.clim(0,1)
                  cbar=plt.colorbar()
                  cbar.set_label('Amplitude (binary)')
                  plt.savefig('./'+Plt_Dir+'/Duet_'+ f'{i}' +'_Bin.png',dpi=1200,bbox_inches='tight')
                  plt.close()

              matlab_dir = f'./{Plt_Dir}/Matlab'
              os.makedirs(matlab_dir, exist_ok=True)

              def save_in_parts(data, base_filename, num_parts):
                  num_fmc = num_parts  # Nombre de FMC
                  for fmc_idx in tqdm(range(num_fmc)):  # Boucle sur chaque FMC
                      fmc_data = data[fmc_idx,:,:]  # Extraction de la FMC
                      fmc_filename = os.path.join(matlab_dir, f"{base_filename}_FMC{fmc_idx+1}.mat")
                      sio.savemat(fmc_filename, {f"{base_filename}_FMC{fmc_idx+1}": fmc_data})
                      #print(f'Saved {filename} with shape {data_part.shape}')

              # chunked save to disk
              save_in_parts(AmpR, "T_FMC", train_size)
              save_in_parts(Rec, "T_Rec", train_size)
              save_in_parts(BinR, "T_Bin", train_size)

              save_in_parts(AmpV, "V_FMC", val_size)
              save_in_parts(Val, "V_Rec", val_size)
              save_in_parts(BinV, "V_Bin", val_size)

              save_in_parts(AmpD, "Duet_FMC",duet_size)
              save_in_parts(ValD, "Duet_Rec",duet_size)
              save_in_parts(BinD, "Duet_Bin",duet_size)

if __name__ == "__main__":
    main()