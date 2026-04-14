import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as func
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import os
import sys
from pathlib import Path
import scipy.io as sio
import datetime
import h5py
from torch.utils.tensorboard import SummaryWriter
import glob
import copy
import argparse

# Allow running this file as a script: `python training/train.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Config and Utils
import configs.config_mobileunet as config
from utils.utils import EarlyStopping, ncc, NCC_MSE_Loss, save_fmc_images, save_metrics_plots

# Models via central registry
from models import get_model, AVAILABLE_MODELS

"""Standard training script with optional fixed random seed."""


class StagnationEarlyStopping:
    """
    Stagnation-based early stopping based on Lutz Prechelt (1998).
    Stops if validation loss doesn't improve by at least min_delta for 'patience' epochs.
    
    Args:
        patience (int): Number of epochs to wait for significant improvement.
        min_delta (float): Minimum change to qualify as an improvement.
    """
    def __init__(self, patience=15, min_delta=5e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model_wts = None
        self.absolute_best_loss = np.Inf
        self.absolute_best_wts = None
        self.early_stop = False
        self.status = ""

    def __call__(self, model, val_loss):
        # Initialize if first run
        if self.best_loss is None:
            self.best_loss = val_loss
            self.absolute_best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.absolute_best_wts = copy.deepcopy(model.state_dict())
            self.status = "Init"
        else:
            # Current improvement vs best validation loss
            diff = self.best_loss - val_loss
            
            # Check for stagnation (Stopping Criteria)
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                self.status = f"Imp: {diff:.1E} > {self.min_delta:.1e}"
            else:
                self.counter += 1
                self.status = f"Stag {self.counter}/{self.patience} (Δ:{diff:.1E})"
                if self.counter >= self.patience:
                    self.early_stop = True

            # Track absolute best model for saving (regardless of epsilon)
            if val_loss < self.absolute_best_loss:
                self.absolute_best_loss = val_loss
                self.absolute_best_wts = copy.deepcopy(model.state_dict())

        return self.early_stop


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
        truncated = base[-8:] #+ '.mat'
        with h5py.File(filepath, 'r') as f:
            fmc = torch.tensor(f['FMC'][()].astype('float32'))
            bin = torch.tensor(f['Bin'][()].astype('float32'))
    
        # Align FMC / binary tensor layout (H, W)
        fmc = fmc.permute(1, 0)
        bin = bin.permute(1, 0)
    
        fmc = fmc.unsqueeze(0).to(self.device)
        bin = bin.unsqueeze(0).to(self.device)
    
        return fmc, bin, truncated





def main(args):
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

    learning_rate = np.multiply(config.learning_rate, 1e-6)
    itrain = learning_rate[0] if isinstance(learning_rate, (list, np.ndarray)) else learning_rate

    # Date and time for the saving of the results
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    seed_value = 69738009
    set_seed(seed_value)
    print(f"Seed: {seed_value}")

    # Directories
    experiment_run_id = f'MbViTPixel2_p32x32_XXSMALL4_Multifrequency_FF8_Amplitude_batchsize{config.batch_size}_NW_plus_Wedge_BruitFixe_BruitDuet'
    summary = f'./runs/{experiment_run_id}'
    writer = SummaryWriter(summary)
    Plt_Dir = f'Test_End_seed_lr_{experiment_run_id}{np.round(1e6*itrain).astype(int)}e-6_{date_str}_{config.num_epochs}_epochs_seed_{seed_value}'
    Plt_Dir_epoch = f'Test_End_seed_lr_{experiment_run_id}{np.round(1e6*itrain).astype(int)}e-6_{date_str}_epochs_intermediaire_{config.num_epochs}_epochs_seed_{seed_value}'
    
    # FMC datasets: place HDF5 MATLAB v7.3 `.mat` files (FMC + Bin) under these folders
    data_dir = str(PROJECT_ROOT / "data" / "train_dataset")
    data_dir_valid = str(PROJECT_ROOT / "data" / "valid_dataset")
    print(f"Train data dir: {data_dir}")
    print(f"Valid data dir: {data_dir_valid}")

    print(f"Loading model: {args.model_name}")
    Model = get_model(
        args.model_name,
        image_size=(4096, 64),
        mode='xx_small4',
        num_classes=1000,
        patch_size=(32, 32),
    ).to(device)

    # Creation of the Training and validation Dataset
    train_dataset = MatDataset(directory=data_dir, device=str(device))
    val_dataset = MatDataset(directory=data_dir_valid, device=str(device))
    if len(train_dataset) == 0:
        raise FileNotFoundError(
            f"No .mat files found for training under:\n  {data_dir}\n"
            "Add HDF5 `.mat` samples under data/train_dataset and data/valid_dataset (project root), "
            "or change data_dir / data_dir_valid in training/train.py."
        )

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    
    losscalc =  nn.MSELoss() # Compute loss
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
    
    # Early stopping definition and variables
    # Early stopping definition and variables
    # Stagnation-based early stopping
    stagnation_es = StagnationEarlyStopping()

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
        # Training Loop
        for (amp, bin, fname) in loop:
            amp, bin = amp.to(device), bin.to(device)
            optimizer.zero_grad()
            #with torch.cuda.amp.autocast(enabled=True):
            recon = Model(bin)
                # placeholder for pseudo-label / auxiliary loss if needed
                #recon_resized = func.interpolate(recon, size=(amp.size(2),amp.size(3)), mode='bilinear', align_corners=False)
                #print(amp.shape,recon.shape)     
            loss = losscalc(amp, recon)
      
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
        
            #loop.set_description(f"Epoch {epoch} | File: {fname} | Loss: {loss.item():.4e} | NCC: {acc_val:.4f}")
            loop.set_description(f"Epoch {epoch} | Loss: {loss.item():.4e} | NCC: {acc_val:.4f}")
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
                ampV, binV = ampV.to(device), binV.to(device)
                #with torch.cuda.amp.autocast(enabled=True):
                reconV = Model(binV)
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
       
        
        
        # Record losses and run early-stopping logic
        if epoch % 1 == 0:
      
            running_loss = np.append(running_loss,np.mean(losses))
            val_running_loss = np.append(val_running_loss,np.mean(val_losses))  
            accuracy_loss.append(np.mean(accuracy))
            val_accuracy_loss.append(np.mean(val_accuracy))

            epoch_inc=np.append(epoch_inc,epoch)
            losses = []
            val_losses = []
            
            # Call Stagnation Early Stopping
            stop_flag = stagnation_es(Model, val_running_loss[-1])

            pbar.set_description(
                f"Epoch: {epoch}, lr: {my_lr:.2E}, tloss: {running_loss[-1]:.4E}, vloss: {val_running_loss[-1]:.4E}, acc: {val_accuracy_loss[-1]:.3E}, {stagnation_es.status}"
            )

            scheduler.step(val_running_loss[-1])

            if stop_flag:
                done = True
                print(f"\nStopping triggered: {stagnation_es.status}")
        if epoch==10 or epoch==2 or epoch %25 == 0:
            epoch_dir = os.path.join(Plt_Dir_epoch, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            save_metrics_plots(epoch, running_loss, val_running_loss, accuracy_loss, val_accuracy_loss, os.path.join(epoch_dir, 'train_loss.png'), os.path.join(epoch_dir, 'train_accuracy.png'))

            # Save interim best model (absolute best)
            if stagnation_es.absolute_best_wts is not None:
                torch.save(stagnation_es.absolute_best_wts, os.path.join(epoch_dir, "Model.pth"))
            else:
                torch.save(Model.state_dict(), os.path.join(epoch_dir, "Model.pth"))
            
            
            # Save metrics to a .mat file
            mat_metrics_path = os.path.join(epoch_dir, f"metrics_epoch_{epoch}.mat")
            sio.savemat(mat_metrics_path, {
                "epoch": epoch,
                "running_loss": running_loss,
                "val_running_loss": val_running_loss,
                "accuracy_loss": accuracy_loss,
                "val_accuracy_loss": val_accuracy_loss,
                "epoch_inc": epoch_inc,
            })

            
            if epoch%50==0 or epoch ==25 or epoch == 2:  
                n_train_vis = min(SizeTrain, 3)
                n_val_vis = min(SizeVal, 5)
                # Using helper function from utils
                save_fmc_images(AmpR, Rec, epoch_dir, 'T', n_train_vis)
                save_fmc_images(AmpV, Val, epoch_dir, 'V', n_val_vis)

                    
                
      
        #if epoch % 20 == 0 or epoch == 2:
        
            #writer.add_image("Train/Input", amp[0], epoch, dataformats='CHW')
            #writer.add_image("Train/Recon", recon[0], epoch, dataformats='CHW')
            #writer.add_image("Train/Error", torch.abs(amp - recon)[0], epoch, dataformats='CHW')
            
        for name, param in Model.named_parameters():
            writer.add_histogram(f'{name}/weights', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'{name}/gradients', param.grad, epoch)
      
            
        save_metrics_plots(epoch, running_loss, val_running_loss, accuracy_loss, val_accuracy_loss, f'./RunningLoss/{Plt_Dir}.png', f'./RunningAccuracy/{Plt_Dir}.png')
        
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
    #np.savez('./'+Plt_Dir+'/loss.npz',epoch_inc=np.arange(epoch),running_loss=running_loss,val_running_loss=val_running_loss)
    #np.savez('./'+Plt_Dir+'/Train.npz',FMCT=AmpR,Rec=Rec,BinR=BinR)
    #np.savez('./'+Plt_Dir+'/Valid.npz',FMCV=AmpV,Val=Val,BinV=BinV)
      
    #torch.save(Model.state_dict(),'./'+Plt_Dir+'/Model.pth') 
    # Save the TRUE best model (absolute min loss) instead of the last one
    if stagnation_es.absolute_best_wts is not None:
        print(f"Saving Best Model (Success) with Loss: {stagnation_es.absolute_best_loss:.6e}")
        torch.save(stagnation_es.absolute_best_wts, f'./{Plt_Dir}/Model.pth')
    else:
        # Fallback if no best model found (should not happen)
        torch.save(Model.state_dict(), f'./{Plt_Dir}/Model.pth')
    #print(f"Finished training with seed {seed_value}, learning rate {itrain}")
    writer.close()
    
    # Final metrics save at last epoch
    mat_metrics_path_final = os.path.join(Plt_Dir, f"metrics_final_epoch_{epoch}.mat")
    sio.savemat(mat_metrics_path_final, {
        "epoch": epoch,
        "running_loss": running_loss,
        "val_running_loss": val_running_loss,
        "accuracy_loss": accuracy_loss,
        "val_accuracy_loss": val_accuracy_loss,
        "epoch_inc": epoch_inc,
    })

    
    
    # Using helper function from utils for final images
    save_fmc_images(AmpR, Rec, Plt_Dir, 'T', SizeTrain//SizeTrain if SizeTrain > 0 else 0)
    save_fmc_images(AmpV, Val, Plt_Dir, 'V', SizeVal//SizeVal if SizeVal > 0 else 0)
        #tbcallback
    
      
    matlab_dir = f'./{Plt_Dir}/Matlab' + experiment_run_id
    os.makedirs(matlab_dir, exist_ok=True)
    
    def save_in_parts(data, base_filename, num_parts):
        num_fmc = num_parts
        for fmc_idx in tqdm(range(num_fmc)):
            fmc_data = data[fmc_idx, :, :]
            fmc_filename = os.path.join(matlab_dir, f"{base_filename}_FMC{fmc_idx+1}.mat")
            sio.savemat(fmc_filename, {f"{base_filename}_FMC{fmc_idx+1}": fmc_data})
            #print(f'Saved {filename} with shape {data_part.shape}')
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileViT-FPN-PixelShuffle Models")
    parser.add_argument("--model", dest="model_name", type=str, default="MobileViTv3_v1_dynamicFPNpixel2",
              choices=list(AVAILABLE_MODELS.keys()),
              help="Choose which model to train.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model names and exit.",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model_name in sorted(AVAILABLE_MODELS.keys()):
            print(f"- {model_name}")
        raise SystemExit(0)
    
    # Optional override of config based on args
    config.batch_size = args.batch_size
    
    main(args)