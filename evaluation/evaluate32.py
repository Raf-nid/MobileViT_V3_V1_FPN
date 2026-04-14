"""Legacy MobileNetV2-UNet script: 32-panel composite FMC figures. Use ``evaluate_loop2.py`` for main validation."""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import config_MobileUNET as config
from MobileUnet_V2 import MobileNetV2_unet
# from MobileUnet_V2_CRM import MobileNetV2_unet_CRM
import os
from os.path import dirname, join as pjoin
import scipy.io as sio
import mat73
from utils import EarlyStopping, ncc
import datetime
import h5py
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # all GPUs
    # optional: stricter determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def main():
    # Performance-oriented CUDA flags
    torch.backends.cudnn.benchmark = True  # cuDNN autotune for convolutions
    torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 matmul on Ampere+

    # Device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")

    # Matplotlib font settings
    size = 10
    font = {'family': 'sans-serif',
            'sans-serif': 'Times New Roman',
            'size': size}
    plt.rc('font', **font)

    # Timestamp for output folders
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")

    # Fixed seed
    seed_value = 193157014  # Change if needed
    set_seed(seed_value)
    print("Seed used:", seed_value)

    # Learning rate from config (scaled by 1e-6)
    learning_rate = np.multiply(config.learning_rate, 1e-6)

    # Path to saved weights (edit for your setup)
    model_path = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/MobileUnet_V2_modif/Test_End_seed_lr_binaireDIV32_200e-6_20250212_140556_500_epochs/Model.pth"

    # Single-iteration example
    for truc in range(1):
        if truc == 0:
            learning_rate = config.learning_rate
            # Create figure output directory
            Plt_Dir = f'Eval_32DivBinaire_{date_str}_500_epochs'
            Plt_Dir_epoch = f'Eval_32interpPWMampDS2{date_str}_epochs_intermediaire_500_epochs'
            # Path to data file
            data_dir = pjoin(os.getcwd(), 'Data', config.Case, 'FMC_Div32.mat')

        # Load data
        if config.Vmat == 7.3:
            mat_data = mat73.loadmat(data_dir)
            FMC = mat_data['FMC']
            Bin = mat_data['Bin']
        else:
            mat_data = sio.loadmat(data_dir)
            FMC = mat_data['FMC']
            Bin = mat_data['Bin']

        # Move dim 2 to front: (N, H, W)
        FMC = np.transpose(FMC, (2, 0, 1))
        Bin = np.transpose(Bin, (2, 0, 1))

        # To tensors + channel dimension
        FMC = torch.from_numpy(FMC).float().to(device)
        FMC = torch.unsqueeze(FMC, 1)

        Bin = torch.from_numpy(Bin).float().to(device)
        Bin = torch.unsqueeze(Bin, 1)

        # Build model and load weights
        Model = MobileNetV2_unet(pre_trained=None).to(device)
        Model.load_state_dict(torch.load(model_path, map_location=device))

        # Dataset and DataLoader
        Dataset_full = TensorDataset(FMC, Bin)
        # Full tensor used as validation set
        val_loader = DataLoader(dataset=Dataset_full, batch_size=config.batch_size, shuffle=False, drop_last=True)

        # Loss function
        losscalc = nn.MSELoss()

        # Logging buffers (unused in this eval-only script)
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

        epoch = 0
        pbar = tqdm(total=1, leave=True)

        # Accumulators for predictions
        Val = None
        BinV = None
        AmpV = None

        epoch += 1
        pbar.update(1)

        Model.train()

        cnt = 0

        # Validation loop
        with torch.no_grad():
            Model.eval()
            val_losses = []
            val_accuracy = []

            for (amp_batch, bin_batch) in val_loader:
                amp_batch, bin_batch = amp_batch.to(device), bin_batch.to(device)
                recon_batch = Model(bin_batch)
                loss_val = losscalc(amp_batch, recon_batch)

                if epoch == 1:
                    initvalloss = loss_val.item()

                val_losses = np.append(val_losses, loss_val.item() / initvalloss)
                val_accuracy.append(ncc(amp_batch, recon_batch).item())

                # Append batch outputs
                recon_np = torch.squeeze(recon_batch, 1).detach().cpu().numpy()
                amp_np = torch.squeeze(amp_batch, 1).detach().cpu().numpy()
                bin_np = torch.squeeze(bin_batch, 1).detach().cpu().numpy()

                if cnt == 0:
                    Val = recon_np
                    AmpV = amp_np
                    BinV = bin_np
                else:
                    Val = np.concatenate((Val, recon_np), axis=0)
                    AmpV = np.concatenate((AmpV, amp_np), axis=0)
                    BinV = np.concatenate((BinV, bin_np), axis=0)
                cnt += 1

            avg_val_loss = np.mean(val_losses)
            avg_val_accuracy = np.mean(val_accuracy)

        SizeVal = Val.shape[0]

        # Ensure output directory exists
        os.makedirs(Plt_Dir, exist_ok=True)

        # --- Build composite strips of 32 images ---
        # For each group of 32, concatenate along width.
        Ampi_list = []
        Vali_list = []
        Bini_list = []

        for i in tqdm(range(SizeVal // 32), desc="Building composite images"):
            row_amp = []
            row_val = []
            row_bin = []
            for j in range(32):
                idx = i * 32 + j
                # Ground-truth amplitudes
                amp_img = AmpV[idx, :, :]
                max_amp = np.max(np.abs(amp_img)) if np.max(np.abs(amp_img)) != 0 else 1
                amp_img_norm = amp_img / max_amp
                row_amp.append(amp_img_norm)

                # Reconstructions
                val_img = Val[idx, :, :]
                max_val = np.max(np.abs(val_img)) if np.max(np.abs(val_img)) != 0 else 1
                val_img_norm = val_img / max_val
                row_val.append(val_img_norm)

                # Binary inputs (no normalization)
                bin_img = BinV[idx, :, :]
                row_bin.append(bin_img)

            # Horizontal concat of 32 panels
            Ampi_img = np.concatenate(row_amp, axis=1)
            Vali_img = np.concatenate(row_val, axis=1)
            Bini_img = np.concatenate(row_bin, axis=1)

            Ampi_list.append(Ampi_img)
            Vali_list.append(Vali_img)
            Bini_list.append(Bini_img)

        # Save composite PNGs
        for i, amp_img in enumerate(Ampi_list):
            plt.figure(figsize=(20,5))
            plt.imshow(amp_img, cmap='seismic', vmin=-1, vmax=1)
            plt.xlabel("Element axis")
            plt.ylabel("Time Increment")
            plt.yticks([0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096])
            # xticks span 32 concatenated panels
            plt.xticks(np.linspace(0, amp_img.shape[1]-1, 33))
            cbar = plt.colorbar()
            cbar.set_label('Amplitude (linear)')
            plt.savefig(os.path.join(Plt_Dir, f'V_{i}_Amp.png'), dpi=1200, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(20,5))
            plt.imshow(Vali_list[i], cmap='seismic', vmin=-1, vmax=1)
            plt.xlabel("Element axis")
            plt.ylabel("Time Increment")
            plt.yticks([0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096])
            plt.xticks(np.linspace(0, Vali_list[i].shape[1]-1, 33))
            cbar = plt.colorbar()
            cbar.set_label('Amplitude (linear)')
            plt.savefig(os.path.join(Plt_Dir, f'V_{i}_Rec.png'), dpi=1200, bbox_inches='tight')
            plt.close()

            Err = np.abs(Ampi_list[i] - Vali_list[i])
            plt.figure(figsize=(20,5))
            plt.imshow(Err, cmap='inferno', vmin=0, vmax=1)
            plt.xlabel("Element axis")
            plt.ylabel("Time Increment")
            plt.xticks(np.linspace(0, Err.shape[1]-1, 33))
            plt.yticks([0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096])
            cbar = plt.colorbar()
            cbar.set_label('Error (absolute)')
            plt.savefig(os.path.join(Plt_Dir, f'V_{i}_Error.png'), dpi=1200, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(20,5))
            plt.imshow(Bini_list[i], cmap='seismic', vmin=-1, vmax=1)
            plt.xlabel("Element axis")
            plt.ylabel("Time Increment")
            plt.xticks(np.linspace(0, Bini_list[i].shape[1]-1, 33))
            plt.yticks([0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096])
            cbar = plt.colorbar()
            cbar.set_label('Amplitude (binary)')
            plt.savefig(os.path.join(Plt_Dir, f'V_{i}_Bin.png'), dpi=1200, bbox_inches='tight')
            plt.close()

        # Save composite PNGs au format Matlab
        matlab_dir = os.path.join(Plt_Dir, 'Matlab')
        os.makedirs(matlab_dir, exist_ok=True)
        print('Matlab')
        def save_in_parts(data_list, base_filename):
            for idx, data in enumerate(data_list):
                fmc_filename = os.path.join(matlab_dir, f"{base_filename}_FMC{idx+1}.mat")
                sio.savemat(fmc_filename, {f"{base_filename}_FMC{idx+1}": data})

        save_in_parts(Ampi_list, "V_FMC")
        save_in_parts(Vali_list, "V_Rec")
        save_in_parts(Bini_list, "V_Bin")


if __name__ == "__main__":
    main()
