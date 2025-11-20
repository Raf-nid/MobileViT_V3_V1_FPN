import os
import glob
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as func
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
from tqdm import tqdm

# Imports locaux (assurez-vous que ces fichiers sont présents dans le repo)
import config_MobileUNET as config
from utils import EarlyStopping, ncc, NCC_MSE_Loss
from mobilevit_v3_v1_Pixel2 import MobileViTv3_v1_dynamicFPNpixel2

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class MatDataset(Dataset):
    def __init__(self, directory, device='cuda'):
        self.files = sorted(glob.glob(os.path.join(directory, '*.mat')))
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        truncated_name = base_name[-8:]

        with h5py.File(filepath, 'r') as f:
            fmc = torch.tensor(f['FMC'][()].astype('float32'))
            bin_data = torch.tensor(f['Bin'][()].astype('float32'))
    
        fmc = fmc.permute(1, 0).unsqueeze(0).to(self.device)
        bin_data = bin_data.permute(1, 0).unsqueeze(0).to(self.device)
    
        return fmc, bin_data, truncated_name

def save_metrics_plot(running_loss, val_running_loss, accuracy_loss, val_accuracy_loss, save_dir):
    epochs = np.arange(len(running_loss)) + 1
    
    # Loss Plot
    plt.figure()
    plt.plot(epochs, running_loss, 'b-', label='Training')
    plt.plot(epochs, val_running_loss, 'g-', label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300)
    plt.close()

    # Accuracy Plot
    plt.figure()
    plt.plot(epochs, accuracy_loss, 'b-', label='Training')
    plt.plot(epochs, val_accuracy_loss, 'g-', label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("NCC Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'), dpi=300)
    plt.close()

def save_visual_results(amp, recon, epoch, save_dir, prefix='Val'):
    # Normalization for visualization
    amp = amp / np.max(np.abs(amp))
    recon = recon / np.max(np.abs(recon))
    error = np.abs(amp - recon)

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    # Ground Truth
    im0 = axes[0].imshow(amp, cmap='seismic', aspect='auto')
    axes[0].set_title('Ground Truth')
    im0.set_clim(-1, 1)
    plt.colorbar(im0, ax=axes[0])

    # Reconstruction
    im1 = axes[1].imshow(recon, cmap='seismic', aspect='auto')
    axes[1].set_title('Reconstruction')
    im1.set_clim(-1, 1)
    plt.colorbar(im1, ax=axes[1])

    # Error
    im2 = axes[2].imshow(error, cmap='inferno', aspect='auto')
    axes[2].set_title('Absolute Error')
    im2.set_clim(0, 1)
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_Epoch_{epoch}.png'), dpi=300)
    plt.close()

def main():
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 69738009
    set_seed(SEED)
    
    DATA_DIR_TRAIN = config.train_data_path 
    DATA_DIR_VALID = config.valid_data_path
    
    # Run identification
    now = datetime.datetime.now()
    run_id = f"Run_{now.strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join('./results', run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=os.path.join('./runs', run_id))

    # --- Data Loading ---
    train_dataset = MatDataset(directory=DATA_DIR_TRAIN, device=DEVICE)
    val_dataset = MatDataset(directory=DATA_DIR_VALID, device=DEVICE)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    # --- Model Setup ---
    # Example instantiation, modify parameters as needed
    model = MobileViTv3_v1_dynamicFPNpixel2((4096, 1024), 'xx_small4', 1000, (32,32)).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    early_stopping = EarlyStopping(patience=config.patience)

    # --- Training Loop ---
    running_loss_history = []
    val_running_loss_history = []
    accuracy_history = []
    val_accuracy_history = []

    print(f"Starting training on {DEVICE} with seed {SEED}...")

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_losses = []
        epoch_acc = []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}", leave=True)
        
        for amp, bin_data, _ in loop:
            # Ensure tensors are on correct device (handled by dataset, but good practice)
            amp, bin_data = amp.to(DEVICE), bin_data.to(DEVICE)

            optimizer.zero_grad()
            recon = model(bin_data)
            loss = criterion(amp, recon)
            
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            acc = ncc(amp, recon).item()
            epoch_acc.append(acc)
            
            loop.set_description(f"Epoch {epoch} | Loss: {loss.item():.4e} | NCC: {acc:.4f}")

        avg_train_loss = np.mean(epoch_losses)
        avg_train_acc = np.mean(epoch_acc)
        
        running_loss_history.append(avg_train_loss)
        accuracy_history.append(avg_train_acc)

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", avg_train_acc, epoch)

        # --- Validation ---
        model.eval()
        val_losses = []
        val_acc = []
        
        # Holders for visualization (save first batch)
        sample_amp = None
        sample_recon = None

        with torch.no_grad():
            for idx, (amp, bin_data, _) in enumerate(val_loader):
                amp, bin_data = amp.to(DEVICE), bin_data.to(DEVICE)
                recon = model(bin_data)
                
                loss = criterion(amp, recon)
                val_losses.append(loss.item())
                val_acc.append(ncc(amp, recon).item())

                if idx == 0:
                    sample_amp = torch.squeeze(amp, 1).cpu().numpy()[0]
                    sample_recon = torch.squeeze(recon, 1).cpu().numpy()[0]

        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_acc)
        
        val_running_loss_history.append(avg_val_loss)
        val_accuracy_history.append(avg_val_acc)

        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", avg_val_acc, epoch)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # --- Saving & Visuals ---
        if epoch % 10 == 0 or epoch == 1:
            save_metrics_plot(running_loss_history, val_running_loss_history, 
                              accuracy_history, val_accuracy_history, output_dir)
            save_visual_results(sample_amp, sample_recon, epoch, output_dir)
            
            # Save checkpoint
            torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch}.pth'))

        # Early Stopping Check
        if early_stopping(model, avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
    # Final Save
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_final.pth'))
    
    # Save metrics to .mat
    sio.savemat(os.path.join(output_dir, "training_metrics.mat"), {
        "train_loss": running_loss_history,
        "val_loss": val_running_loss_history,
        "train_acc": accuracy_history,
        "val_acc": val_accuracy_history
    })

    writer.close()
    print("Training completed.")

if __name__ == "__main__":
    main()