import copy
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import configs.config_mobileunet as config

try:
    import mat73
except ImportError:
    mat73 = None

try:
    import pytorch_msssim
except ImportError:
    pytorch_msssim = None

class EarlyStopping:
    def __init__(self, patience=config.patience, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False
  
class EarlyStopping2:
    def __init__(self, patience=config.patience*2, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping final triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False
 
        
# Fast NCC over FMC columns (mean over column vectors)
def ncc(y_true, y_pred, eps=1e-8):
    # Optional: subtract mean per vector (disabled)
    #y_true = y_true - torch.mean(y_true, dim=(0,1,3), keepdim=True)
    #y_pred = y_pred - torch.mean(y_pred, dim=(0,1,3), keepdim=True)
    # Dot product along time axis (dim=2)
    numerateur = torch.sum(y_true * y_pred, dim=2)
    # L2 norms along time axis (dim=2)
    norme_y_true = torch.sqrt(torch.sum(y_true ** 2, dim=2))
    norme_y_pred = torch.sqrt(torch.sum(y_pred ** 2, dim=2))
    denominateur = norme_y_true * norme_y_pred + eps

    # Per-vector NCC
    ncc = numerateur / denominateur
    # Mean over element axis, batch, and channel
    return torch.mean(ncc)


class NCC_MSE_Loss(nn.Module):
    def __init__(self, alpha=0.84, epsilon=1e-8):

        super(NCC_MSE_Loss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):

        mse = self.mse_loss(y_pred, y_true)

        # === 2. NCC term ===
        numerateur = torch.sum(y_true * y_pred, dim=2)
        norme_y_true = torch.sqrt(torch.sum(y_true ** 2, dim=2))
        norme_y_pred = torch.sqrt(torch.sum(y_pred ** 2, dim=2))

        ncc = numerateur / (norme_y_true * norme_y_pred + self.epsilon)
        ncc_mean = torch.mean(ncc)

        # === 3. Combined loss (MSE + NCC) ===
        loss = self.alpha * mse + (1 - self.alpha) * (1 - ncc_mean)

        return loss



"""
Alternate NCC implementation (kept for reference; not used).
"""



# MSSSIMLoss requires `pytorch-msssim` (see pip install note in MSSSIMLoss.forward).

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
    
    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
    
    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(pred, kernel_size=(3, 3), stride=1, padding=1)
        mu_y = F.avg_pool2d(target, kernel_size=(3, 3), stride=1, padding=1)
        sigma_x = F.avg_pool2d(pred ** 2, kernel_size=(3, 3), stride=1, padding=1) - mu_x ** 2
        sigma_y = F.avg_pool2d(target ** 2, kernel_size=(3, 3), stride=1, padding=1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(pred * target, kernel_size=(3, 3), stride=1, padding=1) - mu_x * mu_y
        
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        return 1 - ssim.mean()

class MSSSIMLoss(nn.Module):
    def __init__(self):
        super(MSSSIMLoss, self).__init__()
    
    def forward(self, pred, target):
        if pytorch_msssim is None:
            raise ImportError("pytorch_msssim is required for MSSSIMLoss. Install with `pip install pytorch-msssim`.")
        return 1 - pytorch_msssim.ms_ssim(pred, target, data_range=1.0, size_average=True, win_size=11)

class MixLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super(MixLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, pred, target):
        return self.alpha * MSSSIMLoss()(pred, target) + (1 - self.alpha) * L1Loss()(pred, target)

# Utilisation des fonctions de perte PyTorch existantes
#MSELoss = nn.MSELoss()


class RDropLoss(nn.Module):
    def __init__(self, lambda_rdrop=1.0, reduction='batchmean'):
        super(RDropLoss, self).__init__()
        self.lambda_rdrop = lambda_rdrop
        self.mse_loss = nn.MSELoss()
        self.reduction = reduction

    def forward(self, y_pred1, y_pred2, y_true):
        # MSE between each prediction and ground truth
        mse1 = self.mse_loss(y_pred1, y_true)
        mse2 = self.mse_loss(y_pred2, y_true)
        mse_total = 0.5 * (mse1 + mse2)

        # Symmetric KL between the two predictions
        log_probs1 = F.log_softmax(y_pred1, dim=-1)
        log_probs2 = F.log_softmax(y_pred2, dim=-1)
        probs1 = F.softmax(y_pred1, dim=-1)
        probs2 = F.softmax(y_pred2, dim=-1)

        kl1 = F.kl_div(log_probs1, probs2.detach(), reduction=self.reduction)
        kl2 = F.kl_div(log_probs2, probs1.detach(), reduction=self.reduction)
        kl_total = 0.5 * (kl1 + kl2)

        # Total loss
        total_loss = mse_total + self.lambda_rdrop * kl_total
        return total_loss
class FMC_Dataset(torch.utils.data.Dataset):
    def __init__(self, mat_path, use_mat73=True, max_cache_size=100):
        self.use_mat73 = use_mat73
        self.mat_path = mat_path
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size

        if use_mat73:
            if mat73 is None:
                raise ImportError("mat73 is required when use_mat73=True. Install with `pip install mat73`.")
            mat = mat73.loadmat(mat_path)
        else:
            mat = sio.loadmat(mat_path)

        self.FMC = np.transpose(mat['FMC'], (2, 0, 1))  # (N, H, W)
        self.Bin = np.transpose(mat['Bin'], (2, 0, 1))

    def __len__(self):
        return self.FMC.shape[0]

    def __getitem__(self, idx):
        if idx in self.cache:
            amp, bin = self.cache[idx]
        else:
            amp = self.FMC[idx]
            bin = self.Bin[idx]

            amp = torch.from_numpy(amp).float().unsqueeze(0)  # (1, H, W)
            bin = torch.from_numpy(bin).float().unsqueeze(0)

            # Cache management
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)  # remove oldest item
            self.cache[idx] = (amp, bin)

        return amp, bin  # tensors on CPU





def save_fmc_images(Amp, Rec, save_dir, prefix, num_samples):
    num_cols = Amp.shape[2]
    figsize = (max(1.5, 3 * num_cols / 1024), 5)
    
    for i in tqdm(range(num_samples), desc=f'Saving {prefix} images'):
        Amp[i,:,:] = Amp[i,:,:] / np.max(np.absolute(Amp[i,:,:]))
        Rec[i,:,:] = Rec[i,:,:] / np.max(np.absolute(Rec[i,:,:]))
        
        # Original Amplitude Plot
        plt.figure(figsize=figsize)
        plt.imshow(Amp[i,:,:])
        plt.xlabel("Element axis")
        plt.ylabel("Time Increment")
        plt.set_cmap('seismic')
        plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
        if num_cols == 64:
            plt.xticks([0, num_cols])
        else:
            plt.xticks([0, num_cols // 2, num_cols])            
        plt.clim(-1,1)
        cbar=plt.colorbar()
        cbar.set_label('Amplitude (linear)')
        plt.savefig(os.path.join(save_dir, f'{prefix}_{i}_Amp.png'),dpi=1200,bbox_inches='tight')
        plt.close()
        
        # Reconstructed Amplitude Plot
        plt.figure(figsize=figsize)
        plt.imshow(Rec[i,:,:])
        plt.xlabel("Element axis")
        plt.ylabel("Time Increment")
        plt.set_cmap('seismic')
        plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
        if num_cols == 64:
            plt.xticks([0, num_cols])
        else:
            plt.xticks([0, num_cols // 2, num_cols])            
        plt.clim(-1,1)
        cbar=plt.colorbar()
        cbar.set_label('Amplitude (linear)')
        plt.savefig(os.path.join(save_dir, f'{prefix}_{i}_Rec.png'),dpi=1200,bbox_inches='tight')
        plt.close()
        
        # Error Plot
        Err = np.absolute(Amp[i,:,:] - Rec[i,:,:])
        plt.figure(figsize=figsize)
        plt.imshow(Err)
        plt.xlabel("Element axis")
        plt.ylabel("Time Increment")
        plt.set_cmap('gray_r')
        plt.yticks([0,512,1024,1536,2048,2560,3072,3584,4096])
        if num_cols == 64:
            plt.xticks([0, num_cols])
        else:
            plt.xticks([0, num_cols // 2, num_cols])            
        plt.clim(0,1)
        cbar=plt.colorbar()
        cbar.set_label('Error (absolute)')
        plt.savefig(os.path.join(save_dir, f'{prefix}_{i}_Error.png'),dpi=1200,bbox_inches='tight')
        plt.close()



def save_metrics_plots(epoch, running_loss, val_running_loss, accuracy_loss, val_accuracy_loss, loss_path, acc_path):
    # Ensure directories exist
    os.makedirs(os.path.dirname(loss_path), exist_ok=True)
    os.makedirs(os.path.dirname(acc_path), exist_ok=True)

    min_train_loss = np.ones(epoch) * np.min(running_loss)
    min_val_loss = np.ones(epoch) * np.min(val_running_loss)
    
    # Check lengths for accuracy arrays
    len_acc = len(accuracy_loss)
    len_val_acc = len(val_accuracy_loss)
    
    min_train_acc = np.ones(len_acc) * (np.max(accuracy_loss) if len_acc > 0 else 0)
    min_val_acc = np.ones(len_val_acc) * (np.max(val_accuracy_loss) if len_val_acc > 0 else 0)

    # Plot Loss
    plt.figure()
    plt.plot(np.arange(epoch), running_loss, 'b-', label='Training')
    plt.plot(np.arange(epoch), min_train_loss, 'b--')
    plt.plot(np.arange(epoch), val_running_loss, 'g-', label='Validation')
    plt.plot(np.arange(epoch), min_val_loss, 'g--')
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.title(f'Minimal Train Loss : {np.min(running_loss):.4E} / Val Loss : {np.min(val_running_loss):.4E}')
    plt.legend()
    plt.savefig(loss_path, dpi=1200)
    plt.close()

    # Plot Accuracy
    if len_acc > 0 and len_val_acc > 0:
        plt.figure()
        plt.plot(np.arange(len_acc), accuracy_loss, 'b-', label='Training')
        plt.plot(np.arange(len_acc), min_train_acc, 'b--')
        plt.plot(np.arange(len_val_acc), val_accuracy_loss, 'g-', label='Validation')
        plt.plot(np.arange(len_val_acc), min_val_acc, 'g--')
        plt.xlabel("Training step")
        plt.ylabel("Accuracy")
        plt.yscale("log")
        plt.grid(True)
        plt.suptitle('Training and Validation Accuracy Across Models', fontsize=14, y=1.10)
        plt.title(f'Train: {np.max(accuracy_loss):.4E} | Val: {np.max(val_accuracy_loss):.4E}')
        plt.legend()
        plt.savefig(acc_path, dpi=1200)
        plt.close()

