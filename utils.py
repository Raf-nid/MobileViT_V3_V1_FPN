import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import mat73
import pytorch_msssim
from collections import OrderedDict
from typing import Optional, Tuple, Dict


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience: int = 20, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model: nn.Module, val_loss: float) -> bool:
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
    """
    Secondary EarlyStopping class with extended patience (default: 2x standard).
    Useful for multi-stage training or fine-tuning.
    """
    def __init__(self, patience: int = 40, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model: nn.Module, val_loss: float) -> bool:
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


def ncc(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculates the Normalized Cross-Correlation (NCC).
    Assumes inputs are centered or uses dot product logic directly.
    """
    # Scalar product along length dimension (dim=2)
    numerateur = torch.sum(y_true * y_pred, dim=2)
    
    # Norms
    norme_y_true = torch.sqrt(torch.sum(y_true ** 2, dim=2))
    norme_y_pred = torch.sqrt(torch.sum(y_pred ** 2, dim=2))
    
    denominateur = norme_y_true * norme_y_pred + eps
    ncc_val = numerateur / denominateur
    
    # Mean over batch and channels
    return torch.mean(ncc_val)


class NCC_MSE_Loss(nn.Module):
    """
    Hybrid Loss combining MSE and NCC.
    Loss = alpha * MSE + (1 - alpha) * (1 - NCC)
    """
    def __init__(self, alpha: float = 0.84, epsilon: float = 1e-8):
        super(NCC_MSE_Loss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse = self.mse_loss(y_pred, y_true)

        numerateur = torch.sum(y_true * y_pred, dim=2)
        norme_y_true = torch.sqrt(torch.sum(y_true ** 2, dim=2))
        norme_y_pred = torch.sqrt(torch.sum(y_pred ** 2, dim=2))

        ncc_val = numerateur / (norme_y_true * norme_y_pred + self.epsilon)
        ncc_mean = torch.mean(ncc_val)

        loss = self.alpha * mse + (1 - self.alpha) * (1 - ncc_mean)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target))


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(pred, kernel_size=(3, 3), stride=1, padding=1)
        mu_y = F.avg_pool2d(target, kernel_size=(3, 3), stride=1, padding=1)
        
        sigma_x = F.avg_pool2d(pred ** 2, kernel_size=(3, 3), stride=1, padding=1) - mu_x ** 2
        sigma_y = F.avg_pool2d(target ** 2, kernel_size=(3, 3), stride=1, padding=1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(pred * target, kernel_size=(3, 3), stride=1, padding=1) - mu_x * mu_y
        
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        
        return 1 - ssim.mean()


class MSSSIMLoss(nn.Module):
    def __init__(self):
        super(MSSSIMLoss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Requires pytorch_msssim installed via pip
        return 1 - pytorch_msssim.ms_ssim(pred, target, data_range=1.0, size_average=True, win_size=11)


class MixLoss(nn.Module):
    """
    Combines MS-SSIM and L1 Loss.
    """
    def __init__(self, alpha: float = 0.84):
        super(MixLoss, self).__init__()
        self.alpha = alpha
        self.msssim = MSSSIMLoss()
        self.l1 = L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.msssim(pred, target) + (1 - self.alpha) * self.l1(pred, target)


class RDropLoss(nn.Module):
    """
    Regularization Drop Loss: Enforces consistency between two different dropouts of the same input.
    """
    def __init__(self, lambda_rdrop: float = 1.0, reduction: str = 'batchmean'):
        super(RDropLoss, self).__init__()
        self.lambda_rdrop = lambda_rdrop
        self.mse_loss = nn.MSELoss()
        self.reduction = reduction

    def forward(self, y_pred1: torch.Tensor, y_pred2: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # MSE Loss against ground truth
        mse1 = self.mse_loss(y_pred1, y_true)
        mse2 = self.mse_loss(y_pred2, y_true)
        mse_total = 0.5 * (mse1 + mse2)

        # KL Divergence between predictions (Symmetric)
        log_probs1 = F.log_softmax(y_pred1, dim=-1)
        log_probs2 = F.log_softmax(y_pred2, dim=-1)
        probs1 = F.softmax(y_pred1, dim=-1)
        probs2 = F.softmax(y_pred2, dim=-1)

        kl1 = F.kl_div(log_probs1, probs2.detach(), reduction=self.reduction)
        kl2 = F.kl_div(log_probs2, probs1.detach(), reduction=self.reduction)
        kl_total = 0.5 * (kl1 + kl2)

        return mse_total + self.lambda_rdrop * kl_total


class FMC_Dataset(torch.utils.data.Dataset):
    """
    Custom Dataset for FMC data loading from MAT files with caching.
    Supports both v7.3 (h5py based via mat73) and legacy MAT formats.
    """
    def __init__(self, mat_path: str, use_mat73: bool = True, max_cache_size: int = 100):
        self.use_mat73 = use_mat73
        self.mat_path = mat_path
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size

        if use_mat73:
            mat = mat73.loadmat(mat_path)
        else:
            mat = sio.loadmat(mat_path)

        # Transpose to (N, H, W)
        self.FMC = np.transpose(mat['FMC'], (2, 0, 1))
        self.Bin = np.transpose(mat['Bin'], (2, 0, 1))

    def __len__(self) -> int:
        return self.FMC.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx in self.cache:
            amp, bin_data = self.cache[idx]
        else:
            amp = self.FMC[idx]
            bin_data = self.Bin[idx]

            # Convert to Tensor and add Channel dim: (1, H, W)
            amp = torch.from_numpy(amp).float().unsqueeze(0)
            bin_data = torch.from_numpy(bin_data).float().unsqueeze(0)

            # LRU Cache management
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)  # Remove oldest
            self.cache[idx] = (amp, bin_data)

        # Note: Data is returned on CPU. Move to GPU in the training loop.
        return amp, bin_data


if __name__ == "__main__":
    # Example tensors (batch_size=2, channels=1, H=4096, W=1024)
    y_true = torch.randn(2, 1, 4096, 1024)
    y_pred = torch.randn(2, 1, 4096, 1024, requires_grad=True)
    
    # Test MixLoss
    try:
        loss_fn = MixLoss()
        loss = loss_fn(y_pred, y_true)
        print(f"Loss MixLoss: {loss.item():.4f}")
        loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print(f"Error testing MixLoss: {e}")