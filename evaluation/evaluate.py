# -*- coding: utf-8 -*-
"""
Single-configuration FMC evaluation (older layout). For batched multi-experiment runs, use
``evaluate_loop2.py``.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import datetime
import os
from os.path import join as pjoin
import scipy.io as sio
import mat73
from tqdm import tqdm
import glob
from torch.utils.data import Dataset, DataLoader
import h5py
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Importation des modules specifiques a votre projet
import config_MobileUNET as config
from MobileUnet_V2_CRM import MobileNetV2_dynamicFPN
from utils import EarlyStopping, ncc, NCC_MSE_Loss  # si vous en avez besoin
from mobilevit_v3_v1 import MobileViTv3_v1_dynamicFPN
from Model import UNETred # legacy: cuda device note
from models.segmentation.mobilevit_v3_pixel2 import MobileViTv3_v1_dynamicFPNpixel2

def set_seed(seed):
    """
    Set random seed for reproducible results.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

class MatDataset(Dataset):
    def __init__(self, directory, device='cuda:1'):
        self.files = sorted(glob.glob(os.path.join(directory, '*.mat')))
        if not self.files:
            raise ValueError(f"No .mat files found in directory: {directory}")

        self.filenames = [os.path.basename(f) for f in self.files]
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        base = os.path.splitext(self.filenames[idx])[0]
        truncated = base + '.mat'

        try:
            with h5py.File(filepath, 'r') as f:
                fmc = torch.tensor(f['FMC'][()].astype('float32'))
                bin_data = torch.tensor(f['Bin'][()].astype('float32'))
        except Exception as e:
            print(f"Erreur lors du chargement du fichier {filepath}: {e}")
            raise

        # Correction de l'orientation
        fmc = fmc.permute(1, 0)
        bin_data = bin_data.permute(1, 0)

        # Ajout de la dimension channel et deplacement vers le device
        fmc = fmc.unsqueeze(0).to(self.device)
        bin_data = bin_data.unsqueeze(0).to(self.device)

        return fmc, bin_data, truncated

def safe_normalize(data, eps=1e-8):
    """
    Safe normalization to avoid division by zero.
    """
    max_val = np.max(np.abs(data))
    if max_val > eps:
        return data / max_val
    else:
        return data

def create_and_save_plot(data, title, filename, cmap='seismic', vmin=None, vmax=None,
                        colorbar_label='Amplitude'):
    """
    Utility to build and save plots.
    """
    plt.figure(figsize=(3, 5))
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel("Element axis")
    plt.ylabel("Time Increment")
    plt.yticks([0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096])
    plt.xticks([0, 512, 1024])
    cbar = plt.colorbar()
    cbar.set_label(colorbar_label)
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

def main():
    try:
        # CUDA performance flags
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        device = torch.device("cuda:1")
        print(f"Using device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")

        # Parametres d'affichage
        size = 10
        plt.rc('font', family='Times New Roman', size=size)

        # Timestamp for output folders
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")

        # Generer une seed aleatoire et l'appliquer
        seed_value = np.random.randint(1e6, int(1e9))
        set_seed(seed_value)
        print(f"Seed utilisee: {seed_value}")
        # Create directory for plots


        model_type = "MbViT_XXS4_FF8_Normalisation_NW_W" #'UNETred_copper_50_50_BruitDuet_fixe_NW_et_W'


        num_epochs = getattr(config, 'num_epochs', 'unknown')
        Plt_Dir = f'Evaluation_{model_type}_{date_str}_{num_epochs}_epochs'
        os.makedirs(Plt_Dir, exist_ok=True)
        print(f"Dossier de sauvegarde cree: {Plt_Dir}")

        # Repertoire des donnees de validation
        data_dir_valid = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Verif_8x8_NW_W_NewAmp"

        if not os.path.exists(data_dir_valid):
            raise FileNotFoundError(f"Le repertoire de donnees n'existe pas: {data_dir_valid}")

        # Chargement du modele pre-entraine
        try:
            Model = MobileViTv3_v1_dynamicFPN((4096, 64), 'xx_small4', 1000, (32,32)).to(device)
            #Model = UNETred().to(device)
            #Model = torch.compile(Model)
            pretrained_path = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/MobileUnet_V2_modif/Test_End_seed_lr_MbViT_XXSMALL4_FF8_batchsize16_NW_plus_Wedge_BruitFixe_BruitDuet200e-6_20251119_132327_250_epochs_seed_69738009/Model.pth"
            #"/mnt/STORAGE-1/home/rniddam/UNET_Binary/MobileUnet_V2_modif/AAA_Test_End_seed_lr_MobileVitv3_v1_XXSMALL4_dropout00_batchsize2_NW_plus_Wedge_BruitFixe_BruitDuet_rd32prem_rd32last_50_50New200e-6_20250731_101341_350_epochs_seed_69738009/Model.pth" #XXSMALL4

            if os.path.exists(pretrained_path):
                Model.load_state_dict(torch.load(pretrained_path, map_location=device))
                print(f"Modele pre-entraine charge depuis {pretrained_path}")
            else:
                raise FileNotFoundError(f"Modele pre-entraine non trouve: {pretrained_path}")

        except Exception as e:
            print(f"Erreur lors du chargement du modele: {e}")
            raise

        # Validation DataLoader
        try:
            val_dataset = MatDataset(directory=data_dir_valid, device=device)
            val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
            print(f"Dataset created with {len(val_dataset)} samples")
        except Exception as e:
            print(f"Erreur lors de la creation du dataset: {e}")
            raise

        # Loss function
        losscalc = nn.MSELoss()

        # Passage en mode evaluation
        Model.eval()
        val_losses = []
        val_accuracy = []



        print("Debut de l'evaluation...")
        start_time = datetime.datetime.now()

        with torch.no_grad():
            for batch_idx, (ampV, binV, truncated_names) in enumerate(tqdm(val_loader, desc="Validation")):
                try:
                    ampV = ampV.to(device)
                    binV = binV.to(device)

                    reconV = Model(binV)
                    loss_val = losscalc(ampV, reconV)
                    ncc_val = ncc(ampV, reconV)

                    val_losses.append(loss_val.item())
                    val_accuracy.append(ncc_val.item())

                    print(f"Batch {batch_idx + 1}: Loss = {loss_val.item():.6f}, NCC = {ncc_val.item():.6f}")

                    # Directory for MATLAB exports
                    matlab_dir = os.path.join(Plt_Dir, f'Matlab_{model_type}')
                    os.makedirs(matlab_dir, exist_ok=True)
                    # === dynamic export ===
                    for i in range(reconV.size(0)):
                        base_name = truncated_names[i].replace('.mat', '')
                        amp_np = ampV[i, 0].cpu().numpy()
                        rec_np = reconV[i, 0].cpu().numpy()
                        bin_np = binV[i, 0].cpu().numpy()

                        # save images
                        create_and_save_plot(safe_normalize(amp_np), "", os.path.join(Plt_Dir, f'{base_name}_Amp.png'),
                                             'seismic', -1, 1, 'Amplitude')
                        create_and_save_plot(safe_normalize(rec_np), "", os.path.join(Plt_Dir, f'{base_name}_Rec.png'),
                                             'seismic', -1, 1, 'Amplitude')
                        create_and_save_plot(np.abs(amp_np - rec_np), "", os.path.join(Plt_Dir, f'{base_name}_Err.png'),
                                             'inferno', 0, 1, 'Erreur')
                        create_and_save_plot(bin_np, "", os.path.join(Plt_Dir, f'{base_name}_Bin.png'),
                                             'seismic', 0, 1, 'Binaire')

                        # save MATLAB
                        sio.savemat(os.path.join(matlab_dir, f'{base_name}_Amp.mat'), {'Amp': amp_np})
                        sio.savemat(os.path.join(matlab_dir, f'{base_name}_Rec.mat'), {'Rec': rec_np})

                    # Lib�ration m�moire
                    del ampV, binV, reconV
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Erreur lors du traitement du batch {batch_idx}: {e}")
                    continue


        duration = datetime.datetime.now() - start_time
        print(f"Temps total de validation: {duration}")

        if val_losses:
            print(f"Loss moyenne: {np.mean(val_losses):.6f}")
            print(f"Loss std: {np.std(val_losses):.6f}")
            print(f"Accuracy moyenne: {np.mean(val_accuracy):.6f}")
            print(f"Accuracy std: {np.std(val_accuracy):.6f}")
        else:
            print("No data processed successfully!")
            return



        # save statistics
#        stats = {
#            'seed': seed_value,+
#            'mean_loss': np.mean(val_losses),
#            'std_loss': np.std(val_losses),
#            'mean_accuracy': np.mean(val_accuracy),
#            'std_accuracy': np.std(val_accuracy),
#            'validation_time': str(duration)
#        }
#
#        stats_file = os.path.join(Plt_Dir, 'validation_stats.txt')
#        with open(stats_file, 'w') as f:
#            for key, value in stats.items():
#                f.write(f"{key}: {value}\n")





        def save_in_parts(data, base_filename, names_list):
            for fmc_idx in tqdm(range(data.shape[0]), desc=f"Saving {base_filename}"):
                try:
                    fmc_data = data[fmc_idx, :, :]
                    filename = names_list[fmc_idx].replace('.mat', f'_{base_filename}.mat')
                    fmc_filename = os.path.join(matlab_dir, filename)
                    sio.savemat(fmc_filename, {f"{base_filename}": fmc_data})
                except Exception as e:
                    print(f"Erreur lors de la sauvegarde {base_filename} index {fmc_idx}: {e}")
                    continue


        print(f"Evaluation terminee! Resultats sauvegardes dans: {Plt_Dir}")

    except Exception as e:
        print(f"Fatal error in main(): {e}")
        raise

if __name__ == "__main__":
    main()