# -*- coding: utf-8 -*-
"""
MoE evaluation (baseline). Prefer ``evaluate_moe2.py`` for the extended / updated pipeline.
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
from mobilevit_v3_v1_MOE_Pixel2 import MobileViTv3_v1_dynamicFPN_MOE_Pixel2

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
    def __init__(self, directory, device='cuda:0'):
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
    plt.xticks([0, 8])
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

        device = torch.device("cuda:0")
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


        model_type = "MbViT_XXS4_FF8_MF_Normalisation_NW_W" #'UNETred_copper_50_50_BruitDuet_fixe_NW_et_W'


        num_epochs = getattr(config, 'num_epochs', 'unknown')
        Plt_Dir = f'Evaluation_{model_type}_{date_str}_{num_epochs}_epochs'
        os.makedirs(Plt_Dir, exist_ok=True)
        print(f"Dossier de sauvegarde cree: {Plt_Dir}")

        # Repertoire des donnees de validation
        data_dir_valid = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/Matlab_plus/Verif_copper_defect"

        if not os.path.exists(data_dir_valid):
            raise FileNotFoundError(f"Le repertoire de donnees n'existe pas: {data_dir_valid}")

        # Chargement du modele pre-entraine (MoE)
        try:
            # MoE model with same hyperparameters as baseline
            Model = MobileViTv3_v1_dynamicFPN_MOE_Pixel2((4096, 64), 'x_small', 1000, (32,32),noise_std=0.0).to(device)
            #Model = MobileViTv3_v1_dynamicFPN((4096, 64), 'xx_small4', 1000, (32,32)).to(device)
            #Model = UNETred().to(device)
            #Model = torch.compile(Model)
            pretrained_path = "/mnt/STORAGE-1/home/rniddam/UNET_Binary/MobileUnet_V2_modif/Test_End_seed_lr_MbViTPixel2_MOEV2_2E_XXSMALL4_noise00_FF8_Normal_MF_batchsize16_NW_W_BruitFixe_BruitDuet200e-6_20251218_161959_400_epochs_seed_69738009/Model.pth"
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

        # collect per-sample expert stats
        all_expert_data = []  # list of dicts, one entry per sample

        print("Debut de l'evaluation...")
        start_time = datetime.datetime.now()

        with torch.no_grad():
            for batch_idx, (ampV, binV, truncated_names) in enumerate(tqdm(val_loader, desc="Validation")):
                try:
                    ampV = ampV.to(device)
                    binV = binV.to(device)

                    # forward with MoE aux stats
                    reconV, aux_stats = Model(binV)
                    loss_val = losscalc(ampV, reconV)
                    ncc_val = ncc(ampV, reconV)

                    val_losses.append(loss_val.item())
                    val_accuracy.append(ncc_val.item())

                    print(f"Batch {batch_idx + 1}: Loss = {loss_val.item():.6f}, NCC = {ncc_val.item():.6f}")
                    if aux_stats is not None:
                        expert_fractions = [f"{aux_stats['gate_fraction'][i].item():.4f}" for i in range(len(aux_stats['gate_fraction']))]
                        print(f"  Expert usage fractions: {expert_fractions}")

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

                        # collect expert usage per sample
                        if aux_stats is not None:
                            sample_expert_data = {'sample_name': base_name}

                            # debug: print aux_stats keys
                            if batch_idx == 0 and i == 0:
                                print(f"  Debug aux_stats keys: {list(aux_stats.keys())}")
                                if 'expert_indices' in aux_stats:
                                    print(f"  Debug expert_indices shape: {aux_stats['expert_indices'].shape}")
                                if 'gate_fraction' in aux_stats:
                                    print(f"  Debug gate_fraction: {aux_stats['gate_fraction']}")

                            # per-token expert indices if available
                            # expert_indices shape: [B, num_encoder_layers, S, top_k]
                            if 'expert_indices' in aux_stats:
                                expert_indices_batch = aux_stats['expert_indices'].cpu().numpy()
                                # Debug
                                if batch_idx == 0 and i == 0:
                                    print(f"  Debug expert_indices_batch shape: {expert_indices_batch.shape}")
                                    print(f"  Debug expert_indices_batch min/max: {expert_indices_batch.min()}/{expert_indices_batch.max()}")

                                # slice sample i from batch
                                if expert_indices_batch.shape[0] > i:
                                    expert_indices_sample = expert_indices_batch[i]  # [num_encoder_layers, S, top_k]
                                else:
                                    print(f"  Warning: batch index {i} out of range for expert_indices shape {expert_indices_batch.shape}")
                                    expert_indices_sample = None

                                if expert_indices_sample is not None:
                                    num_experts = len(aux_stats['gate_fraction'])
                                    num_layers = expert_indices_sample.shape[0]

                                    # Debug
                                    if batch_idx == 0 and i == 0:
                                        print(f"  Debug expert_indices_sample shape: {expert_indices_sample.shape}")
                                        print(f"  Debug expert_indices_sample min/max: {expert_indices_sample.min()}/{expert_indices_sample.max()}")

                                    # Creer une sequence d'experts par couche (mode le plus frequent par couche)
                                    expert_sequence = []
                                    expert_sequence_per_layer = []

                                    # count expert picks for this sample
                                    expert_usage_sample = np.zeros(num_experts, dtype=np.int32)
                                    total_tokens = 0

                                    for layer_idx in range(num_layers):
                                        layer_indices = expert_indices_sample[layer_idx].flatten()  # [S * top_k]
                                        total_tokens += len(layer_indices)

                                        # count per expert in this layer
                                        expert_counts_layer = np.zeros(num_experts, dtype=np.int32)
                                        for expert_id in range(num_experts):
                                            count = np.sum(layer_indices == expert_id)
                                            expert_counts_layer[expert_id] = count
                                            expert_usage_sample[expert_id] += count

                                        # dominant expert in this layer
                                        if expert_counts_layer.sum() > 0:
                                            most_used_expert = np.argmax(expert_counts_layer)
                                            expert_sequence.append(int(most_used_expert) + 1)  # +1 for 1-based indexing
                                        else:
                                            expert_sequence.append(0)  # Pas d'expert utilise

                                        # full per-layer sequence for analysis
                                        expert_sequence_per_layer.append(layer_indices.astype(np.int32) + 1)  # +1 for 1-based indexing

                                    # Calculer les fractions par echantillon
                                    if total_tokens > 0:
                                        expert_fraction_sample = expert_usage_sample.astype(np.float32) / total_tokens
                                    else:
                                        expert_fraction_sample = np.zeros(num_experts, dtype=np.float32)

                                    # store this sample's tensors
                                    sample_expert_data['expert_sequence'] = np.array(expert_sequence, dtype=np.int32)  # Sequence par couche
                                    sample_expert_data['expert_sequence_per_layer'] = expert_sequence_per_layer  # Sequences completes par couche
                                    sample_expert_data['expert_usage'] = expert_usage_sample.astype(np.int32)
                                    sample_expert_data['expert_fraction'] = expert_fraction_sample.astype(np.float32)
                                    sample_expert_data['total_tokens'] = int(total_tokens)
                                    sample_expert_data['num_layers'] = int(num_layers)
                                    sample_expert_data['num_experts'] = int(num_experts)

                                    # print sample summary
                                    if len(expert_sequence) > 0:
                                        expert_seq_str = ','.join(map(str, expert_sequence))
                                        print(f"  Echantillon {base_name}: Expert sequence = [{expert_seq_str}]")
                                else:
                                    print(f"  Warning: expert_indices_sample is None for sample {base_name}")
                                    # Stocker des valeurs par defaut
                                    num_experts = len(aux_stats['gate_fraction']) if 'gate_fraction' in aux_stats else 3
                                    sample_expert_data['expert_sequence'] = np.array([], dtype=np.int32)
                                    sample_expert_data['expert_usage'] = np.zeros(num_experts, dtype=np.int32)
                                    sample_expert_data['expert_fraction'] = np.zeros(num_experts, dtype=np.float32)
                                    sample_expert_data['num_experts'] = int(num_experts)

                            # Ajouter a la liste globale
                            all_expert_data.append(sample_expert_data)

                    # Lib�ration m�moire
                    del ampV, binV, reconV
                    if aux_stats is not None:
                        del aux_stats
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Erreur lors du traitement du batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
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

        # Directory for MATLAB exports (si pas deja cree)
        matlab_dir = os.path.join(Plt_Dir, f'Matlab_{model_type}')
        os.makedirs(matlab_dir, exist_ok=True)

        # save all expert stats to one .mat
        if all_expert_data:
            print("\nSaving expert usage statistics...")

            # package arrays for MATLAB
            num_samples = len(all_expert_data)
            sample_names = [data['sample_name'] for data in all_expert_data]

            # Determiner le nombre d'experts et de couches
            num_experts = all_expert_data[0].get('num_experts', 3)
            num_layers = all_expert_data[0].get('num_layers', 0)

            # usage count / fraction matrices
            expert_usage_matrix = np.zeros((num_samples, num_experts), dtype=np.int32)
            expert_fraction_matrix = np.zeros((num_samples, num_experts), dtype=np.float32)

            # cell array: one expert sequence per sample
            expert_sequences = []
            expert_sequences_detailed = []  # Sequences completes par couche

            for idx, data in enumerate(all_expert_data):
                if 'expert_usage' in data:
                    expert_usage_matrix[idx, :] = data['expert_usage']
                if 'expert_fraction' in data:
                    expert_fraction_matrix[idx, :] = data['expert_fraction']
                if 'expert_sequence' in data:
                    expert_sequences.append(data['expert_sequence'])
                else:
                    expert_sequences.append(np.array([], dtype=np.int32))

                if 'expert_sequence_per_layer' in data:
                    expert_sequences_detailed.append(data['expert_sequence_per_layer'])
                else:
                    expert_sequences_detailed.append([])

            # MATLAB cell arrays for names and sequences
            # MATLAB cells as numpy object arrays
            sample_names_cell = np.empty((num_samples,), dtype=object)
            for idx, name in enumerate(sample_names):
                sample_names_cell[idx] = name

            # cell array: one expert sequence per sample
            expert_sequences_cell = np.empty((num_samples,), dtype=object)
            for idx, seq in enumerate(expert_sequences):
                expert_sequences_cell[idx] = seq

            # dict for savemat
            expert_summary = {
                'sample_names': sample_names_cell,  # Cell array de noms
                'expert_sequences': expert_sequences_cell,  # Cell array de sequences (une par echantillon)
                'expert_usage': expert_usage_matrix,  # [num_samples, num_experts] - nombre d'utilisations
                'expert_fraction': expert_fraction_matrix,  # [num_samples, num_experts] - fractions d'utilisation
                'num_samples': np.array([num_samples], dtype=np.int32),
                'num_experts': np.array([num_experts], dtype=np.int32),
                'num_layers': np.array([num_layers], dtype=np.int32)
            }

            # Ajouter les sequences detaillees si disponibles (optionnel, peut etre volumineux)
            if expert_sequences_detailed and len(expert_sequences_detailed[0]) > 0:
                expert_sequences_detailed_cell = np.empty((num_samples,), dtype=object)
                for idx, seq_list in enumerate(expert_sequences_detailed):
                    # nested cell: sample -> layer
                    layer_cell = np.empty((len(seq_list),), dtype=object)
                    for layer_idx, layer_seq in enumerate(seq_list):
                        layer_cell[layer_idx] = layer_seq
                    expert_sequences_detailed_cell[idx] = layer_cell
                expert_summary['expert_sequences_detailed'] = expert_sequences_detailed_cell

            # write .mat
            expert_summary_file = os.path.join(matlab_dir, 'ExpertUsage_Summary.mat')
            sio.savemat(expert_summary_file, expert_summary, oned_as='column')
            print(f"Statistiques d'experts sauvegardees dans: {expert_summary_file}")
            print(f"  - {num_samples} echantillons analyses")
            print(f"  - {num_experts} experts")
            print(f"  - {num_layers} couches par echantillon")

            # Afficher un resume des sequences
            print("\nResume des sequences d'experts:")
            for idx, data in enumerate(all_expert_data[:10]):  # Afficher les 10 premiers
                if 'expert_sequence' in data:
                    seq_str = ','.join(map(str, data['expert_sequence']))
                    print(f"  {data['sample_name']}: [{seq_str}]")
            if num_samples > 10:
                print(f"  ... et {num_samples - 10} autres echantillons")

        print(f"\nEvaluation terminee! Resultats sauvegardes dans: {Plt_Dir}")

    except Exception as e:
        print(f"Fatal error in main(): {e}")
        raise

if __name__ == "__main__":
    main()