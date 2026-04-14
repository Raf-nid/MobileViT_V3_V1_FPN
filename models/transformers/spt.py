import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import math

class ShiftedPatchTokenization(nn.Module):
    def __init__(self, in_dim, dim, merging_size=2, exist_class_t=False, shift_mode="3"):
        print(shift_mode)
        """
        Paramètres :
         - in_dim : dimension d'entrée (nombre de canaux)
         - dim : dimension de sortie après fusion des patchs
         - merging_size : taille de regroupement (patch)
         - exist_class_t : présence d'un token de classe
         - shift_mode : mode de décalage à appliquer, valeurs possibles : "4_cardinal", "4_diagonal", "8"
        """
        super().__init__()
        self.exist_class_t = exist_class_t
        self.merging_size = merging_size
        self.patch_shifting = PatchShifting(merging_size, mode=shift_mode)
        
        # Calcul de la dimension du patch :
        # Dans le cas "4_cardinal" ou "4_diagonal", on concatène le patch original + 4 patchs décalés
        # Dans le cas "8", on concatène le patch original + 8 patchs décalés
        if shift_mode == "2":
            patch_multiplier = 9
        else:
            patch_multiplier = 5
        
        patch_dim = in_dim * patch_multiplier * (merging_size ** 2)
        
        if exist_class_t:
            self.class_linear = nn.Linear(in_dim, dim)
        
        self.merging = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=merging_size, p2=merging_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )
    
    def forward(self, x):
        # x est supposé être une carte de caractéristiques de forme [B, C, H, W]
        if self.exist_class_t:
            visual_tokens = x
            out_visual = self.patch_shifting(visual_tokens)
            out_visual = self.merging(out_visual)
            return out_visual
        else:
            x_shifted = self.patch_shifting(x)
            out = self.merging(x_shifted)  # forme : [B, num_patches, token_dim]
            
            # Pour simuler l'unfolding original, répéter chaque exemple du batch autant de fois que nécessaire.
            # Ici, pour "4_cardinal" ou "4_diagonal", patch_area = merging_size**2; pour "8", le principe reste inchangé.
            patch_area = self.merging_size * self.merging_size
            out = out.repeat_interleave(patch_area, dim=0)  # Nouvelle forme : [B*patch_area, num_patches, token_dim]
            return out

class PatchShifting(nn.Module):
    def __init__(self, patch_size, mode="3"):
        """
        patch_size : taille d'un patch (entier)
        mode : "4 cardinal directions", "4 diagonal directions" ou "8 cardinal directions"
        """
        super().__init__()
        self.shift = int(patch_size * 0.5)
        self.mode = mode
    
    def forward(self, x):
        # x de forme [B, C, H, W]
        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        
        if self.mode == "1": #
            # Extraction des décalages sur les 4 directions cardinales
            x_left   = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
            x_right  = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
            x_top    = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
            x_bottom = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
            # Concaténer l'original + 4 shift
            x_cat = torch.cat([x, x_left, x_right, x_top, x_bottom], dim=1)
        
        elif self.mode == "2":
            # Extraction des 4 directions cardinales
            x_left   = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
            x_right  = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
            x_top    = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
            x_bottom = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
            # Extraction des 4 directions diagonales
            x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
            x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
            x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
            x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
            # Concaténer l'original + 4 cardinales + 4 diagonales
            x_cat = torch.cat([x, x_left, x_right, x_top, x_bottom, x_lu, x_ru, x_lb, x_rb], dim=1)
        
        else:  # Par défaut, mode "4_diagonal"
            x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
            x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
            x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
            x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
            x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1)
        
        return x_cat
