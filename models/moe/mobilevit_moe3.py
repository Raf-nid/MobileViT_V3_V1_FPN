import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple, Dict, List


def conv_2d(inp, oup, kernel_size=3, stride=1, padding=0, groups=1, bias=False, norm=True, act=True):
    """Convolution 2D avec BatchNorm et activation optionnelles"""
    conv = nn.Sequential()
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.SiLU())
    return conv


class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual Block"""
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0))
        self.block.add_module('conv_3x3', conv_2d(hidden_dim, hidden_dim, kernel_size=3, 
                                                   stride=stride, padding=1, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, 
                                                  stride=1, padding=0, act=False))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class Attention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, embed_dim, heads=4, dim_head=8, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, S, C = x.shape
        
        # [B, S, C] -> [B, S, 3*C] -> [B, S, 3, num_heads, dim_head]
        qkv = self.qkv_proj(x).reshape(B, S, 3, self.num_heads, -1)
        # [B, S, 3, num_heads, dim_head] -> [B, num_heads, 3, S, dim_head]
        qkv = qkv.permute(0, 3, 2, 1, 4)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Scaled dot-product attention
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))  # [B, num_heads, S, S]
        attn = F.softmax(attn.float(), dim=-1).to(attn.dtype)
        attn = self.attn_dropout(attn)
        
        # Weighted sum
        out = torch.matmul(attn, v)  # [B, num_heads, S, dim_head]
        out = out.transpose(1, 2).reshape(B, S, C)  # [B, S, C]
        out = self.out_proj(out)
        
        return out


class MoEFeedForward(nn.Module):
    """
    Mixture of Experts Feed-Forward Layer
    Improvements: efficient routing, correct normalization, load balancing
    """
    def __init__(self, dim, hidden_dim, num_experts=3, top_k=1, dropout=0.0, noise_std=0.1):
        super().__init__()
        assert 1 <= top_k <= num_experts, f"top_k must be in [1, {num_experts}]"
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.eps = 1e-9
        
        # Experts : chaque expert est un petit MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(dim, num_experts, bias=True)
        
        # Xavier initialization for the gate (stability)
        nn.init.xavier_uniform_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, S, C] input tensor
        Returns:
            out: [B, S, C] output tensor
            aux: dict with auxiliary statistics for loss computation
        """
        B, S, C = x.shape
        
        # 1. Compute gating logits
        gate_logits = self.gate(x)  # [B, S, num_experts]
        
        # 2. Add noise during training for exploration (Gshard-style)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        
        # 3. Softmax to get routing probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)  # [B, S, num_experts]
        
        # 4. Top-k routing
        topk_vals, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        # [B, S, top_k], [B, S, top_k]
        
        # 5. Renormalize weights of selected experts (important)
        topk_vals_normalized = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + self.eps)
        
        # 6. Efficient top-k routing
        if self.top_k == 1:
            out = self._route_topk1_efficient(x, topk_idx.squeeze(-1), 
                                                topk_vals_normalized.squeeze(-1))
        else:
            out = self._route_topk_general(x, topk_idx, topk_vals_normalized)
        
        # 7. Compute auxiliary statistics for load balancing loss
        aux = self._compute_aux_stats(gate_probs, topk_idx, B, S)
        
        return out, aux
    
    def _route_topk1_efficient(self, x: torch.Tensor, expert_idx: torch.Tensor, 
                                weights: torch.Tensor) -> torch.Tensor:
        """
        Routing efficace pour top_k=1
        Computes only the experts that are needed
        """
        B, S, C = x.shape
        output = torch.zeros_like(x)
        
        # For each expert, process all routed tokens in batch
        for expert_id in range(self.num_experts):
            # mask of tokens routed to this expert
            expert_mask = (expert_idx == expert_id)
            
            if expert_mask.any():
                # Indices des tokens pour cet expert
                batch_idx, seq_idx = expert_mask.nonzero(as_tuple=True)
                
                # Extraire les tokens
                expert_input = x[batch_idx, seq_idx]  # [num_tokens, C]
                
                # forward through this expert
                expert_output = self.experts[expert_id](expert_input)  # [num_tokens, C]
                
                # Appliquer les poids de gating
                expert_weights = weights[batch_idx, seq_idx].unsqueeze(-1)  # [num_tokens, 1]
                weighted_output = expert_output * expert_weights
                
                # Write back into the output tensor
                output[batch_idx, seq_idx] = weighted_output
        
        return output
    
    def _route_topk_general(self, x: torch.Tensor, topk_idx: torch.Tensor, 
                            topk_vals: torch.Tensor) -> torch.Tensor:
        """
        General routing when top_k > 1
        Calcule tous les experts (moins efficace mais correct)
        """
        B, S, C = x.shape
        
        # build a sparse mask
        mask = torch.zeros(B, S, self.num_experts, device=x.device)
        mask.scatter_(-1, topk_idx, 1.0)
        
        # weighted mask from normalized values
        weighted_mask = torch.zeros(B, S, self.num_experts, device=x.device)
        weighted_mask.scatter_(-1, topk_idx, topk_vals)
        
        # Calculer les sorties de tous les experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        # [B, S, num_experts, C]
        
        # weighted combination
        weighted_mask = weighted_mask.unsqueeze(-1)  # [B, S, num_experts, 1]
        output = (expert_outputs * weighted_mask).sum(dim=2)  # [B, S, C]
        
        return output
    
    def _compute_aux_stats(self, gate_probs: torch.Tensor, topk_idx: torch.Tensor, 
                           B: int, S: int) -> Dict[str, torch.Tensor]:
        """
        Calculer les statistiques auxiliaires pour les losses
        """
        # mean routing probabilities per expert
        gate_probs_mean = gate_probs.mean(dim=[0, 1])  # [num_experts]
        
        # Comptage des tokens par expert
        if self.top_k == 1:
            expert_idx = topk_idx.squeeze(-1).flatten()
            gate_usage = torch.bincount(
                expert_idx,
                minlength=self.num_experts
            ).float()
        else:
            mask = torch.zeros(B, S, self.num_experts, device=gate_probs.device)
            mask.scatter_(-1, topk_idx, 1.0)
            gate_usage = mask.sum(dim=[0, 1])
        
        # Fraction de tokens par expert
        gate_fraction = gate_usage / (B * S + self.eps)
        
        # Load balancing loss (Switch Transformer style)
        # Encourage uniform distribution: sum(f_i * P_i) where f_i is fraction, P_i is mean prob
        load_loss = self.num_experts * torch.sum(gate_fraction * gate_probs_mean)
        
        return {
            'gate_probs_mean': gate_probs_mean,
            'gate_usage': gate_usage,
            'gate_fraction': gate_fraction,
            'load_loss': load_loss
        }


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, heads=8, dim_head=8, 
                 dropout=0.0, attn_dropout=0.0, use_moe=False, 
                 num_experts=3, top_k=1, noise_std=0.0):
        super().__init__()
        self.use_moe = use_moe
        
        # keep the same structure as the non-MoE version
        self.liteTransformer = nn.Sequential(
            Attention(embed_dim, heads, dim_head, attn_dropout),
            nn.Dropout(dropout)
        )
        
        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True),
            self.liteTransformer
        )
        
        self.pre_norm_ffn_norm = nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True)
        
        if not use_moe:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_latent_dim, bias=True),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_latent_dim, embed_dim, bias=True),
                nn.Dropout(dropout)
            )
        else:
            self.ffn = MoEFeedForward(
                embed_dim, ffn_latent_dim, 
                num_experts=num_experts, 
                top_k=top_k, 
                dropout=dropout,
                noise_std=noise_std
            )
    
    def forward(self, x):
        x = x + self.pre_norm_mha(x)
        
        if not self.use_moe:
            x = x + self.ffn(self.pre_norm_ffn_norm(x))
            return x, None
        else:
            ffn_out, aux = self.ffn(self.pre_norm_ffn_norm(x))
            x = x + ffn_out
            return x, aux


class MobileViTBlockV3_v1(nn.Module):
    """MobileViT Block avec support MoE"""
    def __init__(self, inp, attn_dim, ffn_multiplier, heads, dim_head, 
                 attn_blocks, patch_size, use_moe=False, num_experts=3, 
                 top_k=1, noise_std=0.0):
        super(MobileViTBlockV3_v1, self).__init__()
        self.patch_h, self.patch_w = patch_size
        self.patch_area = int(self.patch_h * self.patch_w)
        
        # Local representation (convolutions)
        self.local_rep = nn.Sequential(
            conv_2d(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp),
            conv_2d(inp, attn_dim, kernel_size=1, stride=1, norm=False, act=False)
        )
        
        # Global representation (transformers)
        self.global_rep = nn.ModuleList()
        ffn_dims = [int((ffn_multiplier * attn_dim) // 16 * 16)] * attn_blocks
        
        for i in range(attn_blocks):

            encoder = TransformerEncoder(
                attn_dim, ffn_dims[i], 
                heads=heads, 
                dim_head=dim_head,
                dropout=0.0, 
                attn_dropout=0.0,
                use_moe=use_moe, 
                num_experts=num_experts,
                top_k=top_k,
                noise_std=noise_std
            )
            self.global_rep.append(encoder)
        
        self.layernorm_after = nn.LayerNorm(attn_dim, eps=1e-5)
        
        # Projection and fusion
        self.conv_proj = conv_2d(attn_dim, inp, kernel_size=1, stride=1)
        self.fusion = conv_2d(inp + attn_dim, inp, kernel_size=1, stride=1)

    def unfolding(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Unfold feature map into patches"""
        B, C, H, W = feature_map.shape
        
        # Pad to multiple of patch size
        new_h = int(math.ceil(H / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(W / self.patch_w) * self.patch_w)
        
        interpolate = False
        if new_w != W or new_h != H:
            feature_map = F.interpolate(
                feature_map, size=(new_h, new_w), 
                mode="bilinear", align_corners=False
            )
            interpolate = True
        
        # Reshape into patches
        num_patch_h = new_h // self.patch_h
        num_patch_w = new_w // self.patch_w
        num_patches = num_patch_h * num_patch_w
        
        # [B, C, H, W] -> [B*patch_area, num_patches, C]
        patches = feature_map.reshape(
            B, C, num_patch_h, self.patch_h, num_patch_w, self.patch_w
        )
        patches = patches.permute(0, 3, 5, 1, 2, 4).contiguous()
        patches = patches.reshape(B * self.patch_area, num_patches, C)
        
        info_dict = {
            "orig_size": (H, W),
            "batch_size": B,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }
        
        return patches, info_dict

    def folding(self, patches: torch.Tensor, info_dict: Dict) -> torch.Tensor:
        """Fold patches back into feature map"""
        B = info_dict["batch_size"]
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]
        
        # [B*patch_area, num_patches, C] -> [B, C, H, W]
        patches = patches.reshape(
            B, self.patch_h, self.patch_w, -1, num_patch_h, num_patch_w
        )
        feature_map = patches.permute(0, 3, 4, 1, 5, 2).contiguous()
        feature_map = feature_map.reshape(
            B, -1, num_patch_h * self.patch_h, num_patch_w * self.patch_w
        )
        
        # Interpolate back to original size if needed
        if info_dict["interpolate"]:
            feature_map = F.interpolate(
                feature_map, size=info_dict["orig_size"],
                mode="bilinear", align_corners=False
            )
        
        return feature_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        res = x.clone()
        
        # Local representation
        fm_conv = self.local_rep(x)
        
        # Unfold into patches
        patches, info_dict = self.unfolding(fm_conv)
        
        # Global representation (transformers)
        aux_list = []
        for encoder in self.global_rep:
            patches, aux = encoder(patches)
            if aux is not None:
                aux_list.append(aux)
        
        # Aggregate aux stats
        #aux_stats = aux_list[-1] if aux_list else None
        if aux_list:
            aux_stats = {
                'gate_probs_mean': torch.stack([a['gate_probs_mean'] for a in aux_list]).mean(0),
                'gate_usage': torch.stack([a['gate_usage'] for a in aux_list]).sum(0),
                'gate_fraction': torch.stack([a['gate_fraction'] for a in aux_list]).mean(0),
                'load_loss': sum(a['load_loss'] for a in aux_list) / len(aux_list)
            }
        else:
            aux_stats = None
        # Layer norm and fold back
        patches = self.layernorm_after(patches)
        x = self.folding(patches, info_dict)
        
        # Projection and fusion
        x = self.conv_proj(x)
        x = self.fusion(torch.cat([fm_conv, x], dim=1))
        x = x + res
        
        return x, aux_stats


class MobileViTv3_v1_dynamicFPN_MOE3_Pixel2(nn.Module):
    """
    MobileViT v3 avec FPN dynamique et Mixture of Experts
    MoE placed only in layer_5 for efficiency
    """
    def __init__(self, image_size: Tuple[int, int], mode: str, 
                 num_classes: int = 1000, patch_size: Tuple[int, int] = (64, 64),
                 num_experts: int = 3, top_k: int = 1, noise_std: float = 0.0):
        super().__init__()
        
        ih, iw = image_size
        self.ph, self.pw = patch_size
        assert ih % self.ph == 0 and iw % self.pw == 0, "Image size must be divisible by patch size"
        assert mode in ['xx_small', 'xx_small4', 'x_small', 'small'], f"Invalid mode: {mode}"
        
        # Configuration selon le mode
        configs = {
            'xx_small': {
                'mv2_exp_mult': 2,
                'ffn_multiplier': 2,
                'channels': [16, 16, 24, 64, 80, 128],
                'attn_dim': [64, 80, 96]
            },
            'xx_small4': {
                'mv2_exp_mult': 4,
                'ffn_multiplier': 2,
                'channels': [16, 16, 24, 64, 80, 128],
                'attn_dim': [64, 80, 96]
            },
            'x_small': {
                'mv2_exp_mult': 4,
                'ffn_multiplier': 2,
                'channels': [16, 32, 48, 96, 160, 160],
                'attn_dim': [96, 120, 144]
            },
            'small': {
                'mv2_exp_mult': 4,
                'ffn_multiplier': 2,
                'channels': [16, 32, 64, 128, 256, 320],
                'attn_dim': [144, 192, 240]
            }
        }
        
        cfg = configs[mode]
        channels = cfg['channels']
        attn_dim = cfg['attn_dim']
        mv2_exp_mult = cfg['mv2_exp_mult']
        ffn_multiplier = cfg['ffn_multiplier']
        
        # Stem
        self.conv_0 = conv_2d(1, channels[0], kernel_size=3, stride=2, padding=1)
        
        # Stages
        self.layer_1 = InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=mv2_exp_mult)
        
        self.layer_2 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=mv2_exp_mult),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=mv2_exp_mult),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=mv2_exp_mult)
        )
        
        self.layer_3 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockV3_v1(
                channels[3], attn_dim[0], ffn_multiplier, 
                heads=4, dim_head=8, attn_blocks=2, 
                patch_size=patch_size, use_moe=True
            )
        )
        
        self.layer_4 = nn.Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockV3_v1(
                channels[4], attn_dim[1], ffn_multiplier,
                heads=4, dim_head=8, attn_blocks=4,
                patch_size=patch_size, use_moe=False
            )
        )
        
        # Layer 5 MoE
        self.layer_5 = nn.Sequential(
            InvertedResidual(channels[4], channels[5], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockV3_v1(
                channels[5], attn_dim[2], ffn_multiplier,
                heads=4, dim_head=8, attn_blocks=3,
                patch_size=patch_size, use_moe=False,
                num_experts=num_experts, top_k=top_k,
                noise_std=noise_std
            )
        )
        
        # FPN (Feature Pyramid Network)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.top_layer = nn.Conv2d(channels[-1], 256, kernel_size=1, stride=1, padding=0)
        
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(channels[2], 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(channels[3], 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(channels[4], 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(channels[5], 256, kernel_size=1, stride=1, padding=0)
        ])
        
        self.smooth_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) 
            for _ in range(5)
        ])
        
        # Segmentation head
        self.pixel_shuffle_conv = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=4)
        self.last_conv = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)  # collapse to 1 output channel
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialisation des poids"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Upsample x and add to y"""
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass
        Returns:
            output: [B, 1, H, W] segmentation map
            aux_stats: dict with MoE statistics (or None)
        """
        # Stem
        x_0 = x
        x = self.conv_0(x)
        
        # Backbone stages
        lat_1 = self.layer_1(x)
        lat_2= self.layer_2(lat_1)
        lat_3, aux_stats = self.layer_3(lat_2)
        lat_4, _ = self.layer_4(lat_3)
        lat_5,  _= self.layer_5(lat_4)  # MoE dans layer_5
        
        # FPN
        x = self.global_pool(lat_5)
        x = self.top_layer(x)
        
        lateral_features = [
            self.lateral_layers[0](lat_2),
            self.lateral_layers[1](lat_3),
            self.lateral_layers[2](lat_4),
            self.lateral_layers[3](lat_5)
        ]
        lateral_features.reverse()
        
        fpn_features = [x]
        for lat in lateral_features:
            fpn_features.append(self._upsample_add(fpn_features[-1], lat))
        
        smooth_features = [
            smooth(fpn) for smooth, fpn in zip(self.smooth_layers, fpn_features)
        ]
        
        # Segmentation head
        out = self.pixel_shuffle_conv(smooth_features[-1])  # 256 -> 16 channels
        out = self.pixel_shuffle(out)  # H/4 x W/4 -> H x W avec 1 channel
        out = torch.cat((out, x_0), dim=1)  # expand to 2 channels when needed
        out = self.last_conv(out)  # project back to 1 channel
        out = 2*out-1
        return out, aux_stats


class MoELoss2(nn.Module):
    """
    Combined loss for MoE:
    - MSE (tâche principale)
    - Load balancing loss (utilisation uniforme des experts)
    - Entropy loss optionnelle (exploration du gating)
    """
    def __init__(self, lambda_balance: float = 0.00001, lambda_entropy: float = 0.000001):
        super().__init__()
        self.lambda_balance = lambda_balance
        self.lambda_entropy = lambda_entropy
        self.mse_loss = nn.MSELoss()
        self.eps = 1e-9

    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                aux_stats: Optional[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: model predictions [B, 1, H, W]
            target: ground truth [B, 1, H, W]
            aux_stats: dict with MoE statistics or None
        Returns:
            total_loss: combined loss
            stats: dict with individual loss components
        """
        # Loss principale (MSE)
        mse = self.mse_loss(pred, target)
        
        # Si pas de MoE (early layers ou pas d'experts)
        if aux_stats is None or 'load_loss' not in aux_stats:
            return mse, {
                'mse': mse.item(),
                'balance': 0.0,
                'entropy': 0.0,
                'total': mse.item()
            }
        
        # load balancing loss (computed in MoEFeedForward)
        balance_loss = aux_stats['load_loss']
        
        # Entropy loss optionnelle (encourage l'exploration)
        if self.lambda_entropy > 0:
            probs_mean = aux_stats['gate_probs_mean']
            entropy_loss = -torch.sum(probs_mean * torch.log(probs_mean + self.eps))
        else:
            entropy_loss = torch.tensor(0.0, device=mse.device)
        
        # total combined loss
        total_loss = (
            mse + 
            self.lambda_balance * balance_loss + 
            self.lambda_entropy * entropy_loss
        )
        
        # Statistiques pour le logging
        stats = {
            'mse': mse.item(),
            'balance': balance_loss.item() if isinstance(balance_loss, torch.Tensor) else balance_loss,
            'entropy': entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss,
            'total': total_loss.item()
        }
        
        # Ajouter les fractions d'utilisation des experts si disponibles
        if 'gate_fraction' in aux_stats:
            for i, frac in enumerate(aux_stats['gate_fraction']):
                stats[f'expert_{i}_usage'] = frac.item()
        
        return total_loss, stats, mse




# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    # dummy batch
    input_tensor = torch.randn(1, 1, 4096, 1024)  # input
    target_tensor = torch.randn(1, 1, 4096, 1024)  # cible

    # model
    model = MobileViTv3_v1_dynamicFPN_MOE3_Pixel2(
        (4096, 1024), 'xx_small4', num_classes=1000, patch_size=(32, 32)
    )

    # model output
    out, aux_stats = model(input_tensor)
    print("out.shape:", out.shape)

    # Instancie la loss
    loss_fn = MoELoss()

    # Calcule la loss
    total_loss, stats, mse = loss_fn(out, target_tensor, aux_stats)

    # print results
    print("Total loss:", total_loss.item())
    print("MSE:", stats['mse'])
    print("Balance loss:", stats['balance'])
    print("Entropy loss:", stats['entropy'])
    print("Expert usages:", {k: v for k, v in stats.items() if 'expert' in k})

