import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def conv_2d(inp, oup, kernel_size=3, stride=1, padding=0, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.SiLU())
    return conv


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0))
        self.block.add_module('conv_3x3', conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, act=False))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)  


class Attention(nn.Module):
    def __init__(self, embed_dim, num_patches, heads=4, dim_head=8, attn_dropout=0):
        super().__init__()
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim
        self.num_heads = heads
        init_scale = dim_head ** -0.5
        self.scale = nn.Parameter(init_scale * torch.ones(heads))
        
        self.num_patches = num_patches
        # Correction: ne pas ajouter +1
        #mask = torch.eye(self.num_patches, self.num_patches)
        #print(f"mask.shape: {mask.shape}")
        #self.mask = torch.nonzero(mask == 1, as_tuple=False)

    def forward(self, x):
        b_sz, S_len, in_channels = x.shape
        qkv = self.qkv_proj(x).reshape(b_sz, S_len, 3, self.num_heads, -1)
        qkv = qkv.transpose(1, 3).contiguous()  # [B, heads, 3, S, c]
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # On ajuste self.scale pour la diffusion
        q = q * self.scale.view(1, self.num_heads, 1, 1)
        k = k.transpose(-1, -2)
        attn = torch.matmul(q, k)

        # Application du masque sur l'axe 2 et 3
        patch = attn.shape[2]
        mask = torch.eye(patch, patch)
        mask = torch.nonzero(mask == 1, as_tuple=False)

        attn[:, :, mask[:, 0], mask[:, 1]] = -987654321

        attn_dtype = attn.dtype
        attn = self.softmax(attn.float()).to(attn_dtype)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)
        return out



class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, num_patches, heads=8, dim_head=8, dropout=0, attn_dropout=0):
        super().__init__()
        # pass arguments to Attention in the expected order
        self.liteTransformer = nn.Sequential(
            Attention(embed_dim, num_patches, heads, dim_head, attn_dropout),
            nn.Dropout(dropout)
        )

        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True),
            self.liteTransformer
        )
        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True),
            nn.Linear(embed_dim, ffn_latent_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_latent_dim, embed_dim, bias=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.pre_norm_mha(x)  # Self-attention
        x = x + self.pre_norm_ffn(x)   # Feed-Forward Network
        return x


class MobileViTBlockV3_v1(nn.Module):
    def __init__(self, inp, attn_dim, ffn_multiplier, heads, dim_head, attn_blocks, patch_size, image_size=(4096, 1024)):
        super(MobileViTBlockV3_v1, self).__init__()
        self.patch_h, self.patch_w = patch_size
        self.patch_area = self.patch_h * self.patch_w
        self.image_height, self.image_width = image_size
        # Correction du calcul du nombre de patches :
        self.num_patches = (self.image_height // self.patch_h) * (self.image_width // self.patch_w)
        print(f"num_patches: {self.num_patches}")

        # local representation
        self.local_rep = nn.Sequential(
            conv_2d(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp),
            conv_2d(inp, attn_dim, kernel_size=1, stride=1, norm=False, act=False)
        )
        
        # global representation: stacked TransformerEncoder blocks
        self.global_rep = nn.Sequential()
        ffn_dims = [int((ffn_multiplier * attn_dim) // 16 * 16)] * attn_blocks
        for i in range(attn_blocks):
            ffn_dim = ffn_dims[i]
            self.global_rep.add_module(
                f'TransformerEncoder_{i}',
                TransformerEncoder(attn_dim, ffn_dim, self.num_patches, heads, dim_head)
            )
        self.global_rep.add_module('LayerNorm', nn.LayerNorm(attn_dim, eps=1e-5, elementwise_affine=True))

        self.conv_proj = conv_2d(attn_dim, inp, kernel_size=1, stride=1)
        self.fusion = conv_2d(inp + attn_dim, inp, kernel_size=1, stride=1)

    def unfolding(self, feature_map):
        patch_w, patch_h = self.patch_w, self.patch_h
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / patch_h) * patch_h)
        new_w = int(math.ceil(orig_w / patch_w) * patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        num_patch_w = new_w // patch_w
        num_patch_h = new_h // patch_h
        num_patches = num_patch_h * num_patch_w

        # Reshape et permutation pour extraire les patches
        reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        transposed_fm = reshaped_fm.transpose(1, 2)
        reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, self.patch_area)
        transposed_fm = reshaped_fm.transpose(1, 3)
        patches = transposed_fm.reshape(batch_size * self.patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return patches, info_dict

    def folding(self, patches, info_dict):
        assert patches.dim() == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        patches = patches.contiguous().view(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        patches = patches.transpose(1, 3)
        feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        feature_map = feature_map.transpose(1, 2)
        feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return feature_map

    def forward(self, x):
        res = x.clone()
        fm_conv = self.local_rep(x)
        x, info_dict = self.unfolding(fm_conv)
        x = self.global_rep(x)
        x = self.folding(x, info_dict)
        x = self.conv_proj(x)
        x = self.fusion(torch.cat((fm_conv, x), dim=1))
        x = x + res
        return x


class MobileViTv3_v1(nn.Module):
    def __init__(self, image_size, mode, num_classes, patch_size=(32, 32)):  
        """
        MobileViTv3 implementation based on v1
        """
        super().__init__()
        ih, iw = image_size
        self.ph, self.pw = patch_size
        assert ih % self.ph == 0 and iw % self.pw == 0 
        assert mode in ['xx_small', 'x_small', 'small']

        if mode == 'xx_small':
            mv2_exp_mult = 2
            ffn_multiplier = 2
            last_layer_exp_factor = 4
            channels = [16, 16, 24, 64, 80, 128]
            attn_dim = [64, 80, 96]
        elif mode == 'x_small':
            mv2_exp_mult = 4
            ffn_multiplier = 2
            last_layer_exp_factor = 4
            channels = [16, 32, 48, 96, 160, 160]
            attn_dim = [96, 120, 144]
        elif mode == 'small':
            mv2_exp_mult = 4
            ffn_multiplier = 2
            last_layer_exp_factor = 3
            channels = [16, 32, 64, 128, 256, 320]
            attn_dim = [144, 192, 240]
        else:
            raise NotImplementedError

        self.conv_0 = conv_2d(1, channels[0], kernel_size=3, stride=2)

        self.layer_1 = nn.Sequential(
            InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=mv2_exp_mult)
        )
        self.layer_2 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=mv2_exp_mult),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=mv2_exp_mult),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=mv2_exp_mult)
        )
        self.layer_3 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockV3_v1(channels[3], attn_dim[0], ffn_multiplier, heads=4, dim_head=8, attn_blocks=2, patch_size=patch_size)
        )
        self.layer_4 = nn.Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockV3_v1(channels[4], attn_dim[1], ffn_multiplier, heads=4, dim_head=8, attn_blocks=4, patch_size=patch_size)
        )
        self.layer_5 = nn.Sequential(
            InvertedResidual(channels[4], channels[5], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockV3_v1(channels[5], attn_dim[2], ffn_multiplier, heads=4, dim_head=8, attn_blocks=3, patch_size=patch_size)
        )
        self.conv_1x1_exp = conv_2d(channels[-1], channels[-1]*last_layer_exp_factor, kernel_size=1, stride=1)
        self.out = nn.Linear(channels[-1]*last_layer_exp_factor, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x) 
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        x = torch.mean(x, dim=[-2, -1])
        x = self.out(x)
        return x
    

class MobileViTv3_v1_dynamicFPN_LSA(nn.Module):
    def __init__(self, image_size, mode, num_classes, patch_size=(32, 32)):  
        """
        MobileViTv3 with dynamic FPN
        """
        super().__init__()
        ih, iw = image_size
        self.ph, self.pw = patch_size
        assert ih % self.ph == 0 and iw % self.pw == 0 
        assert mode in ['xx_small', 'x_small', 'small']

        if mode == 'xx_small':
            mv2_exp_mult = 2
            ffn_multiplier = 2
            last_layer_exp_factor = 4
            channels = [16, 16, 24, 64, 80, 128]
            attn_dim = [64, 80, 96]
        elif mode == 'x_small':
            mv2_exp_mult = 4
            ffn_multiplier = 2
            last_layer_exp_factor = 4
            channels = [16, 32, 48, 96, 160, 160]
            attn_dim = [96, 120, 144]
        elif mode == 'small':
            mv2_exp_mult = 4
            ffn_multiplier = 2
            last_layer_exp_factor = 3
            channels = [16, 32, 64, 128, 256, 320]
            attn_dim = [144, 192, 240]
        else:
            raise NotImplementedError

        self.conv_0 = conv_2d(1, channels[0], kernel_size=3, stride=2)

        self.layer_1 = nn.Sequential(
            InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=mv2_exp_mult)
        )
        self.layer_2 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=mv2_exp_mult),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=mv2_exp_mult),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=mv2_exp_mult)
        )
        self.layer_3 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockV3_v1(channels[3], attn_dim[0], ffn_multiplier, heads=4, dim_head=8, attn_blocks=2, patch_size=patch_size, image_size=image_size)
        )
        self.layer_4 = nn.Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockV3_v1(channels[4], attn_dim[1], ffn_multiplier, heads=4, dim_head=8, attn_blocks=4, patch_size=patch_size, image_size=image_size)
        )
        self.layer_5 = nn.Sequential(
            InvertedResidual(channels[4], channels[5], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockV3_v1(channels[5], attn_dim[2], ffn_multiplier, heads=4, dim_head=8, attn_blocks=3, patch_size=patch_size, image_size=image_size)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.top_layer = nn.Conv2d(channels[-1], 256, kernel_size=1, stride=1, padding=0)

        # Lateral connections
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(channels[2], 256, kernel_size=1, stride=1, padding=0),  # layer_2
            nn.Conv2d(channels[3], 256, kernel_size=1, stride=1, padding=0),  # layer_3
            nn.Conv2d(channels[4], 256, kernel_size=1, stride=1, padding=0),  # layer_4
            nn.Conv2d(channels[5], 256, kernel_size=1, stride=1, padding=0)   # layer_5
        ])

        # Smooth layers
        self.smooth_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) for _ in range(5)
        ])

        # Upsample layers
        self.final_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.final_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.final_conv3 = nn.Conv2d(64, 1, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        x = self.conv_0(x)
        lat_1 = self.layer_1(x)
        lat_2 = self.layer_2(lat_1)
        lat_3 = self.layer_3(lat_2)
        lat_4 = self.layer_4(lat_3)
        lat_5 = self.layer_5(lat_4)
        
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

        smooth_features = [smooth(fpn) for smooth, fpn in zip(self.smooth_layers, fpn_features)]

        out = self.final_conv1(smooth_features[-1])
        out = self.final_conv2(out)
        out = self.final_conv3(out)
        
        return out


# Exemple d'utilisation
if __name__ == '__main__':
    input_tensor = torch.randn(1, 1, 4096, 1024)
    # Vous pouvez choisir entre MobileViTv3_v1 et MobileViTv3_v1_dynamicFPN selon votre besoin
    model = MobileViTv3_v1_dynamicFPN_LSA((4096, 1024), 'xx_small', 1000, (32, 32))
    # output = model(input_tensor)
    # print(output.size())
    # flops = FlopCountAnalysis(model, input_tensor)
    # print(flops.total())
    writer = SummaryWriter()
    writer.add_graph(model, input_tensor)
    writer.close()
