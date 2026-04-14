import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

__all__ = ['MobileNetV3', 'mobilenetv3']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3_forFPN(nn.Module):
    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3_forFPN, self).__init__()
        input_channel = 16
        last_channel = 1280

        if mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel

        self.layers_os4 = [conv_bn(1, input_channel, 2, nlin_layer=Hswish)]#after 1st bottle neck
        self.layers_os8 = [] #after 3rd bottle neck
        self.layers_os16 = [] #after 8th bottle neck
        self.layers_os32 = [] #last
        layers = [self.layers_os4, self.layers_os8, self.layers_os16, self.layers_os32]
        last_bneck = [1, 3, 8, 0]
        
        n_layer = 0
        layer = layers[n_layer]
        n_last_bneck = last_bneck[n_layer]
        # building mobile blocks
        for i, (k, exp, c, se, nl, s) in enumerate(mobile_setting):
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            layer.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

            if i+1 == n_last_bneck:
                n_layer +=1
                layer = layers[n_layer]
                n_last_bneck = last_bneck[n_layer]

        # make it nn.Sequential
        self.layers_os4, self.layers_os8, self.layers_os16, self.layers_os32 = [nn.Sequential(*layer) for layer in layers]

        self._initialize_weights()

    def forward(self, x):
        x1 = self.layers_os4(x)
        x2 = self.layers_os8(x1)
        x3 = self.layers_os16(x2)
        x4 = self.layers_os32(x3)

        return x4

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2, nlin_layer=Hswish)] #ici
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class MobileNetV3_dynamicFPN(nn.Module):
    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3_dynamicFPN, self).__init__()
        # Initialisation des canaux
        input_channel = 16
        last_channel = 1280
        self.width_mult = width_mult

        if mode == 'large':
            # Table 1 du papier
            mobile_setting = [
                # k, exp,  c,   se,    nl,  s
                [3, 16,   16,  False, 'RE', 1],
                [3, 64,   24,  False, 'RE', 2],
                [3, 72,   24,  False, 'RE', 1],
                [5, 72,   40,  True,  'RE', 2],
                [5, 120,  40,  True,  'RE', 1],
                [5, 120,  40,  True,  'RE', 1],
                [3, 240,  80,  False, 'HS', 2],
                [3, 200,  80,  False, 'HS', 1],
                [3, 184,  80,  False, 'HS', 1],
                [3, 184,  80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # Table 2 du papier
            mobile_setting = [
                # k, exp,  c,   se,    nl,  s
                [3, 16,   16,  True,  'RE', 2],
                [3, 72,   24,  False, 'RE', 2],
                [3, 88,   24,  False, 'RE', 1],
                [5, 96,   40,  True,  'HS', 2],
                [5, 240,  40,  True,  'HS', 1],
                [5, 240,  40,  True,  'HS', 1],
                [5, 120,  48,  True,  'HS', 1],
                [5, 144,  48,  True,  'HS', 1],
                [5, 288,  96,  True,  'HS', 2],
                [5, 576,  96,  True,  'HS', 1],
                [5, 576,  96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # Vérification de la taille d'entrée
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * self.width_mult) if self.width_mult > 1.0 else last_channel

        # --- Première couche ---
        self.first_layer = conv_bn(1, input_channel, 2, nlin_layer=Hswish)

        # --- Blocs mobiles ---
        self.inverted_residual_blocks = nn.ModuleList()
        self.block_strides = []
        self.block_out_channels = []
        # On utilise une variable locale pour suivre le nombre de canaux
        in_channel_local = input_channel
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * self.width_mult)
            exp_channel = make_divisible(exp * self.width_mult)
            self.inverted_residual_blocks.append(
                MobileBottleneck(in_channel_local, output_channel, k, s, exp_channel, se, nl)
            )
            self.block_strides.append(s)
            self.block_out_channels.append(output_channel)
            in_channel_local = output_channel
        # Mise à jour du nombre de canaux après ces blocs
        input_channel = in_channel_local

        # --- Dernières couches ---
        self.features = []
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.last_convolution = conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish)
            self.average_pool = nn.AdaptiveAvgPool2d(1)
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.last_convolution = conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish)
            self.average_pool = nn.AdaptiveAvgPool2d(1)
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError
        self.features = nn.Sequential(*self.features)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, n_class),
        )

        # --- FPN ---
        # Top layer : on utilise le nombre de canaux en sortie du dernier bloc mobile (input_channel)
        self.top_layer = nn.Conv2d(last_channel, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers : on sélectionne les blocs (sauf le dernier) dont le stride > 1
        self.lateral_indices = [i for i, s in enumerate(self.block_strides[:-1]) if s > 1]
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(self.block_out_channels[i], 256, kernel_size=1, stride=1, padding=0)
            for i in self.lateral_indices
        ])
        # Smooth layers : création de couches indépendantes
        self.smooth_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            for _ in range(len(self.lateral_layers) + 1)
        ])

        # --- Décodage final (upsample) ---
        self.final_conv = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.final_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.DerDesDer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # 1) Première couche
        x = self.first_layer(x)

        # 2) Passage dans chaque bloc Mobile (récupération des features latérales)
        lateral_tensors = []
        n_lateral_connections = 0
        for i, block in enumerate(self.inverted_residual_blocks):
            out = block(x)
            # Si le bloc a un stride > 1 et qu'on dispose d'une couche latérale
            if self.block_strides[i] > 1 and n_lateral_connections < len(self.lateral_layers):
                lateral_feat = self.lateral_layers[n_lateral_connections](out)
                lateral_tensors.append(lateral_feat)
                n_lateral_connections += 1
            x = out

        # 3) Dernières couches
        x = self.last_convolution(x)
        x = self.average_pool(x)
        x = self.features(x)

        # 4) FPN : top layer
        m_layers = [self.top_layer(x)]
        # 5) Reconstruction du FPN en remontant les lateral_tensors (ordre inversé)
        lateral_tensors.reverse()
        for lateral_tensor in lateral_tensors:
            top = self._upsample_add(m_layers[-1], lateral_tensor)
            m_layers.append(top)
        # 6) Lissage de chaque niveau
        p_layers = [smooth_layer(m_layer) for smooth_layer, m_layer in zip(self.smooth_layers, m_layers)]
        # 7) Décodage final / upsample final
        highest_feature = p_layers[-1]
        out = self.final_conv(highest_feature)
        out = self.final_conv2(out)
        fin = self.DerDesDer(out)
        return fin

    def _initialize_weights(self):
        # Initialisation des poids
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



def mobilenetv3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
        model.load_state_dict(state_dict, strict=True)
        # raise NotImplementedError
    return model

if __name__ == '__main__':
    #net = mobilenetv3()
    #print('mobilenetv3:\n', net)
    #print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    input_size=(1, 1, 4096, 1024)
    x = torch.randn(input_size)
    #out = net(x)
    #print(out.shape)
    # writer = SummaryWriter()
    # writer.add_graph(net, input_to_model=torch.randn(input_size))
    # writer.close()
#    net = MobileNetV3_dynamicFPN()
#    print('mobilenetv3:\n', net)
#    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
#    writer = SummaryWriter()
#    writer.add_graph(net, input_to_model=torch.randn(input_size))
#    writer.close()
#    # pip install --upgrade git+https://github.com/kuan-wang/pytorch-OpCounter.git
#    #from thop import profile
#    #flops, params = profile(net, input_size=input_size)
#    # print(flops)
#    # print(params)
#    #print('Total params: %.2fM' % (params/1000000.0))
#    #print('Total flops: %.2fM' % (flops/1000000.0))
#    x = torch.randn(input_size)
#    out = net(x)
#    print(out.shape)


