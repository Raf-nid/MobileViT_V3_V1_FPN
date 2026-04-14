from torch.autograd import Variable
import torch.nn as nn
import torch
import math
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

def Conv_3x3(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def Conv_1x1(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def SepConv_3x3(inp, oup): #input=32, output=16
    return nn.Sequential(
        # dw
        nn.Conv2d(inp, inp , 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, kernel // 2, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MnasNet(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MnasNet, self).__init__()

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, k
            [3, 24,  3, 2, 3],  # -> 56x56
            [3, 40,  3, 2, 5],  # -> 28x28
            [6, 80,  3, 2, 5],  # -> 14x14
            [6, 96,  2, 1, 3],  # -> 14x14
            [6, 192, 4, 2, 5],  # -> 7x7
            [6, 320, 1, 1, 3],  # -> 7x7
        ]

        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        # building first two layer
        self.features = [Conv_3x3(1, input_channel, 2), SepConv_3x3(input_channel, 16)]
        input_channel = 16

        # building inverted residual blocks (MBConv)
        for t, c, n, s, k in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t, k))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t, k))
                input_channel = output_channel

        # building last several layers
        self.features.append(Conv_1x1(input_channel, self.last_channel))
        self.features.append(nn.AdaptiveAvgPool2d(1))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MnasNet_dynamic(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MnasNet_dynamic, self).__init__()

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, k
            [3, 24,  3, 2, 3],  # -> 56x56
            [3, 40,  3, 2, 5],  # -> 28x28
            [6, 80,  3, 2, 5],  # -> 14x14
            [6, 96,  2, 1, 3],  # -> 14x14
            [6, 192, 4, 2, 5],  # -> 7x7
            [6, 320, 1, 1, 3],  # -> 7x7
        ]

        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel_num = int(1280 * width_mult) if width_mult > 1.0 else 1280

        # building first two layers
        self.first_layer = nn.Sequential(
            Conv_3x3(1, input_channel, 2),
            SepConv_3x3(input_channel, 16)
        )
        input_channel = 16

        # building inverted residual blocks (MBConv)
        self.inverted_residual_blocks = nn.ModuleList()
        for t, c, n, s, k in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.inverted_residual_blocks.append(InvertedResidual(input_channel, output_channel, s, t, k))
                else:
                    self.inverted_residual_blocks.append(InvertedResidual(input_channel, output_channel, 1, t, k))
                input_channel = output_channel

        # building last several layers
        self.last_conv = Conv_1x1(input_channel, self.last_channel_num)
        self.average_pool = nn.AdaptiveAvgPool2d(1)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel_num, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        
        x = self.first_layer(x)
        for block in self.inverted_residual_blocks:
            x = block(x)
        x = self.last_conv(x)
        x = self.average_pool(x)
        x = x.view(x.size(0), self.last_channel_num)  # Assure que le tenseur est correctement aplati
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    

class MnasNet_dynamicFPN(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MnasNet_dynamicFPN, self).__init__()

        # Paramètres des MBConv (InvertedResidual)
        self.interverted_residual_setting = [
            # t,  c,   n,  s,  k
            [3,  24,   3,  2,  3],
            [3,  40,   3,  2,  5],
            [6,  80,   3,  2,  5],
            [6,  96,   2,  1,  3],
            [6, 192,   4,  2,  5],
            [6, 320,   1,  1,  3],
        ]
        self.width_mult = width_mult
        assert input_size % 32 == 0

        # -------------------------------------------------
        # 1) Construction des premières couches
        # -------------------------------------------------
        input_channel = int(32 * self.width_mult)
        self.last_channel_num = int(1280 * self.width_mult) if self.width_mult > 1.0 else 1280

        # Première couche + conv séparables
        self.first_layer = nn.Sequential(
            Conv_3x3(1, input_channel, 2),
            SepConv_3x3(input_channel, 16)
        )
        input_channel = 16

        # -------------------------------------------------
        # 2) Construction des MBConv
        # -------------------------------------------------
        self.inverted_residual_blocks = nn.ModuleList()
        # On crée une liste "dépliée" des strides pour chaque bloc
        self.block_strides = []
        for t, c, n, s, k in self.interverted_residual_setting:
            output_channel = int(c * self.width_mult)
            # Premier bloc du groupe : stride = s
            self.inverted_residual_blocks.append(
                InvertedResidual(input_channel, output_channel, s, t, k)
            )
            self.block_strides.append(s)

            # Blocs suivants du groupe : stride = 1
            for _ in range(n - 1):
                self.inverted_residual_blocks.append(
                    InvertedResidual(output_channel, output_channel, 1, t, k)
                )
                self.block_strides.append(1)

            input_channel = output_channel

        # -------------------------------------------------
        # 3) Dernières couches
        # -------------------------------------------------
        self.last_conv = Conv_1x1(input_channel, self.last_channel_num)
        self.average_pool = nn.AdaptiveAvgPool2d(1)

        # Classification
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel_num, n_class),
        )

        # -------------------------------------------------
        # 4) FPN
        # -------------------------------------------------
        # a) Top layer
        self.top_layer = nn.Conv2d(
            int(self.interverted_residual_setting[-1][1] * self.width_mult),
            256, kernel_size=1, stride=1, padding=0
        )
        
        # b) Lateral layers
        # On ignore le dernier groupe, puis on ne retient que ceux avec stride > 1
        self.lateral_setting = [
            setting for setting in self.interverted_residual_setting[:-1]
            if setting[3] > 1
        ]
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(int(setting[1] * self.width_mult),
                      256, kernel_size=1, stride=1, padding=0)
            for setting in self.lateral_setting
        ])
        
        # c) Smooth layers (1 par niveau de FPN)
        # !! IMPORTANT: utiliser une list comprehension pour avoir des couches distinctes
        self.smooth_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            for _ in range(len(self.lateral_layers) + 1)
        ])

        # d) Upsample final
        self.final_conv = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=4, stride=2, padding=1
        )
        self.final_conv2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=4, stride=2, padding=1
        )
        self.DerDesDer = nn.Conv2d(64, 1, kernel_size=1)

        # Initialisation
        self._initialize_weights()

    def _upsample_add(self, x, y):
        """
        Upsample x pour qu'il ait la même taille que y,
        puis additionne les deux tenseurs.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # 1) Passage dans la première couche
        x = self.first_layer(x)

        # 2) Passage dans chaque bloc MBConv
        #    + récupération des features "latéraux"
        lateral_tensors = []
        n_lateral_connections = 0

        for i, block in enumerate(self.inverted_residual_blocks):
            out = block(x)
            stride_i = self.block_strides[i]
            
            # Si le bloc a un stride > 1 et qu'on a encore des couches latérales dispo
            if stride_i > 1 and n_lateral_connections < len(self.lateral_layers):
                # On applique la couche latérale correspondante
                lateral_feat = self.lateral_layers[n_lateral_connections](out)
                lateral_tensors.append(lateral_feat)
                n_lateral_connections += 1

            x = out

        # 3) Pooling global + top layer du FPN
        x = self.average_pool(x)  # shape = [N, C, 1, 1]
        m_layers = [self.top_layer(x)]  # shape = [N, 256, 1, 1]

        # 4) On reconstruit le FPN en remontant les lateral_tensors (inversés)
        lateral_tensors.reverse()  # on part du plus profond au plus haut
        for lateral_tensor in lateral_tensors:
            # On upsample le dernier feature map et on l’additionne à la carte latérale
            top = self._upsample_add(m_layers[-1], lateral_tensor)
            m_layers.append(top)

        # 5) Lissage de chaque niveau de FPN
        p_layers = [
            smooth_layer(m_layer)
            for smooth_layer, m_layer in zip(self.smooth_layers, m_layers)
        ]

        # 6) On prend le dernier niveau (plus haute résolution) pour la suite
        highest_feature = p_layers[-1]

        # 7) Décodage/upsample final
        out = self.final_conv(highest_feature)
        out = self.final_conv2(out)
        fin = self.DerDesDer(out)

        return fin

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    net = MnasNet_dynamicFPN()  # ou MnasNet, ou MnasNet_dynamic
    net.to(device)
    x_image = torch.randn(1, 1, 4096, 1024).to(device)
    y = net(x_image)
    #writer = SummaryWriter()
    #writer.add_graph(net, x_image)
    #writer.close()
    print(y.shape)