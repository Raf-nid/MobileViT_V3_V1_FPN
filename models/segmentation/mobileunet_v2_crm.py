import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.tensorboard import SummaryWriter

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(in_channels=inp, out_channels=inp * expand_ratio,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # depthwise convolution via groups
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio,
                      kernel_size=3, stride=stride, padding=1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pointwise linear convolution
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=oup,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_dynamicFPN(nn.Module):
    def __init__(self, width_mult=1.):
        super(MobileNetV2_dynamicFPN, self).__init__()

        self.input_channel = int(32 * width_mult)
        self.width_mult = width_mult

        # First layer
        self.first_layer = nn.Sequential(
            nn.Conv2d(1, self.input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.input_channel),
            nn.ReLU6(inplace=True)
        )

        # Inverted residual blocks (each n layers)
        self.inverted_residual_setting = [
            {'expansion_factor': 1, 'width_factor': 16, 'n': 1, 'stride': 1},
            {'expansion_factor': 6, 'width_factor': 24, 'n': 2, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 32, 'n': 3, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 64, 'n': 4, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 96, 'n': 3, 'stride': 1},
            {'expansion_factor': 6, 'width_factor': 160, 'n': 3, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 320, 'n': 1, 'stride': 1},
        ]
        self.inverted_residual_blocks = nn.ModuleList(
            [self._make_inverted_residual_block(**setting)
             for setting in self.inverted_residual_setting])

        # reduce feature maps to one pixel
        self.average_pool = nn.AdaptiveAvgPool2d(1)

        # Top layer
        self.top_layer = nn.Conv2d(
            int(self.inverted_residual_setting[-1]['width_factor'] * self.width_mult),
            256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.lateral_setting = [setting for setting in self.inverted_residual_setting[:-1]
                                if setting['stride'] > 1]
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(int(setting['width_factor'] * self.width_mult),
                      256, kernel_size=1, stride=1, padding=0)
            for setting in self.lateral_setting])

        # Smooth layers (one per lateral level + top)
        self.smooth_layers = nn.ModuleList(
            [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)] *
            (len(self.lateral_layers) + 1)
        )

        # Final decoder head (single-channel output path)
        self.final_conv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.final_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        
        self.DerDesDer = nn.Conv2d(64,1,kernel_size=1)

        self._initialize_weights()

    def _make_inverted_residual_block(self, expansion_factor, width_factor, n, stride):
        inverted_residual_block = []
        output_channel = int(width_factor * self.width_mult)
        for i in range(n):
            # only the first block in the stage uses the configured stride
            current_stride = stride if i == 0 else 1
            inverted_residual_block.append(
                InvertedResidual(self.input_channel, output_channel, current_stride, expansion_factor))
            self.input_channel = output_channel

        return nn.Sequential(*inverted_residual_block)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

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

    def forward(self, x):
        # -----------------
        # Bottom-up pathway
        x = self.first_layer(x)

        lateral_tensors = []
        n_lateral_connections = 0
        for i, block in enumerate(self.inverted_residual_blocks):
            output = block(x)
            # add a lateral connection when spatial resolution shrinks
            if self.inverted_residual_setting[i]['stride'] > 1 and n_lateral_connections < len(self.lateral_layers):
                lateral_tensors.append(self.lateral_layers[n_lateral_connections](output))
                n_lateral_connections += 1
            x = output

        x = self.average_pool(x)
        m_layers = [self.top_layer(x)]

        # -----------------
        # Top-down pathway
        lateral_tensors.reverse()
        for lateral_tensor in lateral_tensors:
            m_layers.append(self._upsample_add(m_layers[-1], lateral_tensor))

        # Apply smoothing conv on each pyramid level
        p_layers = [smooth_layer(m_layer) for smooth_layer, m_layer in zip(self.smooth_layers, m_layers)]
        
        # Highest-resolution pyramid level, then decode
        highest_feature = p_layers[-1]  # e.g. (N, 256, H, W)
        #upsampled = F.interpolate(highest_feature, size=(4069, 1024), mode='bilinear', align_corners=False)
        out = self.final_conv(highest_feature)  # 256 -> 128 channels
        out = self.final_conv2(out)
        fin = self.DerDesDer(out)
        return fin


if __name__ == '__main__':
    # Smoke test / TensorBoard graph
    net = MobileNetV2_dynamicFPN()
    print(net)
    input_tensor = torch.randn(1, 1, 4096, 1024)
    output = net(input_tensor)
    print("Output shape:", output.shape)
    # TensorBoard graph export:
    writer = SummaryWriter()
    writer.add_graph(net, input_tensor)
    writer.close()
    # TensorBoard: tensorboard --logdir=runs
