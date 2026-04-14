import logging
import math
import sys

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from ..backbones.mobilenet_v2 import InvertedResidual, MobileNetV2


class MobileNetV2_unet(nn.Module):
    def __init__(self, pre_trained='weights/mobilenet_v2.pth.tar', mode='train'):
        super(MobileNetV2_unet, self).__init__()

        self.mode = mode
        self.backbone = MobileNetV2()

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2) # 1280 is the output of the last layer of the backbone 96 is the number of channels in the first layer of the decoder
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)

        self.conv_last = nn.Conv2d(16, 1, 1)  # single-channel output

        #self.conv_score = nn.Conv2d(3, 1, 1)
        self.final_deconv = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)
        
        self._init_weights()

        if pre_trained is not None:
            # self.backbone.load_state_dict(torch.load(pre_trained, map_location="cpu"))
            checkpoint = torch.load(pre_trained, map_location="cuda:1")
            self.backbone.load_state_dict(torch.load(pre_trained))

    def forward(self, x):
        for n in range(0, 2):
            x = self.backbone.features[n](x)  # run early backbone stages
        x1 = x
        logging.debug((x1.shape, 'x1'))

        for n in range(2, 4):
            x = self.backbone.features[n](x)
        x2 = x
        logging.debug((x2.shape, 'x2'))

        for n in range(4, 7):
            x = self.backbone.features[n](x)
        x3 = x
        logging.debug((x3.shape, 'x3'))

        for n in range(7, 14):
            x = self.backbone.features[n](x)
        x4 = x
        logging.debug((x4.shape, 'x4'))

        for n in range(14, 19):
            x = self.backbone.features[n](x)
        x5 = x
        logging.debug((x5.shape, 'x5'))

        up1 = torch.cat([  # concat encoder skip with decoder upsample
            x4,
            self.dconv1(x)
        ], dim=1)  # channel-wise concat
        up1 = self.invres1(up1)
        logging.debug((up1.shape, 'up1'))

        up2 = torch.cat([
            x3,
            self.dconv2(up1)
        ], dim=1)
        up2 = self.invres2(up2)
        logging.debug((up2.shape, 'up2'))

        up3 = torch.cat([
            x2,
            self.dconv3(up2)
        ], dim=1)
        up3 = self.invres3(up3)
        logging.debug((up3.shape, 'up3'))

        up4 = torch.cat([
            x1,
            self.dconv4(up3)
        ], dim=1)
        up4 = self.invres4(up4)
        logging.debug((up4.shape, 'up4'))

        x = self.conv_last(up4)
        logging.debug((x.shape, 'conv_last'))

        #x = self.conv_score(x)
        #logging.debug((x.shape, 'conv_score'))
        x = self.final_deconv(x)
        #if self.mode == "eval":
        #x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        #logging.debug((x.shape, 'interpolate'))

        x = x  # optional: torch.tanh(x)
        #x = torch.tanh(x)
        # x = torch.nn.Softmax(x)
        # x = torch.nn.sigmoid(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    # Debug
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = MobileNetV2_unet(pre_trained=None)
    net(torch.randn(1, 1, 1024, 4096))

    input_tensor = torch.randn(1, 1, 1024, 4096)
    output_tensor = net(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
