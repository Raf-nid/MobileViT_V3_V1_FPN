import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        self.pool.to("cuda:0")
        self.downs.to("cuda:0")

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.ups.to("cuda:0")

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.bottleneck.to("cuda:0")

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.final_conv.to("cuda:0")

    def forward(self, x):
        
        x = (x+1)/2

        skip_connections = []

        x = x.to("cuda:0")

        for down in self.downs:
            x = down(x)

            skip_connections.append(x.to("cuda:0"))
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
    
        x = x.to("cuda:0")

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        x = self.final_conv(x)   
        x = 2*x-1

        x = x.to("cuda:0")

        return x 

class UNETred(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[32, 64, 128, 256],
    ):
        super(UNETred, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        self.pool.to("cuda:0")
        self.downs.to("cuda:0")

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.ups.to("cuda:0")

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.bottleneck.to("cuda:0")

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.final_conv.to("cuda:0")

    def forward(self, x):
        
        x = (x+1)/2

        skip_connections = []

        x = x.to("cuda:0")

        for down in self.downs:
            x = down(x)

            skip_connections.append(x.to("cuda:0"))
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
    
        x = x.to("cuda:0")

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        x = self.final_conv(x)   
        x = 2*x-1

        x = x.to("cuda:0")

        return x 