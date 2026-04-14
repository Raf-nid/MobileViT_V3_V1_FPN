
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class FPN(nn.Module):
    def __init__(self, out_channels=256):
        """
        Args:
            out_channels (int): Output channels for each FPN level.
        """
        super(FPN, self).__init__()
        # Lateral 1x1 convolutions to align channel counts
        self.latlayer5 = nn.Conv2d(1280, out_channels, kernel_size=1)
        self.latlayer4 = nn.Conv2d(96, out_channels, kernel_size=1)
        self.latlayer3 = nn.Conv2d(32, out_channels, kernel_size=1)
        self.latlayer2 = nn.Conv2d(24, out_channels, kernel_size=1)
        self.latlayer1 = nn.Conv2d(16, out_channels, kernel_size=1)
        
        # 3x3 smoothing convolutions after fusion
        self.smooth5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(out_channels, 1, kernel_size=1)
    def forward(self, features):
        """
        Args:
            features (list or tuple): Backbone feature maps in order
                [x1, x2, x3, x4, x5] where x1 is the highest resolution.
        Returns:
            Tensor upsampled to (4096, 1024) after fusion and final_conv.
        """
        # Unpack backbone features
        x1, x2, x3, x4, x5 = features
        
        # Top-down pathway
        # Deepest level (x5)
        p5 = self.latlayer5(x5)          # p5: [B, out_channels, 32, 128]
        p5 = self.smooth5(p5)            # optional smoothing on p5
        
        # Merge upsampled p5 with x4
        p4 = self.latlayer4(x4) + F.interpolate(p5, size=x4.shape[2:], mode='nearest')
        p4 = self.smooth4(p4)
        
        # Merge upsampled p4 with x3
        p3 = self.latlayer3(x3) + F.interpolate(p4, size=x3.shape[2:], mode='nearest')
        p3 = self.smooth3(p3)
        
        # Merge upsampled p3 with x2
        p2 = self.latlayer2(x2) + F.interpolate(p3, size=x2.shape[2:], mode='nearest')
        p2 = self.smooth2(p2)
        
        # Merge upsampled p2 with x1
        p1 = self.latlayer1(x1) + F.interpolate(p2, size=x1.shape[2:], mode='nearest')
        p1 = self.smooth1(p1)
        
        # Upsample all levels to p1 spatial size
        p2_upsampled = F.interpolate(p2, size=p1.shape[2:], mode='nearest')
        p3_upsampled = F.interpolate(p3, size=p1.shape[2:], mode='nearest')
        p4_upsampled = F.interpolate(p4, size=p1.shape[2:], mode='nearest')
        p5_upsampled = F.interpolate(p5, size=p1.shape[2:], mode='nearest')

        # Sum aligned feature maps
        combined = p1 + p2_upsampled + p3_upsampled + p4_upsampled + p5_upsampled
        
        # Convolution finale
        output = self.final_conv(combined)
        
        # Final upsampling to target resolution
        output = F.interpolate(output, size=(4096, 1024), mode='nearest')
        
        return output





# Quick sanity check when this module is run as a script
if __name__ == '__main__':
    #writer = SummaryWriter()
    # Dummy tensors mimicking MobileNetV2 multi-scale outputs
    x1 = torch.randn(1, 16, 512, 2048)
    x2 = torch.randn(1, 24, 256, 1024)
    x3 = torch.randn(1, 32, 128, 512)
    x4 = torch.randn(1, 96, 64, 256)
    x5 = torch.randn(1, 1280, 32, 128)
    
    features = [x1, x2, x3, x4, x5]
    
    fpn = FPN(out_channels=256)
    outputs = fpn(features)
    print(f"Output shape: {outputs.shape}")

    #for idx, out in enumerate(outputs, 1):
        #print(f"p{idx} shape: {out.shape}")
    #dummy_input = torch.randn(1, 1, 1024, 4096).to("cuda:1")  # example dummy input for add_graph
    #writer.add_graph(fpn, (features,))
    #writer.close()