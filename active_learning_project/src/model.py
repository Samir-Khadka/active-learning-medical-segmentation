from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Standard Dual Convolution Block with BatchNorm and Dropout.
    A hallmark of robust architecture: modularity and defensive layers.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UNet(nn.Module):
    """
    Lightweight UNet implementation for medical image segmentation.
    Designed for rapid iteration without sacrificing architectural integrity.
    """
    def __init__(self, n_channels: int = 1, n_classes: int = 1, feature_scale: int = 4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Scaling features for lightweight execution
        f = [16 * (2**i) // feature_scale for i in range(5)]
        # Default with scale 4: 4, 8, 16, 32, 64
        # We'll use a slightly larger default for better convergence
        f = [16, 32, 64, 128, 256]

        # Encoder (Contracting Path)
        self.inc = ConvBlock(n_channels, f[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(f[0], f[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(f[1], f[2]))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(f[2], f[3]))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), ConvBlock(f[3], f[4]))

        # Decoder (Expansive Path)
        self.up1 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.conv_up1 = ConvBlock(f[4], f[3])
        
        self.up2 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.conv_up2 = ConvBlock(f[3], f[2])
        
        self.up3 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.conv_up3 = ConvBlock(f[2], f[1])
        
        self.up4 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.conv_up4 = ConvBlock(f[1], f[0])
        
        self.outc = nn.Conv2d(f[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bridge
        b = self.bottleneck(x4)
        
        # Decoder with Skip Connections
        u1 = self.up1(b)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv_up2(u2)
        
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv_up3(u3)
        
        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.conv_up4(u4)
        
        logits = self.outc(u4)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Global Average Pooling of bottleneck features for diversity metrics.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b = self.bottleneck(x4)
        return F.adaptive_avg_pool2d(b, (1, 1)).view(b.size(0), -1)

class DiceBCELoss(nn.Module):
    """
    Veteran Architect's Loss: Combines BCE with Dice Loss.
    Standard BCE fails on severe class imbalance typical in medical imaging.
    Dice loss mathematically enforces spatial overlap regardless of background dominance.
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Binary Cross Entropy
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # Dice Loss
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return bce + (1 - dice)
