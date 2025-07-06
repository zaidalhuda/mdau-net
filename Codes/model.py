"""
MDAU-Net: Multi-Scale U-Net with Dual Attention Module for Pavement Crack Segmentation

Simple implementation of the MDAU-Net architecture.

Author: Zaid Al-Huda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAttentionModule(nn.Module):
    """Dual Attention Module combining GAP and Position Attention."""
    
    def __init__(self, in_channels):
        super(DualAttentionModule, self).__init__()
        
        # Global Average Pooling Attention
        self.gap_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Position Attention
        self.position_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # GAP Attention
        gap_att = self.gap_attention(x)
        gap_feature = x * gap_att
        
        # Position Attention  
        pos_att = self.position_attention(x)
        pos_feature = x * pos_att
        
        # Combine features
        combined = gap_feature + pos_feature
        output = self.relu(self.bn(self.fusion_conv(combined)))
        
        return output


class ConvBlock(nn.Module):
    """Basic convolutional block."""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UpConvBlock(nn.Module):
    """Upsampling block for decoder."""
    
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)
    
    def forward(self, x, skip_connection):
        x = self.upconv(x)
        if x.shape[2:] != skip_connection.shape[2:]:
            x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv_block(x)
        return x


class MDAUNet(nn.Module):
    """
    MDAU-Net: Multi-Scale U-Net with Dual Attention Module
    
    Args:
        in_channels (int): Number of input channels (default: 3)
        out_channels (int): Number of output channels (default: 1)
        base_channels (int): Base number of channels (default: 64)
        scales (list): Multi-scale factors (default: [0.5, 1.0, 1.5])
    """
    
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, scales=[0.5, 1.0, 1.5]):
        super(MDAUNet, self).__init__()
        
        self.scales = scales
        
        # Multi-scale encoders
        self.encoders = nn.ModuleList()
        for _ in scales:
            encoder = nn.ModuleList([
                ConvBlock(in_channels, base_channels),      # 64
                ConvBlock(base_channels, base_channels * 2),    # 128  
                ConvBlock(base_channels * 2, base_channels * 4), # 256
                ConvBlock(base_channels * 4, base_channels * 8), # 512
            ])
            self.encoders.append(encoder)
        
        self.max_pool = nn.MaxPool2d(2, 2)
        
        # Dual Attention Module
        bottleneck_channels = base_channels * 8 * len(scales)
        self.dual_attention = DualAttentionModule(bottleneck_channels)
        
        # Decoder
        self.decoder = nn.ModuleList([
            UpConvBlock(bottleneck_channels, base_channels * 4),
            UpConvBlock(base_channels * 4, base_channels * 2),
            UpConvBlock(base_channels * 2, base_channels),
            UpConvBlock(base_channels, base_channels // 2)
        ])
        
        # Final classification layer
        self.final_conv = nn.Conv2d(base_channels // 2, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Multi-scale encoding
        multi_scale_features = []
        all_skip_connections = []
        
        for scale, encoder in zip(self.scales, self.encoders):
            # Scale input
            if scale != 1.0:
                scaled_h, scaled_w = int(height * scale), int(width * scale)
                scaled_x = F.interpolate(x, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
            else:
                scaled_x = x
            
            # Encode at current scale
            skip_connections = []
            current = scaled_x
            
            for i, conv_block in enumerate(encoder):
                current = conv_block(current)
                if i < len(encoder) - 1:
                    skip_connections.append(current)
                    current = self.max_pool(current)
            
            # Resize features back to consistent size
            target_h, target_w = height // 16, width // 16
            if current.shape[2:] != (target_h, target_w):
                current = F.interpolate(current, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            # Resize skip connections
            resized_skips = []
            for j, skip in enumerate(skip_connections):
                target_skip_h, target_skip_w = height // (2 ** (j + 1)), width // (2 ** (j + 1))
                if skip.shape[2:] != (target_skip_h, target_skip_w):
                    skip = F.interpolate(skip, size=(target_skip_h, target_skip_w), mode='bilinear', align_corners=False)
                resized_skips.append(skip)
            
            multi_scale_features.append(current)
            all_skip_connections.append(resized_skips)
        
        # Fuse multi-scale features
        fused_features = torch.cat(multi_scale_features, dim=1)
        
        # Apply dual attention
        attended_features = self.dual_attention(fused_features)
        
        # Combine skip connections from all scales
        combined_skips = []
        for level in range(len(all_skip_connections[0])):
            level_features = [skips[level] for skips in all_skip_connections]
            combined_skip = torch.cat(level_features, dim=1)
            combined_skips.append(combined_skip)
        
        # Decode to get final segmentation
        current = attended_features
        for i, (decoder_block, skip) in enumerate(zip(self.decoder, reversed(combined_skips))):
            current = decoder_block(current, skip)
        
        # Final output
        output = self.final_conv(current)
        output = F.interpolate(output, size=(height, width), mode='bilinear', align_corners=False)
        output = self.sigmoid(output)
        
        return output


def mdau_net(in_channels=3, out_channels=1, base_channels=64, scales=[0.5, 1.0, 1.5]):
    """Create MDAU-Net model."""
    return MDAUNet(in_channels, out_channels, base_channels, scales)


if __name__ == "__main__":
    # Test model
    model = mdau_net()
    x = torch.randn(2, 3, 512, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")