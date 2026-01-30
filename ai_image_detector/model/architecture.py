import torch
import torch.nn as nn
import timm
from .layers import SRMConv2d

class DualStreamDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(DualStreamDetector, self).__init__()
        
        # Stream 1: RGB Stream (Semantic / Content)
        # Using a lightweight EfficientNet-B0
        self.rgb_backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
        
        # Stream 2: Noise Stream (Artifacts / Signal)
        # Starts with fixed SRM filters, then a small custom CNN or another lightweight backbone
        self.srm_layer = SRMConv2d(in_channels=3) # Output: 9 channels
        
        # We can use a smaller backbone for noise, like MobileNet or a custom sequential
        # Let's use a small ResNet18-like or just a tiny efficientnet with modified input
        # Note: timm models expect 3 input channels. We have 9.
        # We can use a 1x1 conv to reduce 9->3 or initialize a model with 9 input channels.
        # Option A: 1x1 Conv 9->3
        self.noise_adapter = nn.Conv2d(9, 3, kernel_size=1)
        self.noise_backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
        
        # Feature dimensions
        # EfficientNet-B0 num_features = 1280
        self.rgb_dim = self.rgb_backbone.num_features
        self.noise_dim = self.noise_backbone.num_features
        
        # Fusion & Classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.rgb_dim + self.noise_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2) # Real vs AI
        )
        
    def forward(self, x):
        # x: [B, 3, H, W] (RGB, normalized)
        
        # --- Stream 1: RGB ---
        rgb_feat = self.rgb_backbone(x) # [B, 1280]
        
        # --- Stream 2: Noise ---
        # 1. Extract noise residuals (removes content, leaves high-freq artifacts)
        # Note: Ideally x should be denormalized or raw for SRM, but standard normalization 
        # just scales the noise, which is fine for CNNs.
        noise = self.srm_layer(x) # [B, 9, H, W]
        
        # 2. Adapt to 3 channels for backbone
        noise = self.noise_adapter(noise) # [B, 3, H, W]
        
        # 3. Extract high-level noise statistics
        noise_feat = self.noise_backbone(noise) # [B, 1280]
        
        # --- Fusion ---
        concat = torch.cat((rgb_feat, noise_feat), dim=1) # [B, 2560]
        logits = self.classifier(concat)
        
        return logits

if __name__ == "__main__":
    model = DualStreamDetector()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}") # Should be [2, 2]
