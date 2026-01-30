import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SRMConv2d(nn.Module):
    """
    Spatial Rich Model (SRM) Convolution Layer.
    Uses fixed high-pass filters to extract noise residuals from images.
    These filters are standard in steganalysis and digital forensics.
    """
    def __init__(self, in_channels=3):
        super(SRMConv2d, self).__init__()
        self.in_channels = in_channels
        
        # Define 3 standard SRM filters (5x5)
        # Filter 1: KV kernel (preserves high freq edges)
        q_kv = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0

        # Filter 2: Horizontal edge
        q_h = np.array([
            [-1, 2, -2, 2, -1],
            [-2, 4, -4, 4, -2],
            [ 2, -4, 4, -4,  2],
            [-2, 4, -4, 4, -2],
            [-1, 2, -2, 2, -1]
        ], dtype=np.float32) / 4.0

        # Filter 3: Spam 1 (symmetrized)
        q_s = np.array([
            [-1, 2, -2, 2, -1],
            [ 2, -6, 8, -6,  2],
            [-2, 8, -12, 8, -2],
            [ 2, -6, 8, -6,  2],
            [-1, 2, -2, 2, -1]
        ], dtype=np.float32) / 12.0

        # Stack filters: result shape (3, 1, 5, 5)
        filters = np.stack([q_kv, q_h, q_s])
        filters = torch.from_numpy(filters).float()
        
        # Repeat for each input channel if needed, or apply independently.
        # We want to apply each filter to each channel (R, G, B) or convert to gray.
        # Here we treat RGB channels independently: 3 input channels * 3 filters = 9 output maps
        self.weight = nn.Parameter(filters.unsqueeze(1).repeat(1, in_channels, 1, 1), requires_grad=False)
        # Or simpler: Apply 3 filters to each channel separately. 
        # Better approach for signal: Convert RGB to Grayscale inside or apply to each channel.
        # Let's apply to each channel -> 3 filters * 3 channels = 9 output channels if grouped,
        # but standard SRM is often applied to Y channel. Let's keep it simple:
        # Output 3 channels (one per filter, summed over input) or 3*3 ?
        # Let's do: 3 filters independently applied to 3 channels -> 9 output channels.
        
        # Correct weight shape for separate application: (out_channels, in_channels/groups, k, k)
        # We want 3 output feature maps per input channel.
        # Total out = 3 * 3 = 9. Groups = 3.
        
        self.n_filters = 3
        
        # Reshape to (3*3, 1, 5, 5) for grouped conv
        self.weight = nn.Parameter(torch.cat([filters.unsqueeze(1)]*in_channels, dim=0), requires_grad=False)
        
    def forward(self, x):
        # x: [B, 3, H, W]
        # We effectively apply the 3 SRM filters to each of the R, G, B channels.
        # Result: [B, 9, H, W]
        return F.conv2d(x, self.weight, stride=1, padding=2, groups=self.in_channels)

def simple_test():
    layer = SRMConv2d()
    print("SRM Filters loaded:", layer.weight.shape)
    x = torch.randn(1, 3, 224, 224)
    out = layer(x)
    print("Output shape:", out.shape)

if __name__ == "__main__":
    simple_test()
