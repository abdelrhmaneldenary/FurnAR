import torch
import torch.nn as nn
import torch.nn.functional as F

class ReassembleBlock(nn.Module):
    def __init__(self, in_channels, out_channels=256, spatial_size=14, scale_factor=1):
        super(ReassembleBlock, self).__init__()
        self.spatial_size = spatial_size
        
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        if scale_factor > 1:
            self.resample = nn.ConvTranspose2d(
                out_channels, out_channels, 
                kernel_size=int(scale_factor*2), 
                stride=int(scale_factor), 
                padding=int(scale_factor//2)
            )
        elif scale_factor < 1:
            stride = int(1 / scale_factor)
            self.resample = nn.Conv2d(
                out_channels, out_channels, 
                kernel_size=3, stride=stride, padding=1
            )
        else:
            self.resample = nn.Identity()

    def forward(self, tokens):
        batch_size = tokens.shape[0]
        
        cls_token = tokens[:, 0:1, :]
        spatial_tokens = tokens[:, 1:, :]
        spatial_tokens = spatial_tokens + cls_token 
        
        grid = spatial_tokens.reshape(batch_size, self.spatial_size, self.spatial_size, -1)
        grid = grid.permute(0, 3, 1, 2) 
        
        out = self.project(grid)
        out = self.resample(out)
        
        return out