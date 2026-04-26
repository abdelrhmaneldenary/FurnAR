class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x 

class FeatureFusionBlock(nn.Module):
    def __init__(self, features=256):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
        
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.project = nn.Conv2d(features, features, kernel_size=3, padding=1)

    def forward(self, x, previous_stage_output=None):
        out = self.resConfUnit1(x)
        
        if previous_stage_output is not None:
            out = out + previous_stage_output
            
        out = self.resConfUnit2(out)
        out = self.upsample(out)
        out = self.project(out)
        
        return out