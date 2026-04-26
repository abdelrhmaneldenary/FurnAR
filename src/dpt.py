class DPT(nn.Module):
    def __init__(self, vit_encoder, embed_dim=768, features=256, spatial_size=14):
        super(DPT, self).__init__()
        self.encoder = vit_encoder
        
        self.reassemble_4 = ReassembleBlock(embed_dim, features, spatial_size, scale_factor=4)
        self.reassemble_8 = ReassembleBlock(embed_dim, features, spatial_size, scale_factor=2)
        self.reassemble_16 = ReassembleBlock(embed_dim, features, spatial_size, scale_factor=1)
        self.reassemble_32 = ReassembleBlock(embed_dim, features, spatial_size, scale_factor=0.5)
        
        self.fusion_32 = FeatureFusionBlock(features)
        self.fusion_16 = FeatureFusionBlock(features)
        self.fusion_8 = FeatureFusionBlock(features)
        self.fusion_4 = FeatureFusionBlock(features)
        
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 2, kernel_size=1),
            nn.Tanh() 
        )

    def forward(self, x):
        layer_tokens = self.encoder(x)
        t3, t6, t9, t12 = layer_tokens 
        
        f4 = self.reassemble_4(t3)
        f8 = self.reassemble_8(t6)
        f16 = self.reassemble_16(t9)
        f32 = self.reassemble_32(t12)
        
        out = self.fusion_32(f32)
        out = self.fusion_16(f16, previous_stage_output=out)
        out = self.fusion_8(f8, previous_stage_output=out)
        out = self.fusion_4(f4, previous_stage_output=out)
        
        color_map = self.head(out)
        
        return color_map