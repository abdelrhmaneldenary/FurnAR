class MyVit(nn.Module):
    def __init__(self, chw=(3,224,224), n_patches=14, hidden_d=768, n_blocks=12, n_heads=12, out_d=10):
        super(MyVit,self).__init__()
        self.chw=chw
        self.n_patches=n_patches
        self.n_blocks=n_blocks
        self.n_heads=n_heads
        self.hidden_d=hidden_d
        self.out_d=out_d
        
        # Calculate physical pixel size of the patch
        self.patch_h = int(chw[1] / n_patches)
        self.patch_w = int(chw[2] / n_patches)

        assert chw[1] % n_patches == 0
        assert chw[2] % n_patches == 0 
        
        # 1. Linear Mapper (Calculates exactly 768)
        self.input_d= int(chw[0] * self.patch_h * self.patch_w)      
        self.linear_mapper=nn.Linear(self.input_d, self.hidden_d)

        # 2. Class Token
        self.class_token=nn.Parameter(torch.rand(1, self.hidden_d))

        # 3. Positional Embedding (Fixed UserWarning)
        pos_data = get_positional_embedding(self.n_patches**2 + 1, self.hidden_d)
        self.pos_embedding = nn.Parameter(pos_data.clone().detach())

        # 4. Transformer Blocks
        self.blocks=nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # 5. MLP (Not used for DPT, but left here to prevent breaking old code)
        self.mlp=nn.Sequential(
            nn.Linear(self.hidden_d, self.out_d),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, images):
        n, c, h, w = images.shape
        
        patches = patchify(images, self.patch_h, self.patch_w)
        
        tokens = self.linear_mapper(patches)
        
        # Add class token
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        
        pos_embed = self.pos_embedding.repeat(n, 1, 1)
        out = tokens + pos_embed

        extracted_tokens = []
        for i, block in enumerate(self.blocks):
            out = block(out)
            if i in [2, 5, 8, 11]: 
                extracted_tokens.append(out)
                
        return extracted_tokens