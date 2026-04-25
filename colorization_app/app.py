import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import gradio as gr
from skimage import color
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

# ====================================================================
# 1. PASTE YOUR ARCHITECTURE HERE
# Copy and paste the following classes from your Kaggle notebook:
# - get_positional_embedding
import torch
import math

def get_positional_embedding(sequence_length, d):

    positions = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
    
    results = torch.zeros(sequence_length, d)
    
    results[:, 0::2] = torch.sin(positions * div_term)
    
    results[:, 1::2] = torch.cos(positions * div_term)
    
    return results
# - patchify
import torch

def patchify(images, patch_h, patch_w):
    n, c, h, w = images.shape
    assert h % patch_h == 0, f"Image height {h} not divisible by patch height {patch_h}"
    assert w % patch_w == 0, f"Image width {w} not divisible by patch width {patch_w}"
    
    num_patches_h = h // patch_h
    num_patches_w = w // patch_w
    
    patches = images.reshape(n, c, num_patches_h, patch_h, num_patches_w, patch_w)
    
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    

    patches = patches.reshape(n, num_patches_h * num_patches_w, -1)
    
    return patches

# - MyMsa
class MyMsa(nn.Module):
    def __init__(self,d,n_heads=2):
        super(MyMsa,self).__init__()
        self.d=d
        self.n_heads=n_heads

        assert d%n_heads==0

        d_head=int(d/n_heads)
        self.q_mappings=nn.ModuleList([nn.Linear(d_head,d_head) for _ in range(self.n_heads)])
        self.k_mappings=nn.ModuleList([nn.Linear(d_head,d_head) for _ in range(self.n_heads)])
        self.v_mappings=nn.ModuleList([nn.Linear(d_head,d_head) for _ in range(self.n_heads)])
        self.d_head=d_head
        self.softmax=nn.Softmax(dim=-1)
        self.out_proj=nn.Linear(d,d)

    def forward(self,sequences):
        result=[]
        for sequence in sequences:
            seq_result=[]
            for head in range(self.n_heads):
                q_mapping=self.q_mappings[head]
                k_mapping=self.k_mappings[head]
                v_mapping=self.v_mappings[head]

                seq=sequence[:,head*self.d_head:(head+1)*self.d_head]
                q,k,v=q_mapping(seq),k_mapping(seq),v_mapping(seq)

                attention=self.softmax(q@k.T/(self.d_head**2))
                seq_result.append(attention@v)

            result.append(torch.hstack(seq_result))

        return self.out_proj(torch.cat([torch.unsqueeze(r,dim=0) for r in result]))
                
        
# - MyViTBlock
class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMsa(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out
# - MyVit
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
# - ReassembleBlock
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
# - ResidualConvUnit
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

# - FeatureFusionBlock

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
# - DPT
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class PretrainedViTEncoder(nn.Module):
    def __init__(self):
        super(PretrainedViTEncoder, self).__init__()
        
        # 1. Download the official, pre-trained ViT-Base (Patch 16, 224x224)
        print("Downloading pre-trained ImageNet weights...")
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # 2. We don't need the final classification head for depth prediction
        self.vit.heads = nn.Identity()
        
        # 3. Setup hooks to extract the 4 intermediate layers
        self.extracted_features = []
        self._register_hooks()

    def _hook_fn(self, module, input, output):
        # This function runs automatically and saves the output of the layer
        self.extracted_features.append(output)

    def _register_hooks(self):
        # We attach the hook to layers 2, 5, 8, and 11 (which are the 3rd, 6th, 9th, and 12th blocks)
        target_layers = [2, 5, 8, 11]
        for i, layer in enumerate(self.vit.encoder.layers):
            if i in target_layers:
                layer.register_forward_hook(self._hook_fn)

    def forward(self, x):
        # Clear the old features
        self.extracted_features = []
        
        # Run the image through the ViT
        self.vit(x)
        
        # Return the 4 saved layers to the DPT Reassemble block!
        return self.extracted_features

# ==========================================
# How to plug it into your existing code:
# ==========================================

def initialize_production_model(device):
    # 1. Initialize the Pre-Trained Encoder
    encoder = PretrainedViTEncoder()
    
    # Freeze the ViT for the first few epochs! (Optional but highly recommended)
    # This prevents your random decoder weights from sending garbage gradients 
    # back and ruining the perfect ImageNet weights.
    for param in encoder.parameters():
        param.requires_grad = False 
        
    # 2. Plug it into your existing DPT Wrapper
    # Notice we keep embed_dim=768 and spatial_size=14 (14x14 patches)
    model = DPT(vit_encoder=encoder, embed_dim=768, features=256, spatial_size=14)
    model = model.to(device)
    
    print("Production Model Ready with Pre-trained Weights!")
    return model
# - PretrainedViTEncoder
import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from skimage import color
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Keep PretrainedViTEncoder exactly as you had it
class PretrainedViTEncoder(nn.Module):
    def __init__(self):
        super(PretrainedViTEncoder, self).__init__()
        print("Downloading pre-trained ImageNet weights...")
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity() 
        self.extracted_features = []
        self._register_hooks()

    def _hook_fn(self, module, input, output):
        self.extracted_features.append(output)

    def _register_hooks(self):
        target_layers = [2, 5, 8, 11]
        for i, layer in enumerate(self.vit.encoder.layers):
            if i in target_layers:
                layer.register_forward_hook(self._hook_fn)

    def forward(self, x):
        self.extracted_features = []
        self.vit(x) 
        return self.extracted_features
# ====================================================================

# ====================================================================
# 2. INITIALIZATION AND INFERENCE LOGIC
# ====================================================================
device = torch.device("cpu") # Web servers usually run on CPU

# Load the model structure
print("Initializing Model...")
encoder = PretrainedViTEncoder()
model = DPT(vit_encoder=encoder).to(device)

# 🔥 TOMORROW: Make sure "best_dpt_colorizer.pth" is in the same folder as this file!
try:
    model.load_state_dict(torch.load("best_dpt_colorizer.pth", map_location=device))
    print("✅ Weights loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load weights. {e}")

model.eval() # Set to evaluation mode!

def colorize_image(input_image):
    """
    Takes a PIL image from Gradio, runs it through the DPT model, 
    and returns a colorized PIL image.
    """
    if input_image is None:
        return None

    # 1. Resize the image to match our training (224x224)
    original_size = input_image.size # Save this so we can resize back later!
    img_resized = input_image.resize((224, 224)).convert("RGB")
    
    # 2. Convert to LAB and extract the L channel
    img_np = np.array(img_resized)
    lab_img = color.rgb2lab(img_np).astype(np.float32)
    
    # Extract L and normalize exactly like we did in training [-1, 1]
    l_channel = lab_img[:, :, 0]
    l_normalized = (l_channel / 50.0) - 1.0 
    
    # Convert to Tensor and duplicate 3 times
    l_tensor = torch.from_numpy(l_normalized).unsqueeze(0).unsqueeze(0) # [1, 1, 224, 224]
    l_tensor_3ch = l_tensor.repeat(1, 3, 1, 1).to(device)               # [1, 3, 224, 224]

    # 3. Predict the ab channels
    with torch.no_grad():
        ab_predicted = model(l_tensor_3ch) # Shape: [1, 2, 224, 224]
    
    # 4. Denormalize the predictions
    ab_out = ab_predicted.squeeze(0).cpu().numpy() # [2, 224, 224]
    ab_out = ab_out.transpose(1, 2, 0)             # [224, 224, 2]
    
    # Denormalize from [-1, 1] back to [-128, 127]
    ab_denormalized = ab_out * 128.0 
    
    # 5. Reconstruct the LAB image
    l_channel_original = np.expand_dims(l_channel, axis=-1) # [224, 224, 1]
    reconstructed_lab = np.concatenate((l_channel_original, ab_denormalized), axis=-1)
    
    # 6. Convert back to RGB
    reconstructed_rgb = color.lab2rgb(reconstructed_lab)
    
    # The output of lab2rgb is [0.0, 1.0]. Convert to [0, 255] for standard images
    final_img_np = (reconstructed_rgb * 255).astype(np.uint8)
    final_img_pil = Image.fromarray(final_img_np)
    
    # 7. Optional: Resize back to the user's original image size
    final_img_pil = final_img_pil.resize(original_size, Image.LANCZOS)
    
    return final_img_pil

# ====================================================================
# 3. BUILD THE WEB INTERFACE
# ====================================================================
demo = gr.Interface(
    fn=colorize_image,
    inputs=gr.Image(type="pil", label="Upload Black & White Image"),
    outputs=gr.Image(type="pil", label="Colorized Output"),
    title="DPT Image Colorization",
    description="Upload a black and white image, and our Dense Prediction Transformer will add the color back in!",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()