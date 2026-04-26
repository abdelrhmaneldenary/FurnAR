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
