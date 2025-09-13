# utils/mask_utils.py
import torch
import torch.nn.functional as F
import math

def _sobel_energy(x):
    # x: (B,3,H,W), normalized. Return per-pixel energy map (B,1,H,W)
    gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    gray = x.mean(1, keepdim=True)
    ex = F.conv2d(gray, gx, padding=1)
    ey = F.conv2d(gray, gy, padding=1)
    return (ex**2 + ey**2).sqrt()

def mask_low_saliency_tokens(images, patch_size=32, keep_ratio=0.5):
    """
    images: (B,3,H,W) -> returns a boolean mask per patch: (B, num_patches)
    keep_ratio: fraction of patches to keep (e.g., 0.5 keeps top 50% by saliency)
    """
    B, _, H, W = images.shape
    sal = _sobel_energy(images)  # (B,1,H,W)
    # average saliency inside non-overlapping patches
    sal_pool = F.avg_pool2d(sal, kernel_size=patch_size, stride=patch_size)  # (B,1,H/ps,W/ps)
    sal_flat = sal_pool.view(B, -1)  # (B, num_patches)
    k = (sal_flat.size(1) * keep_ratio)
    k = int(math.ceil(k))
    # keep top-k
    topk = torch.topk(sal_flat, k=k, dim=1).indices  # (B,k)
    mask = torch.zeros_like(sal_flat, dtype=torch.bool)
    mask.scatter_(1, topk, True)  # True for kept patches
    return mask  # (B, num_patches)
