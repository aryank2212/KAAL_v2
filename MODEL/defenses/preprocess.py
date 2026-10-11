"""
preprocess.py — [legacy] Input-transform defenses on image tensors.

JPEG compression, bit-depth reduction and Gaussian blur on (B, 3, H, W)
tensors in [0, 1], chained by ``defense_pipeline``. The Phase 3 benchmark
(``MODEL/benchmark.py``) re-implements these in NumPy and measured that
bit-depth / feature squeezing push the FFT cue's clean false-positive rate to
100% and blur hides real attacks (``MODEL/RESULTS.md`` §2–3).
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import io

def jpeg_compression(image_tensor, quality=75):
    """
    Applies JPEG compression to a batch of image tensors.
    
    Args:
        image_tensor: Tensor of shape (B, 3, H, W) normalized to [0, 1].
        quality: JPEG quality (1-100).
        
    Returns:
        Tensor of the same shape, differentiable? No, JPG is non-differentiable.
        Since defenses are typically applied at inference, non-differentiable is fine.
    """
    # Convert tensor to uint8 list of images
    device = image_tensor.device
    batch_size = image_tensor.size(0)
    
    images_uint8 = (image_tensor.cpu().detach() * 255).clamp(0, 255).to(torch.uint8)
    defended_images = []
    
    for i in range(batch_size):
        img_np = images_uint8[i].permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(img_np)
        
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        
        compressed_pil = Image.open(buffer)
        compressed_tensor = TF.to_tensor(compressed_pil) # Returns [0, 1] tensor
        defended_images.append(compressed_tensor)
        
    return torch.stack(defended_images).to(device)


def bit_depth_reduction(image_tensor, bits=5):
    """
    Reduces the bit depth of an image to destroy small adversarial perturbations.
    
    Args:
        image_tensor: Tensor of shape (B, 3, H, W) normalized [0, 1].
        bits: Number of bits to keep per channel (1-8).
    """
    steps = 2 ** bits
    # Quantize and dequantize
    quantized = torch.round(image_tensor * (steps - 1)) / (steps - 1)
    return quantized


def gaussian_blur_defense(image_tensor, kernel_size=3, sigma=1.0):
    """
    Applies Gaussian Blur.
    
    Args:
        image_tensor: Input tensor shape (B, 3, H, W).
    """
    # torchvision.transforms.functional can do it, but requires batch support in recent pyTorch
    # TF.gaussian_blur handles batches in PyTorch >= 1.7
    return TF.gaussian_blur(image_tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


def defense_pipeline(image_tensor, use_jpeg=True, jpeg_quality=75, 
                     use_bit_depth=True, bit_depth=5, 
                     use_blur=True, blur_kernel=3, blur_sigma=1.0):
    """
    Multi-layered defense pipeline.
    Applies combinations of image transformations to neutralize adversarial perturbations.
    
    Args:
        image_tensor: (B, 3, H, W) float tensor [0, 1].
    """
    defended = image_tensor
    
    if use_bit_depth:
        defended = bit_depth_reduction(defended, bits=bit_depth)
        
    if use_blur:
        defended = gaussian_blur_defense(defended, kernel_size=blur_kernel, sigma=blur_sigma)
        
    if use_jpeg:
        defended = jpeg_compression(defended, quality=jpeg_quality)
        
    return defended
