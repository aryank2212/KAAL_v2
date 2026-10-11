"""
fgsm.py — [legacy] Fast Gradient Sign Method against YOLOv8.

One signed-gradient step of size ``epsilon`` on a proxy loss
(``pgd.compute_proxy_loss``: "suppress" hides objects, "hallucinate" creates
false positives). Works on the raw ``torch`` model (``yolo.model``) with
images as (B, 3, H, W) tensors in [0, 1]. Not used by the live system; the
KAAL evaluations use physical-style patches from ``adversarial_patch.py``.
"""

import torch
from .pgd import compute_proxy_loss

def fgsm_attack(model, images, epsilon=8/255, target_type="suppress"):
    """
    Fast Gradient Sign Method (FGSM) Single-step attack.
    
    Args:
        model: The underlying PyTorch model (e.g., yolo.model).
        images: Clean images tensor of shape (B, 3, H, W), normalized [0, 1].
        epsilon: Maximum perturbation.
        target_type: Strategy.
    """
    original_images = images.clone().detach()
    perturbed_images = images.clone().detach()
    perturbed_images.requires_grad = True
    
    preds = model(perturbed_images)
    loss = compute_proxy_loss(preds, target_type)
    
    model.zero_grad()
    loss.backward()
    
    # Ascent step
    adv_images = perturbed_images + epsilon * perturbed_images.grad.sign()
    
    # Clip back to valid image range
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
    return adv_images
