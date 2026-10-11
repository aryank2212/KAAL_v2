"""
pgd.py — [legacy] Projected Gradient Descent against YOLOv8 with a proxy loss.

``compute_proxy_loss`` stands in for Ultralytics' target assignment: it
pushes the max class confidence down ("suppress") or up ("hallucinate").
``pgd_attack`` iterates signed steps inside an L-inf ball of ``epsilon``.
Used by ``MODEL/training/adv_train.py`` and ``MODEL/evaluation/evaluate.py``
(both legacy). Not on the live path.
"""

import torch

def compute_proxy_loss(preds, target_type="suppress"):
    """
    Computes a proxy loss for YOLOv8 since directly matching Ultralytics' complex 
    target assignment without the full training loop is difficult.
    
    Args:
        preds: Raw predictions from YOLO module (tuple or tensor).
               Shape typically (B, 4 + num_classes, anchors)
        target_type: "suppress" (hide objects) or "hallucinate" (create false positives).
    """
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
        
    class_preds = preds[:, 4:, :] 
    max_cls_conf, _ = torch.max(class_preds, dim=1) 
    
    if target_type == "suppress":
        # PGD maximizes loss. Maximizing negative confidence minimizes confidence.
        # This causes False Negatives (hides objects).
        loss = -max_cls_conf.sum()
    else:
        # Maximizing positive confidence forces False Positives (hallucinations).
        loss = max_cls_conf.sum()
        
    return loss

def pgd_attack(model, images, epsilon=8/255, alpha=2/255, iters=10, target_type="suppress"):
    """
    Projected Gradient Descent (PGD) Multi-step attack.
    
    Args:
        model: The underlying PyTorch model (e.g., yolo.model).
        images: Clean images tensor of shape (B, 3, H, W), normalized [0, 1].
        epsilon: Maximum perturbation.
        alpha: Step size.
        iters: Number of attack iterations.
        target_type: Strategy.
    """
    original_images = images.clone().detach()
    perturbed_images = images.clone().detach()
    
    # Initialize with random noise uniformly inside the epsilon ball
    perturbed_images = perturbed_images + torch.empty_like(perturbed_images).uniform_(-epsilon, epsilon)
    perturbed_images = torch.clamp(perturbed_images, 0, 1).detach()
    
    for i in range(iters):
        perturbed_images.requires_grad = True
        
        preds = model(perturbed_images)
        loss = compute_proxy_loss(preds, target_type)
        
        model.zero_grad()
        loss.backward()
        
        # Ascent step
        adv_images = perturbed_images + alpha * perturbed_images.grad.sign()
        
        # Projection step
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        perturbed_images = torch.clamp(original_images + eta, min=0, max=1).detach()
        
    return perturbed_images
