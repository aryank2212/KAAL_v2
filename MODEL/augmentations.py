"""
augmentations.py — Custom adversarial augmentation functions.

Integrates the three attack types from the test/ folder:
  - clean: Random square patch overlay (adversarial patch)
  - moisy (noisy): Gaussian noise + blur degradation
  - patched: Black-box occlusion attack

Each attack can be applied independently or combined during training.
"""

import cv2
import numpy as np
from typing import Tuple


# ── Attack Type 1: Adversarial Patch (from test/clean) ────────────────────


def add_patch(image: np.ndarray) -> np.ndarray:
    """
    Place a random noise patch on the image.
    Simulates a physical adversarial sticker attack.

    Args:
        image: BGR uint8 image (H, W, 3).

    Returns:
        Image with adversarial patch applied.
    """
    result = image.copy()
    h, w, _ = result.shape

    # Random square patch size between 30-100 pixels
    patch_size = np.random.randint(30, min(100, min(h, w) - 1))
    patch = np.random.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8)

    x = np.random.randint(0, w - patch_size)
    y = np.random.randint(0, h - patch_size)

    result[y : y + patch_size, x : x + patch_size] = patch
    return result


# ── Attack Type 2: Noise + Blur (from test/patched) ──────────────────────


def add_noise(image: np.ndarray) -> np.ndarray:
    """
    Add Gaussian noise to the image.
    Simulates sensor noise or digital perturbation.

    Args:
        image: BGR uint8 image (H, W, 3).

    Returns:
        Image with additive Gaussian noise.
    """
    result = image.copy()
    noise = np.random.normal(0, 25, result.shape)
    result = np.clip(result.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return result


def blur(image: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur to the image.
    Simulates defocus or motion blur.

    Args:
        image: BGR uint8 image (H, W, 3).

    Returns:
        Blurred image.
    """
    return cv2.GaussianBlur(image, (7, 7), 0)


# ── Attack Type 3: Occlusion (from test/moisy) ──────────────────────────


def occlude(image: np.ndarray) -> np.ndarray:
    """
    Apply a black rectangular occlusion to the image.
    Simulates physical occlusion or censoring.

    Args:
        image: BGR uint8 image (H, W, 3).

    Returns:
        Image with black rectangle occlusion.
    """
    result = image.copy()
    h, w, _ = result.shape
    x1, y1 = np.random.randint(0, w // 2), np.random.randint(0, h // 2)
    x2, y2 = min(x1 + 100, w), min(y1 + 100, h)
    result[y1:y2, x1:x2] = 0
    return result


# ── Attack Type 4: Low-Frequency Perturbation (Phase 1) ─────────────────


def add_low_freq_noise(image: np.ndarray) -> np.ndarray:
    """
    Add a smooth, LOW-frequency perturbation to the image.

    The perturbation is a heavily-blurred random field, so its spectral
    energy stays in the low-frequency band. This is specifically designed
    to test the FFT defense's failure case: it changes pixel content
    without triggering the high-frequency anomaly detector.

    Args:
        image: BGR uint8 image (H, W, 3).

    Returns:
        Image with additive low-frequency perturbation.
    """
    result = image.copy()
    h, w, _ = result.shape

    # Random smooth field: coarse random values upscaled + blurred
    coarse_h = max(2, min(h, 32))
    coarse_w = max(2, min(w, 32))
    field: np.ndarray = np.random.uniform(-120, 120, (coarse_h, coarse_w, 3))
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(h, w) / 20)

    result = np.clip(result.astype(np.float32) + field, 0, 255).astype(np.uint8)
    return result


# ── Combined Attack Pipeline ─────────────────────────────────────────────


def apply_random_attack(
    image: np.ndarray,
    attack_types: str = "all",
) -> Tuple[np.ndarray, str]:
    """
    Apply a randomly selected attack to the image.

    Args:
        image:        BGR uint8 image.
        attack_types: Which attacks to use:
                      "all"     — randomly pick from all four
                      "patch"   — adversarial patch only
                      "noise"   — noise + blur only
                      "occlude" — occlusion only
                      "lowfreq" — low-frequency perturbation only

    Returns:
        (attacked_image, attack_name)
    """
    attacks = {
        "patch": ("patch", lambda img: add_patch(img)),
        "noise": ("noise+blur", lambda img: blur(add_noise(img))),
        "occlude": ("occlusion", lambda img: occlude(img)),
        "lowfreq": ("low-freq", lambda img: add_low_freq_noise(img)),
    }

    if attack_types == "all":
        key = np.random.choice(list(attacks.keys()))
    elif attack_types in attacks:
        key = attack_types
    else:
        key = np.random.choice(list(attacks.keys()))

    name, fn = attacks[key]
    return fn(image), name


def apply_all_attacks(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Apply all three attacks sequentially (worst-case scenario).

    Returns:
        (attacked_image, "combined")
    """
    result = add_patch(image)
    result = add_noise(result)
    result = occlude(result)
    return result, "combined"
