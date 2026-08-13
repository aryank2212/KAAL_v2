"""
adversarial_patch.py — Adversarial patch generation, loading, and injection.

Supports four patch types:
  - NOISE:     Random pixel noise (universal perturbation)
  - PATTERN:   Structured high-frequency checkerboard / grid pattern
  - LOADED:    Pre-made adversarial patch loaded from an image file
  - GRADIENT:  Gradient-optimized against a detector (PGD-style, Phase 1)

Patches are alpha-blended onto objects within their bounding boxes,
preserving label integrity.

Physical-world realism (Phase 1):
  - Random perspective warp applied to the patch before placement.
  - Random lighting / exposure / gamma variation applied to the patch.
"""

import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional

from .config import PatchType

logger = logging.getLogger(__name__)


class AdversarialPatchInjector:
    """
    Generates or loads adversarial patches and applies them to images
    at object bounding-box locations.
    """

    def __init__(
        self,
        patch_type: PatchType = PatchType.NOISE,
        scale_range: Tuple[float, float] = (0.05, 0.5),
        alpha: float = 0.8,
        patch_image_path: Optional[str] = None,
        model: Optional[object] = None,
        use_perspective: bool = True,
        use_lighting: bool = True,
        grad_steps: int = 30,
        grad_lr: float = 0.1,
    ):
        """
        Args:
            patch_type:       Which generation method to use.
            scale_range:      Min/max patch size as a fraction of the bounding-box dimension.
            alpha:            Blending opacity (1.0 = fully opaque patch).
            patch_image_path: File path for PatchType.LOADED.
            model:            Detector (e.g. ultralytics YOLO) used to optimize
                              GRADIENT patches. Optional; falls back to noise.
            use_perspective:  Warp the patch with a random perspective transform.
            use_lighting:     Vary patch brightness/contrast/gamma before blending.
            grad_steps:       PGD steps for GRADIENT patches.
            grad_lr:          PGD step size for GRADIENT patches.
        """
        self.patch_type = patch_type
        self.scale_range = scale_range
        self.alpha = alpha
        self.model = model
        self.use_perspective = use_perspective
        self.use_lighting = use_lighting
        self.grad_steps = grad_steps
        self.grad_lr = grad_lr
        self.loaded_patch: Optional[np.ndarray] = None

        if patch_type == PatchType.LOADED:
            if patch_image_path is None:
                raise ValueError("patch_image_path required when patch_type is LOADED")
            self.loaded_patch = cv2.imread(patch_image_path)
            if self.loaded_patch is None:
                raise FileNotFoundError(f"Could not load patch image: {patch_image_path}")

    # ── Patch Generation ──────────────────────────────────────────────────

    def generate_patch(self, height: int, width: int) -> np.ndarray:
        """
        Create a patch of the given size based on the configured patch type.

        Args:
            height: Patch height in pixels.
            width:  Patch width in pixels.

        Returns:
            BGR uint8 numpy array of shape (height, width, 3).
        """
        if self.patch_type == PatchType.NOISE:
            patch = self._generate_noise_patch(height, width)
        elif self.patch_type == PatchType.PATTERN:
            patch = self._generate_pattern_patch(height, width)
        elif self.patch_type == PatchType.LOADED:
            patch = self._resize_loaded_patch(height, width)
        elif self.patch_type == PatchType.GRADIENT:
            patch = self._generate_gradient_patch(height, width)
        else:
            raise ValueError(f"Unknown patch type: {self.patch_type}")

        # Physical-world realism: warp + relight before the patch is placed
        if self.use_perspective:
            patch = self._apply_perspective(patch)
        if self.use_lighting:
            patch = self._apply_lighting(patch)
        return patch

    @staticmethod
    def _generate_noise_patch(h: int, w: int) -> np.ndarray:
        """Random uniform noise — acts as a universal perturbation."""
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    @staticmethod
    def _generate_pattern_patch(h: int, w: int) -> np.ndarray:
        """
        High-frequency checkerboard pattern with color variation.
        Designed to trigger high-frequency anomaly detectors.
        """
        patch = np.zeros((h, w, 3), dtype=np.uint8)
        cell = max(2, min(h, w) // 8)
        for y in range(0, h, cell):
            for x in range(0, w, cell):
                if ((y // cell) + (x // cell)) % 2 == 0:
                    color = np.random.randint(150, 256, 3).tolist()
                    patch[y : y + cell, x : x + cell] = color
                else:
                    color = np.random.randint(0, 100, 3).tolist()
                    patch[y : y + cell, x : x + cell] = color
        return patch

    def _resize_loaded_patch(self, h: int, w: int) -> np.ndarray:
        """Resize pre-loaded patch to the target dimensions."""
        if self.loaded_patch is None:
            raise ValueError(
                "No patch image loaded (patch_image_path required for PatchType.LOADED)"
            )
        return cv2.resize(self.loaded_patch, (w, h), interpolation=cv2.INTER_LINEAR)

    # ── Gradient-Based Optimization (PGD) ─────────────────────────────────

    def _generate_gradient_patch(self, h: int, w: int) -> np.ndarray:
        """
        Gradient-optimized patch (Phase 1).

        If a detector model is available, runs torch-based PGD that minimizes
        detector objectness confidence (evasion). Otherwise, runs a
        derivative-free FFT-aware optimizer that minimizes the patch's
        high-frequency energy — producing a LOW-frequency perturbation that
        is specifically designed to test the FFT defense's failure cases.
        """
        try:
            if self.model is not None:
                return self._optimize_vs_detector(h, w)
            return self._optimize_low_freq_patch(h, w)
        except Exception as exc:  # pragma: no cover - env-dependent path
            logger.warning(
                "Gradient patch optimization failed (%s); falling back to random noise patch", exc
            )
            return self._generate_noise_patch(h, w)

    def _optimize_low_freq_patch(self, h: int, w: int) -> np.ndarray:
        """
        Derivative-free, model-free optimization that pushes a patch toward
        LOW-frequency content (smooth gradients).

        Objective: minimize the patch's high-frequency FFT ratio, producing a
        perturbation that slips under the FFT dual-stream defense — the key
        failure case the ablation study must expose.

        Returns:
            Smooth, low-frequency patch.
        """
        if h < 4 or w < 4:
            return self._generate_noise_patch(h, w)

        # Start from a few random smooth blobs (inherently low-frequency)
        patch: np.ndarray = np.full((h, w, 3), 128.0, dtype=np.float32)
        for _ in range(np.random.randint(2, 5)):
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            sigma = np.random.uniform(max(2.0, min(h, w) / 8), min(h, w) / 3)
            Y, X = np.ogrid[:h, :w]
            blob = np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2)))
            color = np.random.randint(-90, 90, 3).astype(np.float32)
            patch += blob[..., None] * color

        patch = np.clip(patch, 0, 255)

        # Aggressive low-pass: blur kernel scales with patch size so HF energy
        # stays low even for large patches (the FFT failure case).
        ksize = max(7, (min(h, w) // 4) * 2 + 1)
        for _ in range(max(3, self.grad_steps // 4)):
            patch = cv2.GaussianBlur(patch, (ksize, ksize), ksize / 3)

        logger.info(
            "Low-frequency perturbation patch generated (HF ratio=%.3f)",
            self._high_freq_ratio(patch),
        )
        return patch.astype(np.uint8)

    @staticmethod
    def _high_freq_ratio(patch: np.ndarray) -> float:
        """High-frequency FFT energy ratio of a float patch."""
        gray = cv2.cvtColor(np.clip(patch, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        f_shift = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        magnitude = np.log1p(np.abs(f_shift))
        total = np.sum(magnitude)
        if total < 1e-10:
            return 0.0
        h, w = gray.shape
        radius = int(min(h, w) * 0.375)  # 25% high-freq band
        Y, X = np.ogrid[:h, :w]
        low = ((X - w // 2) ** 2 + (Y - h // 2) ** 2) <= radius**2
        return float((total - np.sum(magnitude[low])) / total)

    def _optimize_vs_detector(self, h: int, w: int) -> np.ndarray:
        """
        PGD optimization against the provided detector (requires torch).

        Plant the patch in a context canvas, forward through the detector,
        and step the patch to minimize the detector's objectness/class
        confidence. Objective: make the model NOT detect the object under
        the patch (evasion).

        Returns:
            Optimized BGR uint8 patch.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("torch required for detector-optimized patches")

        if not hasattr(self.model, "model"):
            raise RuntimeError("detector has no underlying .model module")

        canvas_size = 224
        patch = (torch.rand(1, 3, h, w, device="cpu") * 255.0).requires_grad_(True)
        opt = torch.optim.SGD([patch], lr=self.grad_lr)

        # Neutral gray context canvas, patch centered
        canvas = torch.full((1, 3, canvas_size, canvas_size), 128.0)
        ph, pw = min(h, canvas_size), min(w, canvas_size)
        y0, x0 = (canvas_size - ph) // 2, (canvas_size - pw) // 2

        for step in range(self.grad_steps):
            opt.zero_grad()
            canvas.data = torch.full((1, 3, canvas_size, canvas_size), 128.0)
            canvas[0, :, y0 : y0 + ph, x0 : x0 + pw] = patch[0, :, :ph, :pw]
            im = canvas / 255.0

            pred = self.model.model(im)[0]  # (1, num_preds, 4+1+num_classes)
            # Evasion objective: penalize high objectness
            objness = pred[..., 4:5]
            if objness.numel() == 0:
                raise RuntimeError("empty detector output")
            loss = objness.mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                patch.clamp_(0.0, 255.0)

        logger.info("Gradient patch optimized against detector over %d steps", self.grad_steps)
        return patch[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

    # ── Physical-World Simulation ─────────────────────────────────────────

    @staticmethod
    def _apply_perspective(patch: np.ndarray, jitter: float = 0.06) -> np.ndarray:
        """
        Randomly warp a patch with a perspective transform to simulate a
        sticker viewed from an angle.

        Args:
            patch:  BGR uint8 patch.
            jitter: Max corner displacement as a fraction of patch size.

        Returns:
            Perspective-warped patch (same size).
        """
        h, w = patch.shape[:2]
        jx, jy = w * jitter, h * jitter
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = np.array(
            [
                [np.random.uniform(-jx, jx), np.random.uniform(-jy, jy)],
                [w + np.random.uniform(-jx, jx), np.random.uniform(-jy, jy)],
                [w + np.random.uniform(-jx, jx), h + np.random.uniform(-jy, jy)],
                [np.random.uniform(-jx, jx), h + np.random.uniform(-jy, jy)],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(patch, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return warped

    @staticmethod
    def _apply_lighting(patch: np.ndarray) -> np.ndarray:
        """
        Random lighting/exposure variation: gamma + brightness/contrast.

        Args:
            patch: BGR uint8 patch.

        Returns:
            Relit patch.
        """
        h, w = patch.shape[:2]
        # Random per-channel exposure (prints fade differently per channel)
        mask = np.random.uniform(0.8, 1.2, (1, 1, 3)).astype(np.float32)
        relit = np.clip(patch.astype(np.float32) * mask, 0, 255).astype(np.uint8)

        # Random gamma
        gamma = np.random.uniform(0.7, 1.4)
        relit = (255.0 * np.power(relit / 255.0, gamma)).astype(np.uint8)

        # Random global brightness/contrast
        alpha = np.random.uniform(0.85, 1.15)
        beta = np.random.randint(-15, 16)
        return cv2.convertScaleAbs(relit, alpha=alpha, beta=beta)

    # ── Patch Application ─────────────────────────────────────────────────

    def apply_patch(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        prob: float = 0.5,
    ) -> Tuple[np.ndarray, bool]:
        """
        Probabilistically inject adversarial patches onto objects in the image.

        Args:
            image:  BGR uint8 image (H, W, 3).
            bboxes: List of bounding boxes as (x_center, y_center, w, h) in pixel coords.
            prob:   Probability of applying a patch to each bbox.

        Returns:
            Tuple of (modified_image, was_patched).
            Bounding box labels are NOT altered — the patch is placed *within*
            the existing bbox so annotations remain valid.
        """
        if len(bboxes) == 0:
            return image.copy(), False

        result = image.copy()
        patched = False

        for cx, cy, bw, bh in bboxes:
            if np.random.random() > prob:
                continue

            # Determine patch size as a random fraction of the bbox
            scale = np.random.uniform(*self.scale_range)
            patch_h = max(4, int(bh * scale))
            patch_w = max(4, int(bw * scale))

            # Generate the patch
            patch = self.generate_patch(patch_h, patch_w)

            # Random position WITHIN the bounding box
            x_min = int(cx - bw / 2)
            y_min = int(cy - bh / 2)
            x_max = int(cx + bw / 2) - patch_w
            y_max = int(cy + bh / 2) - patch_h

            if x_max <= x_min or y_max <= y_min:
                # Bbox too small for a patch at this scale — skip
                continue

            px = np.random.randint(x_min, x_max)
            py = np.random.randint(y_min, y_max)

            # Clip to image boundaries
            img_h, img_w = result.shape[:2]
            px = np.clip(px, 0, img_w - patch_w)
            py = np.clip(py, 0, img_h - patch_h)

            # Alpha-blend the patch onto the image
            roi = result[py : py + patch_h, px : px + patch_w]
            blended = cv2.addWeighted(patch, self.alpha, roi, 1 - self.alpha, 0)
            result[py : py + patch_h, px : px + patch_w] = blended
            patched = True

        return result, patched


def parse_yolo_labels(
    label_path: str, img_w: int, img_h: int
) -> Tuple[List[int], List[Tuple[int, int, int, int]]]:
    """
    Parse a YOLO-format label file into pixel-coordinate bounding boxes.

    YOLO format per line: class_id  x_center  y_center  width  height
    (all values normalized to [0, 1])

    Args:
        label_path: Path to the .txt label file.
        img_w:      Image width in pixels.
        img_h:      Image height in pixels.

    Returns:
        (class_ids, bboxes) where bboxes are (cx, cy, w, h) in pixel coords.
    """
    class_ids = []
    bboxes = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    cx = float(parts[1]) * img_w
                    cy = float(parts[2]) * img_h
                    w = float(parts[3]) * img_w
                    h = float(parts[4]) * img_h
                except (ValueError, TypeError):
                    continue  # malformed line — skip it
                class_ids.append(cls_id)
                bboxes.append((int(cx), int(cy), int(w), int(h)))
    except FileNotFoundError:
        pass  # No labels for this image — return empty
    return class_ids, bboxes
