"""
test_temporal_scorer.py — Unit tests for MODEL/temporal_scorer.py (Phase 3/4).
"""

import cv2
import numpy as np
import pytest

from MODEL.temporal_scorer import TemporalConsistencyDetector
from MODEL.failure_analysis import _smooth_gradient


def _clean_frames(n=10, seed=0):
    rng = np.random.RandomState(seed)
    frames = []
    base = _smooth_gradient(rng).astype(np.float32)
    for i in range(n):
        frame = np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def _smooth_offset(shape, seed=0, sigma=9, amp=20):
    """A smooth, low-frequency perturbation (the FFT-blind attack)."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, amp, shape[:2]).astype(np.float32)
    noise = np.stack([noise] * 3, axis=-1)
    return cv2.GaussianBlur(noise, (sigma, sigma), 0).astype(np.float32)


class TestScoring:
    def test_scores_bounded(self):
        det = TemporalConsistencyDetector()
        frames = _clean_frames(5)
        det.calibrate(frames)
        s = det.score(frames[0])
        assert 0.0 <= s <= 1.0

    def test_not_initialized_returns_zero(self):
        det = TemporalConsistencyDetector()
        assert det.score(_clean_frames(1)[0]) == 0.0
        assert not det.initialized

    def test_clean_vs_attack_residual_gap(self):
        """A persistent low-freq perturbation must lift the residual above clean."""
        clean = _clean_frames(12)
        det = TemporalConsistencyDetector()
        det.calibrate(clean)
        clean_scores = [det.score_and_update(f) for f in clean]

        base = clean[-1].astype(np.float32)
        offset = _smooth_offset(base.shape)
        attacked = [
            np.clip(
                base + offset + np.random.RandomState(i).normal(0, 2, base.shape), 0, 255
            ).astype(np.uint8)
            for i in range(6)
        ]
        attack_scores = [det.score_and_update(f) for f in attacked]
        assert np.mean(attack_scores) > np.mean(clean_scores)


class TestCalibration:
    def test_threshold_sane(self):
        det = TemporalConsistencyDetector()
        frames = _clean_frames(8)
        t = det.calibrate(frames)
        assert 0.0 < t < 1.0
        assert det.initialized

    def test_clean_fpr_low(self):
        frames = _clean_frames(20)
        det = TemporalConsistencyDetector()
        t = det.calibrate(frames)
        rate = det.detection_rate(frames, t)
        assert rate < 0.1

    def test_detects_low_freq_attack(self):
        """Low-frequency attacks are exactly what the FFT misses — the temporal cue should catch them."""
        clean = _clean_frames(20)
        det = TemporalConsistencyDetector()
        t = det.calibrate(clean)

        base = clean[-1].astype(np.float32)
        offset = _smooth_offset(base.shape, seed=1)
        attacked = [
            np.clip(
                base + offset + np.random.RandomState(i).normal(0, 2, base.shape), 0, 255
            ).astype(np.uint8)
            for i in range(10)
        ]
        rate = det.detection_rate(attacked, t)
        assert rate >= 0.5


class TestValidation:
    def test_bad_alpha_rejected(self):
        with pytest.raises(ValueError):
            TemporalConsistencyDetector(alpha=1.0)

    def test_bad_grid_rejected(self):
        with pytest.raises(ValueError):
            TemporalConsistencyDetector(grid=2)

    def test_reset_forgets_background(self):
        det = TemporalConsistencyDetector()
        det.calibrate(_clean_frames(4))
        assert det.initialized
        det.reset()
        assert not det.initialized
        assert det.score(_clean_frames(1)[0]) == 0.0


# ── StreamingTemporalCue (live-video fusion) ─────────────────────────────

from MODEL.temporal_scorer import StreamingTemporalCue  # noqa: E402
from MODEL.augmentations import add_low_freq_noise  # noqa: E402


def _scene(rng, base):
    """Static scene + per-frame sensor noise."""
    noise = rng.normal(0, 2.0, base.shape)
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _base_scene(seed=0):
    rng = np.random.RandomState(seed)
    return cv2.GaussianBlur(rng.randint(0, 255, (120, 160, 3), dtype=np.uint8), (21, 21), 0)


class TestStreamingTemporalCue:
    def test_warmup_calibrates_then_stays_quiet_on_clean_stream(self):
        rng = np.random.RandomState(1)
        base = _base_scene()
        cue = StreamingTemporalCue(warmup_frames=20)
        results = [cue.observe(_scene(rng, base)) for _ in range(80)]
        assert not results[0]["ready"]
        assert cue.ready and cue.threshold is not None
        assert sum(r["flagged"] for r in results) == 0

    def test_flags_persistent_low_freq_perturbation(self):
        """Gated updates keep a persistent attack visible (not absorbed by the EMA)."""
        rng = np.random.RandomState(2)
        base = _base_scene()
        cue = StreamingTemporalCue(warmup_frames=20, max_flagged_frames=1000)
        for _ in range(40):
            cue.observe(_scene(rng, base))
        np.random.seed(5)
        attacked = add_low_freq_noise(base)
        flags = [cue.observe(_scene(rng, attacked))["flagged"] for _ in range(60)]
        assert all(flags[5:]), "persistent perturbation must stay flagged"

    def test_rebaselines_after_long_flag_run(self):
        rng = np.random.RandomState(3)
        cue = StreamingTemporalCue(warmup_frames=10, max_flagged_frames=15)
        for _ in range(20):
            cue.observe(_scene(rng, _base_scene(0)))
        new_scene = _base_scene(9)  # camera moved: a genuinely different scene
        for _ in range(16):
            cue.observe(_scene(rng, new_scene))
        assert cue.rebaselines == 1
        assert not cue.ready  # re-calibrating on the new scene

    def test_fixed_threshold_skips_calibration(self):
        rng = np.random.RandomState(4)
        cue = StreamingTemporalCue(threshold=0.5)
        cue.observe(_scene(rng, _base_scene()))
        assert cue.ready and cue.threshold == 0.5

    def test_rejects_tiny_warmup(self):
        with pytest.raises(ValueError):
            StreamingTemporalCue(warmup_frames=1)


class TestStreamingTemporalLocalization:
    """The live cue must catch a small fixed patch, not moving objects or lighting."""

    def _warm(self, cue, rng, base, n=25):
        for _ in range(n):
            cue.observe(_scene(rng, base))
        assert cue.ready

    def test_small_static_patch_is_flagged(self):
        rng = np.random.RandomState(10)
        base = cv2.resize(_base_scene(), (640, 480))
        cue = StreamingTemporalCue(warmup_frames=20)
        self._warm(cue, rng, base)
        patched = base.copy()
        patched[200:240, 300:340] = np.random.RandomState(0).randint(
            0, 256, (40, 40, 3)
        )  # 0.5% of frame
        flags = [cue.observe(_scene(rng, patched))["flagged"] for _ in range(20)]
        assert not any(flags[:3]) and all(flags[5:])

    def test_moving_object_is_not_flagged(self):
        rng = np.random.RandomState(11)
        base = cv2.resize(_base_scene(), (640, 480))
        cue = StreamingTemporalCue(warmup_frames=20)
        self._warm(cue, rng, base)
        flags = []
        for t in range(40):
            frame = base.copy()
            x = 20 + t * 14  # a 30 px "car" crossing the frame
            frame[300:330, x : x + 30] = 30
            flags.append(cue.observe(_scene(rng, frame))["flagged"])
        assert not any(flags)

    def test_global_brightness_change_is_not_flagged(self):
        rng = np.random.RandomState(12)
        base = cv2.resize(_base_scene(), (640, 480))
        cue = StreamingTemporalCue(warmup_frames=20)
        self._warm(cue, rng, base)
        brighter = np.clip(base.astype(np.int16) + 25, 0, 255).astype(np.uint8)
        flags = [cue.observe(_scene(rng, brighter))["flagged"] for _ in range(20)]
        assert not any(flags)

    def test_long_lived_local_change_is_absorbed(self):
        rng = np.random.RandomState(13)
        base = cv2.resize(_base_scene(), (640, 480))
        cue = StreamingTemporalCue(warmup_frames=20, max_flagged_frames=12)
        self._warm(cue, rng, base)
        parked = base.copy()
        parked[100:160, 100:180] = 20
        flags = [cue.observe(_scene(rng, parked))["flagged"] for _ in range(30)]
        assert any(flags[5:12]) and not any(flags[15:])
        assert cue.rebaselines >= 1 and cue.ready  # local absorb, not a full reset
