"""
test_train_detector.py — The CLI's schedule and warm-start options reach RF-DETR.
"""

import sys
import types
from argparse import Namespace

import pytest

from MODEL.train_detector import train_rfdetr


class _FakeRFDETRSmall:
    last = None

    def __init__(self, pretrain_weights=None):
        self.pretrain_weights = pretrain_weights
        self.train_kwargs = None
        _FakeRFDETRSmall.last = self

    def train(self, **kwargs):
        self.train_kwargs = kwargs


@pytest.fixture
def fake_rfdetr(monkeypatch):
    monkeypatch.setitem(sys.modules, "rfdetr", types.SimpleNamespace(RFDETRSmall=_FakeRFDETRSmall))
    return _FakeRFDETRSmall


def _args(**over):
    base = dict(
        arch="rfdetr-small",
        init_weights=None,
        data="/data/kaal_v4",
        epochs=12,
        batch=4,
        grad_accum=4,
        lr=5e-5,
        workers=2,
        patience=5,
        lr_scheduler="step",
        lr_min_factor=0.05,
        imgsz=None,
    )
    base.update(over)
    return Namespace(**base)


def test_cosine_schedule_and_floor_are_passed(fake_rfdetr, tmp_path):
    train_rfdetr(_args(lr_scheduler="cosine", lr_min_factor=0.05), tmp_path)
    kw = fake_rfdetr.last.train_kwargs
    assert kw["lr_scheduler"] == "cosine"
    assert kw["lr_scheduler_kwargs"] == {"min_factor": 0.05}
    assert kw["lr"] == 5e-5 and kw["epochs"] == 12


def test_step_schedule_keeps_rfdetr_defaults(fake_rfdetr, tmp_path):
    train_rfdetr(_args(), tmp_path)
    kw = fake_rfdetr.last.train_kwargs
    assert kw["lr_scheduler"] == "step"
    assert "lr_scheduler_kwargs" not in kw


def test_init_weights_warm_start(fake_rfdetr, tmp_path):
    out = train_rfdetr(_args(init_weights="/w/v4.pth"), tmp_path)
    assert fake_rfdetr.last.pretrain_weights == "/w/v4.pth"
    assert out["weights"].endswith("checkpoint_best_total.pth")
