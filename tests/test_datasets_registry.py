"""
test_datasets_registry.py — The dataset registry stays complete and consistent.

``MODEL/datasets_registry.yaml`` is the record of which data trained which
model. These checks fail when a dataset builder gains a source that the
registry does not describe, or when an entry is malformed.
"""

from pathlib import Path

import yaml

from MODEL.build_dataset import LICENSES

REGISTRY = Path(__file__).resolve().parent.parent / "MODEL" / "datasets_registry.yaml"
STATUSES = {"on_disk", "trashed", "not_on_disk", "rejected", "pending"}
ROLES = {"training_source", "test_only", "built", "historical", "candidate"}
MODEL_STATUSES = {"production", "baseline", "superseded", "pending"}


def _load():
    return yaml.safe_load(REGISTRY.read_text())


def test_schema_and_unique_ids():
    reg = _load()
    assert reg["schema_version"] == 1
    ids = [e["id"] for section in ("sources", "built", "models") for e in reg[section]]
    assert len(ids) == len(set(ids)), "registry ids must be unique"
    for src in reg["sources"]:
        for key in (
            "name",
            "builder_source",
            "local_path",
            "upstream",
            "license",
            "status",
            "role",
        ):
            assert key in src, f"{src['id']} lacks {key}"
        assert src["status"] in STATUSES, src["id"]
        assert src["role"] in ROLES, src["id"]
        if src["status"] == "on_disk":
            assert src["local_path"], f"{src['id']} is on disk but has no local_path"
    for built in reg["built"]:
        assert built["status"] in STATUSES, built["id"]


def test_every_builder_source_is_registered():
    reg = _load()
    registered = {s["builder_source"] for s in reg["sources"] if s["builder_source"]}
    assert set(LICENSES) == registered


def test_built_datasets_and_models_reference_known_ids():
    reg = _load()
    source_ids = {s["id"] for s in reg["sources"]}
    built_ids = {b["id"] for b in reg["built"]}
    for built in reg["built"]:
        assert set(built["sources"]) <= source_ids, built["id"]
    for model in reg["models"]:
        assert model["trained_on"] in built_ids, model["id"]
        assert model["status"] in MODEL_STATUSES, model["id"]
    assert sum(m["status"] == "production" for m in reg["models"]) == 1


def test_test_only_sources_never_feed_a_training_set():
    reg = _load()
    test_only = {s["id"] for s in reg["sources"] if s["role"] == "test_only"}
    for built in reg["built"]:
        if built["id"] != "kaal_ground_test":
            assert not test_only & set(built["sources"]), built["id"]
