"""
test_build_ground_test.py — Label mapping of the independent ground-vehicle test set.
"""

import cv2
import numpy as np

from MODEL.build_dataset import CLS
from MODEL.build_ground_test import (
    IGNORE,
    orion_web_images,
    russian_images,
    search2_images,
)


def _img(path, h=100, w=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), np.uint8))


def test_orion_tank_models_ifvs_and_lav(tmp_path):
    data = tmp_path / "dataset" / "data"
    for folder, label in (
        ("AFV/T-72", "0 0.5 0.5 0.2 0.2\n"),  # tank model -> tank
        ("AFV/AFV Lynx", "0 0.5 0.5 0.2 0.2\n"),  # IFV -> armored_vehicle
        ("MEV/m728 cev", "2 0.5 0.5 0.2 0.2\n3 0.1 0.1 0.1 0.1\n"),  # engineering + LAV
    ):
        _img(data / folder / "a.JPEG")
        (data / folder / "a.txt").write_text(label)
    by_folder = {t.image.parent.name: [b[0] for b in t.boxes] for t in orion_web_images(tmp_path)}
    assert by_folder == {
        "T-72": [CLS["tank"]],
        "AFV Lynx": [CLS["armored_vehicle"]],
        "m728 cev": [CLS["engineering_vehicle"], IGNORE],
    }


def test_russian_collapses_roboflow_copies_and_maps_models(tmp_path):
    xml = (
        "<annotation><object><name>{}</name><bndbox><xmin>1</xmin><ymin>2</ymin>"
        "<xmax>30</xmax><ymax>40</ymax></bndbox></object></annotation>"
    )
    for stem, model in (
        ("7_jpg.rf.aaa", "bm-21"),
        ("7_jpg.rf.bbb", "bm-21"),
        ("8_jpg.rf.ccc", "t-72"),
    ):
        _img(tmp_path / "train" / f"{stem}.jpg")
        (tmp_path / "train" / f"{stem}.xml").write_text(xml.format(model))
    images = russian_images(tmp_path)
    assert sorted((t.name, t.boxes[0][0]) for t in images) == [
        ("7", CLS["artillery"]),
        ("8", CLS["tank"]),
    ]


def test_search2_crop_contains_target(tmp_path):
    base = tmp_path / "search_2"
    base.mkdir()
    (base / "meta.csv").write_text(
        "Image;Target;Distance;X;Y;W;H\n1;T72;3000;6100;30;40;20\n2;HVS;2000;3000;2000;50;30\n"
    )
    tank, truck = search2_images(tmp_path, crop=1024)
    for t in (tank, truck):
        _, x1, y1, x2, y2 = t.boxes[0]
        assert 0 <= x1 < x2 <= 1024 and 0 <= y1 < y2 <= 1024
        cx1, cy1, cx2, cy2 = t.crop
        assert cx2 - cx1 == 1024 and cy2 - cy1 == 1024 and cx2 <= 6144 and cy2 <= 4096
    assert tank.boxes[0][0] == CLS["tank"] and truck.boxes[0][0] == IGNORE
