"""The crop and full-body routes must preprocess images the way their checkpoints were trained.

Both routes build the classifier as plain ``ViT-B-16-SigLIP`` with ``pretrained=None`` so the
shared encoder's weights are not loaded only to be overwritten. That name carries no preprocess
config, and open_clip then falls back to OpenAI-CLIP normalisation and a shortest-side resize with
a centre crop. The checkpoints were trained with mean/std 0.5 and ``squash``. Before the fix the
released full-body route reproduced its own published predictions on 0 of 64 held-out images and
the crop route on 16 of 64: the centre crop cut heads and feet off full-body photographs.

These tests need no model weights. The first checks the transform's behaviour on a synthetic image;
the second fails if any route builds the classifier without applying it.
"""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "moda_fashion_distilled_open_clip_config.json"


def _fake_package(tmp_path: Path) -> SimpleNamespace:
    base = tmp_path / "base"
    base.mkdir()
    (base / "open_clip_config.json").write_text(FIXTURE.read_text())
    return SimpleNamespace(root=tmp_path)


def test_eval_transform_squashes_and_uses_the_encoders_normalisation(tmp_path):
    pytest.importorskip("open_clip")
    from PIL import Image

    from suite._model.routes import _encoder_eval_transform

    # A tall garment-shaped image: a white band across the top quarter, black below.
    image = Image.new("RGB", (100, 400), (0, 0, 0))
    image.paste((255, 255, 255), (0, 0, 100, 100))

    pixels = _encoder_eval_transform(_fake_package(tmp_path))(image)

    assert tuple(pixels.shape) == (3, 224, 224)
    # mean/std 0.5 maps white to +1 and black to -1. OpenAI-CLIP statistics would give about
    # +1.93 and -1.79 instead.
    assert pixels.max().item() == pytest.approx(1.0, abs=1e-3)
    assert pixels.min().item() == pytest.approx(-1.0, abs=1e-3)
    # Squashing keeps the top band, in the top quarter of the output. A centre crop of a 1:4 image
    # keeps only the middle, so the band would vanish entirely.
    assert pixels[:, :40, :].mean().item() == pytest.approx(1.0, abs=1e-2)
    assert pixels[:, 80:, :].mean().item() == pytest.approx(-1.0, abs=1e-2)


def test_every_route_that_builds_the_classifier_applies_the_training_transform():
    from suite._model import routes

    builders = [
        cls
        for _, cls in inspect.getmembers(routes, inspect.isclass)
        if cls.__module__ == routes.__name__
        and "FashionSiglipAttributeClassifier(" in inspect.getsource(cls.__init__)
    ]
    assert {cls.__name__ for cls in builders} >= {"_FashionpediaBackend", "_DfmmBackend"}
    for cls in builders:
        assert "_encoder_eval_transform(package)" in inspect.getsource(cls.__init__), (
            f"{cls.__name__} builds the classifier without the training-time eval transform"
        )


def test_fixture_matches_the_encoder_the_routes_were_trained_on():
    config = json.loads(FIXTURE.read_text())
    assert config["preprocess_cfg"] == {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "interpolation": "bicubic",
        "resize_mode": "squash",
    }
