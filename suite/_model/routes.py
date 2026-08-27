"""Per-route inference for the published MODA_NER(V) weights.

One class per input contract. There is deliberately no dispatcher here: if you
downloaded a route you already know which one you have, and the layer that picks
a route for you is part of the hosted product rather than this release.

Needs torch, open_clip, safetensors, joblib and numpy.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .package import ModaGeneralPackage


class LocalRoute:
    """Point a backend at a plain downloaded model directory.

    The backends were written against a multi-route package layout. A published
    route is a single flat folder, so this adapts one to the other.
    """

    def __init__(self, model_dir: str | Path, schema_dirname: str) -> None:
        self.root = Path(model_dir).resolve().parent
        self._schema_dirname = schema_dirname
        target = self.root / "schemas" / schema_dirname
        if not target.exists():
            self.root = Path(model_dir).resolve()

    def __truediv__(self, other):  # pragma: no cover - path shim
        return self.root / other

CropBackend = None  # rebound below


def _resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_rgb(image: Any) -> Any:
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    if isinstance(image, bytes | bytearray):
        with Image.open(io.BytesIO(bytes(image))) as opened:
            return opened.convert("RGB").copy()
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        with Image.open(path) as opened:
            return opened.convert("RGB").copy()
    raise TypeError("image must be a PIL image, bytes, or a local filesystem path")


def _base_model(package: ModaGeneralPackage, device: str) -> tuple[Any, Any]:
    import open_clip
    import torch
    from safetensors.torch import load_file

    config = json.loads((package.root / "base/open_clip_config.json").read_text())
    preprocess = config["preprocess_cfg"]
    model, _, transform = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP",
        pretrained=None,
        image_mean=tuple(preprocess["mean"]),
        image_std=tuple(preprocess["std"]),
        image_interpolation=str(preprocess["interpolation"]),
        image_resize_mode=str(preprocess["resize_mode"]),
    )
    model.load_state_dict(
        load_file(package.root / "base/open_clip_model.safetensors"), strict=True
    )
    model.requires_grad_(False).eval().to(torch.device(device))
    return model, transform


def _predict_dfmm_head(head: Any, features: Any) -> Any:
    """Decode a packaged direct or applicability-conditional DFMM head."""
    import numpy as np

    if isinstance(head, Mapping):
        if set(head) != {"applicability", "value", "threshold"}:
            raise ValueError("Malformed packaged conditional DFMM head")
        threshold = float(head["threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Conditional DFMM threshold must be in [0, 1]")
        applicability = head["applicability"]
        classes = np.asarray(getattr(applicability, "classes_", ()))
        visible_indices = np.flatnonzero(classes == 1)
        if visible_indices.size != 1:
            raise ValueError("Conditional DFMM head has no unique visible class")
        probabilities = np.asarray(applicability.predict_proba(features))
        values = np.asarray(head["value"].predict(features))
        if probabilities.ndim != 2 or probabilities.shape[0] != values.shape[0]:
            raise ValueError("Conditional DFMM head returned incompatible shapes")
        visible = probabilities[:, int(visible_indices[0])]
        return np.where(visible >= threshold, values, "NA")
    predict = getattr(head, "predict", None)
    if not callable(predict):
        raise TypeError("Packaged direct DFMM head must expose predict(features)")
    return np.asarray(predict(features))


class _FashionpediaBackend:
    def __init__(self, package: ModaGeneralPackage, device: str) -> None:
        import torch
        from safetensors.torch import load_file

        from .contract import AttributeVocabulary
        from .architecture import FashionSiglipAttributeClassifier

        self.device = torch.device(device)
        model_dir = package.root / "schemas/fashionpedia_moda15"
        self.vocabulary = AttributeVocabulary.from_dict(
            json.loads((model_dir / "vocabulary.json").read_text())
        )
        self.thresholds = json.loads((model_dir / "thresholds.json").read_text())
        metrics = json.loads((model_dir / "metrics.json").read_text())
        self.model = FashionSiglipAttributeClassifier(
            self.vocabulary,
            model_name="ViT-B-16-SigLIP",
            # The resolved route checkpoint contains the complete encoder. Loading the shared
            # base first would read another ~800 MB only to overwrite every tensor below.
            pretrained=None,
            use_spatial_tokens=bool(metrics.get("use_spatial_tokens", False)),
            spatial_residual=bool(metrics.get("spatial_residual", False)),
        )
        self.model.load_state_dict(load_file(model_dir / "model.safetensors"), strict=True)
        self.model.requires_grad_(False).eval().to(self.device)

    def predict(self, image: Any) -> dict[str, Any]:
        import torch

        from .calibration import decode_probabilities

        pixels = self.model.preprocess_val(_load_rgb(image)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            outputs = self.model(pixels)
        categories = {
            field: torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()
            for field, logits in outputs["categories"].items()
        }
        applicability = {
            field: float(torch.sigmoid(logits.float())[0].cpu())
            for field, logits in outputs["applicability"].items()
        }
        multi = set(self.vocabulary.multi_label_fields)
        values = {
            field: (
                torch.sigmoid(logits.float())
                if field in multi
                else torch.softmax(logits.float(), dim=-1)
            )[0]
            .cpu()
            .tolist()
            for field, logits in outputs["values"].items()
        }
        return decode_probabilities(
            categories,
            applicability,
            values,
            self.vocabulary,
            self.thresholds,
        )


class _Shopping100kBackend:
    def __init__(self, package: ModaGeneralPackage, device: str) -> None:
        import torch
        from safetensors.torch import load_file

        self.device = torch.device(device)
        self.model, self.transform = _base_model(package, device)
        schema = json.loads(
            (package.root / "schemas/shopping100k_10/schema.json").read_text()
        )
        self.fields = {
            field: {
                "source_head": str(specification["source_head"]),
                "values": tuple(str(value) for value in specification["values"]),
            }
            for field, specification in schema["fields"].items()
        }
        self.heads = {
            name: tensor.to(self.device)
            for name, tensor in load_file(
                package.root / "schemas/shopping100k_10/linear_heads.safetensors"
            ).items()
        }

    def predict(self, image: Any) -> dict[str, str]:
        import torch

        pixels = self.transform(_load_rgb(image)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(pixels, normalize=True).float()
        result: dict[str, str] = {}
        for field, specification in self.fields.items():
            head = specification["source_head"]
            weight = self.heads[f"{head}.weight"]
            bias = self.heads[f"{head}.bias"]
            index = int(torch.nn.functional.linear(features, weight, bias).argmax(dim=1)[0])
            result[field] = specification["values"][index]
        return result


class _DfmmBackend:
    def __init__(self, package: ModaGeneralPackage, device: str) -> None:
        import joblib
        import torch
        from safetensors.torch import load_file

        from .contract import AttributeVocabulary
        from .architecture import FashionSiglipAttributeClassifier

        self.device = torch.device(device)
        root = package.root / "schemas/dfmm_18"
        model_root = root / "encoder"
        vocabulary = AttributeVocabulary.from_dict(
            json.loads((model_root / "vocabulary.json").read_text())
        )
        metrics = json.loads((model_root / "metrics.json").read_text())
        self.model = FashionSiglipAttributeClassifier(
            vocabulary,
            model_name="ViT-B-16-SigLIP",
            # The schema-adapted checkpoint is a full state dict, not a delta.
            pretrained=None,
            train_vision=False,
            use_spatial_tokens=bool(metrics["use_spatial_tokens"]),
            spatial_residual=bool(metrics["spatial_residual"]),
        )
        self.model.load_state_dict(
            load_file(model_root / "model.safetensors"), strict=True
        )
        self.model.requires_grad_(False).eval().to(self.device)
        schema = json.loads((root / "schema.json").read_text())
        self.fields = tuple(schema["fields"])
        self.heads = {
            field: joblib.load(root / "heads" / f"{field}.joblib")
            for field in self.fields
        }

    def predict(self, image: Any) -> dict[str, str]:
        import torch

        pixels = self.model.preprocess_val(_load_rgb(image)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.model.encoder.encode_image(pixels, normalize=True).float()
        matrix = features.cpu().numpy()
        return {
            field: str(_predict_dfmm_head(self.heads[field], matrix)[0])
            for field in self.fields
        }


# Public names. The internal ones stay underscored so the mapping is explicit.
CropBackend = _FashionpediaBackend
CatalogBackend = _Shopping100kBackend
FullbodyBackend = _DfmmBackend

ROUTES = {"crop": CropBackend, "catalog": CatalogBackend, "fullbody": FullbodyBackend}
