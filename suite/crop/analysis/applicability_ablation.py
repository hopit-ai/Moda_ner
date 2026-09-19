"""What does the crop route's applicability head earn, and under which scoring protocol?

The released crop route (HopitAI/moda-ner-v-crop) predicts, per field, an applicability
probability and a value distribution. A calibrated per-field applicability threshold decides
whether the field is emitted at all. This ablates that decision only, holding the model, the
value thresholds and every other setting fixed:

  shipped            thresholds.json as released
  uncalibrated       applicability threshold 0.5 on every field (the head, without calibration)
  applicability_off  applicability threshold 0.0 on every field (always emit a value)

Each variant is scored two ways, reading the result against the sparse-annotation rescore (sparse_annotation_rescore.py):

  exhaustive       the published protocol (suite.crop.score, unmodified)
  annotated-only   each garment judged only on fields its source annotation covers

What this can and cannot show. Under annotated-only scoring every judged cell has a gold value, so
there are no absent cases: suppressing a field can only cost recall, and switching the head off is
favoured BY CONSTRUCTION. The annotated-only row is therefore not evidence that the head learned
annotator habits. What the run does show is how much of the exhaustive score rides on absence
decisions, which Fashionpedia's labels cannot adjudicate: they record no explicit absence and leave
two thirds of judged cells unannotated. Separating visual absence from annotator habit needs
independent annotation.

Guard: the shipped decode must reproduce the published prediction file. The model is run once;
the three variants differ only in decoding.

Section 6 of the resource report, arXiv:2609.13279 v2.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def crop_from_row(opened, row):
    """Copy of the training pipeline's crop, the one used to
    produce the published predictions: round the xywh box, clamp to the image, no padding."""
    image = row.get("image") if isinstance(row.get("image"), dict) else {}
    bbox = row.get("bbox") or image.get("bbox_xywh")
    if not bbox:
        return opened.copy()
    if isinstance(bbox, str):  # the gold stores the box as a JSON string
        bbox = json.loads(bbox)
    x, y, width, height = (float(v) for v in bbox)
    left, top = max(0, round(x)), max(0, round(y))
    right, bottom = min(opened.width, round(x + width)), min(opened.height, round(y + height))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop {bbox} for image size {opened.size}")
    return opened.crop((left, top, right, bottom))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite-root", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True, help="HopitAI/moda-ner-v-crop snapshot")
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--published-predictions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True, help="where the probability cache lives")
    ap.add_argument("--device", default="auto")          # resolved by the suite's own resolver
    # 'released' reproduces the suite backend exactly; 'training' uses the checkpoint's own config.
    ap.add_argument("--preprocess", choices=("released", "training"), default="training")
    ap.add_argument("--base-config", type=Path, help="open_clip_config.json of HopitAI/moda-fashion-distilled")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)  # suite protocol
    ap.add_argument("--seed", type=int, default=42)                  # scorer default
    args = ap.parse_args()

    sys.path.insert(0, str(args.suite_root))
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from suite._model.architecture import FashionSiglipAttributeClassifier
    from suite._model.calibration import decode_probabilities
    from suite._model.contract import AttributeVocabulary
    from suite._model.routes import _resolve_device
    from suite.crop import score as S

    gold = S._load_jsonl(args.gold)
    vocab = AttributeVocabulary.from_dict(json.loads((args.model_dir / "vocabulary.json").read_text()))
    shipped = json.loads((args.model_dir / "thresholds.json").read_text())
    metrics = json.loads((args.model_dir / "metrics.json").read_text())

    # ---- phase 1: run the released model once, cache probabilities ---------------------------
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir / "crop_probabilities.jsonl"
    if not cache.exists():
        device = torch.device(_resolve_device(args.device))
        model = FashionSiglipAttributeClassifier(
            vocab, model_name="ViT-B-16-SigLIP", pretrained=None,
            use_spatial_tokens=bool(metrics.get("use_spatial_tokens", False)),
            spatial_residual=bool(metrics.get("spatial_residual", False)))
        model.load_state_dict(load_file(args.model_dir / "model.safetensors"), strict=True)
        model.requires_grad_(False).eval().to(device)
        if args.preprocess == "training":
            # The checkpoint was trained as hf-hub:HopitAI/moda-fashion-distilled, whose
            # preprocess_cfg is mean/std 0.5, bicubic, resize_mode 'squash'. Building the model
            # as plain ViT-B-16-SigLIP with pretrained=None instead gives OpenAI-CLIP
            # normalisation and a centre crop. Rebuild the eval transform from the base config.
            import open_clip
            cfg = json.loads(args.base_config.read_text())["preprocess_cfg"]
            model.preprocess_val = open_clip.image_transform(
                224, is_train=False, mean=tuple(cfg["mean"]), std=tuple(cfg["std"]),
                interpolation=cfg.get("interpolation", "bicubic"),
                resize_mode=cfg.get("resize_mode", "squash"))
        print("preprocess_val:", str(model.preprocess_val).replace(chr(10), " "), flush=True)
        multi = set(vocab.multi_label_fields)
        opened: dict[str, Any] = {}
        t0, rows_out = time.time(), []
        for start in range(0, len(gold), args.batch_size):
            batch = gold[start:start + args.batch_size]
            pixels = []
            for row in batch:
                name = row["image"]["s3_uri"].rsplit("/", 1)[1]
                if name not in opened:
                    with Image.open(args.images_dir / name) as im:
                        opened[name] = im.convert("RGB").copy()
                pixels.append(model.preprocess_val(crop_from_row(opened[name], row)))
            with torch.inference_mode():
                out = model(torch.stack(pixels).to(device))
            for i, row in enumerate(batch):
                rows_out.append({
                    "record_id": row["record_id"],
                    "categories": {f: torch.softmax(l.float()[i], -1).cpu().tolist() for f, l in out["categories"].items()},
                    "applicability": {f: float(torch.sigmoid(l.float()[i]).reshape(-1)[0].cpu()) for f, l in out["applicability"].items()},
                    "values": {f: (torch.sigmoid(l.float()[i]) if f in multi else torch.softmax(l.float()[i], -1)).cpu().tolist()
                               for f, l in out["values"].items()},
                })
            if start == 0 or (start // args.batch_size) % 20 == 0:
                print(f"  inference {start + len(batch)}/{len(gold)} on {device} ({time.time() - t0:.0f}s)", flush=True)
        cache.write_text("".join(json.dumps(r) + "\n" for r in rows_out))
        print(f"cached probabilities for {len(rows_out)} crops in {time.time() - t0:.0f}s", flush=True)
    probs = [json.loads(l) for l in cache.read_text().splitlines()]

    # ---- phase 2: decode three ways; only the applicability threshold changes ---------------
    def with_applicability(value):
        t = copy.deepcopy(shipped)
        for f in vocab.fields:
            if f in S.CATEGORY_FIELDS:
                continue
            t.setdefault(f, {})["applicability"] = value
        return t

    variants = {"shipped": shipped, "uncalibrated": with_applicability(0.5),
                "applicability_off": with_applicability(0.0)}
    decoded = {name: [{"record_id": p["record_id"],
                       "attributes": decode_probabilities(p["categories"], p["applicability"], p["values"], vocab, t)}
                      for p in probs]
               for name, t in variants.items()}

    # Guard: the shipped decode must reproduce the published predictions.
    published = {r["record_id"]: r["attributes"] for r in S._load_jsonl(args.published_predictions)}
    norm = lambda a: {k: (sorted(v) if isinstance(v, list) else v) for k, v in a.items()}
    identical = sum(norm(r["attributes"]) == norm(published.get(r["record_id"], {})) for r in decoded["shipped"])
    pub_pt = S._point_metrics(list(S._build_group_stats(gold, S._load_jsonl(args.published_predictions))[0].values()))
    rep_pt = S._point_metrics(list(S._build_group_stats(gold, decoded["shipped"])[0].values()))
    guard = {"rows_identical": identical, "rows": len(gold),
             "published_micro": round(pub_pt["attribute_micro_f1"], 4),
             "reproduced_micro": round(rep_pt["attribute_micro_f1"], 4),
             "published_macro": round(pub_pt["attribute_field_macro_f1"], 4),
             "reproduced_macro": round(rep_pt["attribute_field_macro_f1"], 4)}
    print("guard:", guard, flush=True)

    # ---- phase 3: score every variant under both protocols, paired against shipped ----------
    def group_stats(preds, mask):
        """suite.crop.score._build_group_stats, plus the per-row annotation mask of sparse_annotation_rescore.py."""
        pred_by_id = {S._row_id(r, i): r for i, r in enumerate(preds)}
        groups: dict[str, dict[str, Any]] = {}
        fields = sorted({str(f) for r in gold for f in r.get("evaluated_attributes", []) if f not in S.CATEGORY_FIELDS})
        for i, g in enumerate(gold):
            pa = (pred_by_id.get(S._row_id(g, i), {}).get("attributes")) or {}
            ga = S.attributes_from_entry(g)
            gid = str((g.get("image") or {}).get("image_group_id") or S._row_id(g, i))
            st = groups.setdefault(gid, {"rows": 0, "category_correct": 0, "master_correct": 0, "tp": 0, "fp": 0,
                                         "fn": 0, "fields": defaultdict(lambda: [0, 0, 0, 0])})
            st["rows"] += 1
            for f, key in (("category", "category_correct"), ("master_category", "master_correct")):
                st[key] += int(S._values(f, ga.get(f)) == S._values(f, pa.get(f)))
            annotated = set(g.get("supervised_attributes", []))
            for f in fields:
                if mask and f not in annotated:
                    continue
                gv, pv = S._values(f, ga.get(f)), S._values(f, pa.get(f))
                tp, fp, fn = len(gv & pv), len(pv - gv), len(gv - pv)
                st["tp"] += tp; st["fp"] += fp; st["fn"] += fn
                c = st["fields"][f]; c[0] += tp; c[1] += fp; c[2] += fn; c[3] += len(gv)
        return groups

    keys = ("attribute_micro_f1", "attribute_field_macro_f1", "attribute_micro_precision", "attribute_micro_recall")
    results = {}
    for protocol, mask in (("exhaustive", False), ("annotated_only", True)):
        base = group_stats(decoded["shipped"], mask)
        ids = sorted(base)
        results[protocol] = {}
        for name in variants:
            var = group_stats(decoded[name], mask)
            pt = S._point_metrics([var[i] for i in ids])
            entry = {"point": {k: round(pt[k], 4) for k in keys}}
            if name != "shipped":
                bp = S._point_metrics([base[i] for i in ids])
                delta = {k: pt[k] - bp[k] for k in keys[:2]}
                rng, dist = random.Random(args.seed), {k: [] for k in keys[:2]}
                for _ in range(args.bootstrap_samples):
                    smp = [ids[rng.randrange(len(ids))] for _ in ids]
                    vp, sp = S._point_metrics([var[i] for i in smp]), S._point_metrics([base[i] for i in smp])
                    for k in keys[:2]:
                        dist[k].append(vp[k] - sp[k])
                entry["delta_vs_shipped"] = {k: round(v, 4) for k, v in delta.items()}
                entry["ci95"] = {k: {"low": round(S._percentile(v, 0.025), 4), "high": round(S._percentile(v, 0.975), 4)}
                                 for k, v in dist.items()}
            results[protocol][name] = entry
            print(f"  {protocol:15} {name:18} {entry['point']}  {entry.get('delta_vs_shipped', '')}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, preds in decoded.items():
        (args.out_dir / f"predictions_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in preds))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (args.out_dir / "applicability_ablation.json").write_text(json.dumps({
        "question": "What does the crop route's applicability head earn, and under which scoring protocol?",
        "model": {"repo": "HopitAI/moda-ner-v-crop", "weights_sha256": sha(args.model_dir / "model.safetensors"),
                  "thresholds_sha256": sha(args.model_dir / "thresholds.json")},
        "gold_sha256": sha(args.gold),
        "guard": guard,
        "variants": {"shipped": "released thresholds", "uncalibrated": "applicability 0.5 on every field",
                     "applicability_off": "applicability 0.0 on every field"},
        "results": results,
        "prediction_sha256": {n: sha(args.out_dir / f"predictions_{n}.jsonl") for n in decoded},
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed, "unit": "source image"},
    }, indent=2) + "\n")
    print("wrote", args.out_dir / "applicability_ablation.json")


if __name__ == "__main__":
    main()
