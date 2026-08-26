"""The identity contract is load-bearing for attribution: keep it tested."""

from suite import _identity as ident


def test_banner_carries_name_version_and_track():
    out = ident.banner("fullbody")
    assert ident.SUITE_NAME in out
    assert ident.SUITE_VERSION in out
    assert ident.TRACKS["fullbody"] in out
    assert ident.SUITE_URL in out


def test_stamp_attaches_provenance_without_mutating_input():
    payload = {"micro_f1": 0.63}
    out = ident.stamp(payload, "crop")
    assert "suite" not in payload
    assert out["micro_f1"] == 0.63
    assert out["suite"]["name"] == ident.SUITE_NAME
    assert out["suite"]["track"] == ident.TRACKS["crop"]


def test_stamp_never_overwrites_an_existing_suite_key():
    out = ident.stamp({"suite": "preexisting"})
    assert out["suite"] == "preexisting"


def test_markdown_footer_links_the_suite():
    assert ident.SUITE_URL in ident.markdown_footer("catalog")


def test_tracks_do_not_name_source_datasets():
    """Track ids describe the input contract. Naming a corpus here would leak it
    into every stamped result file."""
    banned = ("fashionpedia", "shopping100k", "deepfashion", "dfmm", "imaterialist")
    for key, release in ident.TRACKS.items():
        blob = f"{key} {release}".lower()
        assert not any(b in blob for b in banned), blob


def test_every_model_declares_a_release_tier():
    for name, marker in ident.MODELS.items():
        assert marker in {"*", "**", "***"}, (name, marker)
    assert ident.tier("moda-ner-v-crop") == "*"
    assert ident.tier("unknown-model") == ""


def test_crop_scorer_reproduces_the_published_headline():
    """The launch claim is that shipped predictions regenerate the published
    numbers. If this ever fails, the claim is false and nothing should ship."""
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    published = json.loads(Path("results/crop/moda-ner-v-crop/community_metrics.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "recomputed.json"
        subprocess.run(
            [sys.executable, "-m", "suite.crop.score",
             "--gold", "suite/crop/benchmark.jsonl",
             "--predictions", "results/crop/moda-ner-v-crop/evaluation_predictions.jsonl",
             "--output", str(out)],
            check=True, capture_output=True,
        )
        recomputed = json.loads(out.read_text())

    got = recomputed.get("point_metrics", recomputed)
    want = published["point_metrics"]
    for field in ("attribute_micro_f1", "attribute_field_macro_f1",
                  "category_accuracy", "master_category_accuracy"):
        assert got[field] == want[field], (field, got[field], want[field])
