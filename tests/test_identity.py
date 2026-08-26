"""The identity contract is load-bearing for attribution: keep it tested."""

from suite import _identity as ident


def test_banner_carries_name_version_and_track():
    out = ident.banner("fullbody18")
    assert ident.SUITE_NAME in out
    assert ident.SUITE_VERSION in out
    assert ident.TRACKS["fullbody18"] in out
    assert ident.SUITE_URL in out


def test_stamp_attaches_provenance_without_mutating_input():
    payload = {"micro_f1": 0.63}
    out = ident.stamp(payload, "crop15")
    assert "suite" not in payload
    assert out["micro_f1"] == 0.63
    assert out["suite"]["name"] == ident.SUITE_NAME
    assert out["suite"]["track"] == ident.TRACKS["crop15"]


def test_stamp_never_overwrites_an_existing_suite_key():
    out = ident.stamp({"suite": "preexisting"})
    assert out["suite"] == "preexisting"


def test_markdown_footer_links_the_suite():
    assert ident.SUITE_URL in ident.markdown_footer("catalog10")


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
