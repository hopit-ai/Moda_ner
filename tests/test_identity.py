"""The identity contract is load-bearing for attribution: keep it tested."""

from suite import _identity as ident


def test_banner_carries_name_version_and_track():
    out = ident.banner("dfmm18")
    assert ident.SUITE_NAME in out
    assert ident.SUITE_VERSION in out
    assert ident.TRACKS["dfmm18"] in out
    assert ident.SUITE_URL in out


def test_stamp_attaches_provenance_without_mutating_input():
    payload = {"micro_f1": 0.63}
    out = ident.stamp(payload, "fashionpedia_moda15")
    assert "suite" not in payload
    assert out["micro_f1"] == 0.63
    assert out["suite"]["name"] == ident.SUITE_NAME
    assert out["suite"]["track"] == ident.TRACKS["fashionpedia_moda15"]


def test_stamp_never_overwrites_an_existing_suite_key():
    out = ident.stamp({"suite": "preexisting"})
    assert out["suite"] == "preexisting"


def test_markdown_footer_links_the_suite():
    assert ident.SUITE_URL in ident.markdown_footer("shopping100k_10")
