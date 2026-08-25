# Reproduce (pre-launch stub)

Ports of the manifest builders, scorers, and bootstrap code from the private benchmark
package land here in the week of 1 Sep 2026 (publication sequence, Phase 1). The gate for
launch: fresh clone + `pip install -r requirements.txt` + one command reproduces one track's
headline number from the shipped prediction files.

Per-track quickstarts will follow the pattern:

```bash
python -m suite.fashionpedia_moda15.build_manifest --source data/fashionpedia  # you download the source
python -m suite.fashionpedia_moda15.score --predictions results/fashionpedia/moda_predictions.jsonl
```

## Provenance stamping (contract for the port)

Every scorer in this repository must, without exception:

1. print `suite.banner(track)` to **stderr** at start — stderr so that piping
   stdout to a file keeps machine-readable output clean;
2. pass its result dict through `suite.stamp(payload, track)` before writing
   JSON, so the suite name, version, frozen date, URL and citation travel with
   the numbers;
3. append `suite.markdown_footer(track)` under any table it renders.

The reason is attribution durability: a metric pasted out of a result file three
months from now should still say what produced it. `tests/test_identity.py`
guards the contract.

Bump `SUITE_VERSION` in `suite/_identity.py` in the same commit as any change to
data, split, taxonomy, metric, comparator, or router. Never redefine a version in
place.
