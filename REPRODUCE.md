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
