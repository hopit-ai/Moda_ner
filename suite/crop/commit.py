"""Hash a prediction file before its labels are opened.

The point is ordering. You produce predictions without access to gold, commit
the hash here, and only then does the scorer read labels. After the hash is
recorded neither side can swap the file, which is what makes a later score
worth anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .. import SUITE_NAME, SUITE_VERSION, TRACKS


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Commit a prediction file by hash.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None,
                        help="where to write the commitment (default: alongside the predictions)")
    args = parser.parse_args()

    path: Path = args.predictions
    rows = sum(1 for line in path.read_text().splitlines() if line.strip())
    digest = sha256_of(path)
    record = {
        "suite": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "track": TRACKS["crop"],
        "predictions_file": path.name,
        "sha256": digest,
        "rows": rows,
        "committed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = args.output or path.with_suffix(path.suffix + ".commitment.json")
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"committed: {path.name}  sha256={digest[:8]}...  rows={rows}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
