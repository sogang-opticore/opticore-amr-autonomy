"""Compare an archived opticore-amr/amr_lite_spike with the preserved M1 manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(archive, manifest):
    records = []
    dataset = manifest["dataset"]
    for group, items in [("checkpoint", manifest["checkpoints"]), ("teacher", dataset["files"])]:
        for entry in items:
            if group == "checkpoint":
                parts = Path(entry["path"]).parts
                path = Path(*parts[parts.index("artifacts"):])
            else:
                path = Path("artifacts/datasets/teacher") / entry["path"]
            found = archive / path
            digest = hashlib.sha256(found.read_bytes()).hexdigest() if found.is_file() else None
            records.append({"group": group, "path": str(path), "expected_sha256": entry["sha256"],
                            "observed_sha256": digest, "matches": digest == entry["sha256"]})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="path to the original amr_lite_spike directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = root / "experiments/m1_reproducibility/artifacts/manifest.json"
    records = audit(args.archive, json.loads(manifest.read_bytes()))
    print(json.dumps({"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      "files": records}, indent=2))
    raise SystemExit(0 if all(row["matches"] for row in records) else 1)
