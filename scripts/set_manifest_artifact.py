"""Point one model of the complete release manifest at a new ONNX file.

Updates ``artifacts.release_asset_name/url/bytes`` and ``sha256.onnx``, records the patch report under
``artifacts.directml_patch``, and sets the optional ``artifacts.replaces`` (previous asset names, so newer
INTERSECT versions offer UPDATE and remove the old file) and ``artifacts.update_note`` (tooltip text).
Older plugins ignore the optional fields.

usage: set_manifest_artifact.py MANIFEST.json MODEL_ID FILE.onnx --release v0.2.N
           [--report FILE.report.json --source-url URL] [--replaces OLD.onnx ...] [--update-note TEXT]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_URL = "https://github.com/tucktuckg00se/intersect-stem-models"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", type=Path)
    p.add_argument("model_id")
    p.add_argument("file", type=Path)
    p.add_argument("--release", required=True, help="release tag that will host FILE, e.g. v0.2.2")
    p.add_argument("--report", type=Path, help="JSON report from patch_onnx_for_directml.py")
    p.add_argument("--source-url", help="where the unpatched source file can be downloaded")
    p.add_argument("--replaces", nargs="*", default=None, help="previous release asset names of this model")
    p.add_argument("--update-note", help="one line shown with the UPDATE button")
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    entry = next((e for e in manifest["models"] if e.get("model_id") == args.model_id), None)
    if entry is None:
        raise SystemExit(f"model_id {args.model_id} not in {args.manifest}")
    data = args.file.read_bytes()
    art = entry.setdefault("artifacts", {})
    previous = art.get("release_asset_name")
    art["release_asset_name"] = args.file.name
    art["release_asset_url"] = f"{REPO_URL}/releases/download/{args.release}/{args.file.name}"
    art["release_asset_bytes"] = len(data)
    art.setdefault("sha256", {})["onnx"] = hashlib.sha256(data).hexdigest()
    if args.report:
        report = json.loads(args.report.read_text())
        if report["output"]["sha256"] != art["sha256"]["onnx"]:
            raise SystemExit("report doesn't belong to this file (sha256 mismatch)")
        art["directml_patch"] = {
            "script": "scripts/patch_onnx_for_directml.py",
            "source_url": args.source_url or art.get("directml_patch", {}).get("source_url"),
            "source_sha256": report["source"]["sha256"],
            "rewrites": [k for k, v in report["rewrites"].items() if isinstance(v, int) and v > 0],
            "rewrite_counts": {k: v for k, v in report["rewrites"].items() if isinstance(v, int) and v > 0},
            "barrier_options": report.get("barrier_options"),
            "cpu_bit_identical": report.get("verify", {}).get("bit_identical"),
        }
    if args.replaces is not None:
        art["replaces"] = sorted(set(args.replaces) - {args.file.name})
    elif previous and previous != args.file.name:
        art["replaces"] = sorted(set(art.get("replaces", [])) | {previous})
    if args.update_note:
        art["update_note"] = args.update_note
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.model_id}: {args.file.name} ({len(data)} bytes, sha256 {art['sha256']['onnx']}), replaces {art.get('replaces')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
