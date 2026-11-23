"""Zip a converted Core ML package and point one model of the release manifest at it.

Writes OUT_DIR/<package>_coreml.zip (the .mlpackage folder at the zip's root, sorted entries, fixed
timestamps, so the same package always gives the same zip) and sets ``artifacts.coreml`` on the model:
``release_asset_name/url/bytes``, ``sha256`` (of the zip), ``package_name``, and where it came from.
INTERSECT on macOS downloads and unpacks it next to the ONNX file and runs it for DEVICE = GPU; older
plugins and other platforms ignore the field.

usage: set_manifest_coreml.py MANIFEST.json MODEL_ID PACKAGE.mlpackage --release v0.2.N --out-dir DIR
           --source-url URL --source-sha256 HEX [--fidelity-db DB]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/tucktuckg00se/intersect-stem-models"


def zip_package(package: Path, target: Path) -> None:
    files = sorted(p for p in package.rglob("*") if p.is_file())
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            info = zipfile.ZipInfo(str(Path(package.name) / f.relative_to(package)), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, f.read_bytes())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", type=Path)
    p.add_argument("model_id")
    p.add_argument("package", type=Path)
    p.add_argument("--release", required=True, help="release tag that will host the zip, e.g. v0.2.3")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-url", required=True, help="the ONNX file the package was converted from")
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--fidelity-db", type=float, help="SNR of Core ML against ONNX Runtime on the reference input")
    args = p.parse_args()

    package = args.package.resolve()
    if package.suffix != ".mlpackage" or not (package / "Manifest.json").is_file():
        raise SystemExit(f"{package} is not an .mlpackage")
    manifest = json.loads(args.manifest.read_text())
    entry = next((e for e in manifest["models"] if e.get("model_id") == args.model_id), None)
    if entry is None:
        raise SystemExit(f"model_id {args.model_id} not in {args.manifest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_name = package.stem + "_coreml.zip"
    target = args.out_dir / zip_name
    zip_package(package, target)
    data = target.read_bytes()

    coreml = {
        "release_asset_name": zip_name,
        "release_asset_url": f"{REPO_URL}/releases/download/{args.release}/{zip_name}",
        "release_asset_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "package_name": package.name,
        "compute_precision": "fp32",
        "minimum_macos": "13.0",
        "converted_from": {"url": args.source_url, "sha256": args.source_sha256,
                           "script": "scripts/coreml/convert_coreml.py"},
    }
    if args.fidelity_db is not None:
        coreml["fidelity_db_vs_onnxruntime"] = round(args.fidelity_db, 1)
    entry.setdefault("artifacts", {})["coreml"] = coreml
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.model_id}: {zip_name} ({len(data) / 1e6:.1f} MB, sha256 {coreml['sha256'][:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
