"""Fetch checksum-pinned NUbots and B-Human independent K1 skill exports."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fetch_assets import request

ROOT = Path(__file__).resolve().parents[1]


def fetch_manifest(manifest_path, directory):
    manifest = json.loads(manifest_path.read_text())
    destination = ROOT / "policies" / directory

    def fetch(record):
        target = destination / record["path"]
        data = target.read_bytes() if target.exists() else b""
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            url = f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['revision']}/{record['path']}"
            data = request(url)
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"Checksum mismatch: {record['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        print(record["path"], len(data), flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, manifest["files"]))
    (destination / "SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    for filename, directory in [
        ("skill_sources.json", "nubots"),
        ("bhuman_skill_sources.json", "bhuman"),
    ]:
        fetch_manifest(ROOT / "configs" / filename, directory)


if __name__ == "__main__":
    main()
