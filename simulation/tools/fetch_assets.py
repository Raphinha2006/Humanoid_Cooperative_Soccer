"""Fetch only K1 model/meshes plus BSD license at a pinned official commit.

Can run without the simulation dependencies. Existing matching files are verified.
"""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "12a97516f57469078707afb587a5f84c1cb110d1"
REPO = "BoosterRobotics/booster_assets"


def request(url):
    error = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "booster-soccer-sim"}
            )
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except (urllib.error.URLError, OSError) as e:
            error = e
            time.sleep(attempt + 1)
    raise error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets/booster_assets",
    )
    args = parser.parse_args()
    dest = args.destination.resolve()
    tree = json.loads(
        request(f"https://api.github.com/repos/{REPO}/git/trees/{REVISION}?recursive=1")
    )
    records = [
        r
        for r in tree["tree"]
        if r["type"] == "blob"
        and (
            r["path"] in ("LICENSE", "README.md", "robots/K1/K1_22dof.xml")
            or r["path"].startswith("robots/K1/meshes/")
        )
    ]

    def fetch(record):
        target = dest / record["path"]

        def valid(data):
            return (
                hashlib.sha1(
                    b"blob " + str(len(data)).encode() + b"\0" + data
                ).hexdigest()
                == record["sha"]
            )

        if target.exists() and valid(target.read_bytes()):
            return record["path"]
        data = request(
            f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{record['path']}"
        )
        if not valid(data):
            raise RuntimeError(f"Git blob checksum mismatch: {record['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return record["path"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(fetch, records):
            print(name, flush=True)
    (dest / "SOURCE.json").write_text(
        json.dumps(
            {"repository": REPO, "revision": REVISION, "files": records}, indent=2
        )
    )


if __name__ == "__main__":
    main()
