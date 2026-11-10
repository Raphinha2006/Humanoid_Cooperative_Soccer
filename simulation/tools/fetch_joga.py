"""Fetch public G1 demonstration resources for local research; do not bundle assets."""

import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://joga.csail.mit.edu/demo/assets/"


def main():
    out = ROOT / "assets/joga"
    out.mkdir(parents=True, exist_ok=True)
    pin = ROOT / "configs/joga_sources.json"
    prior = json.loads(pin.read_text()) if pin.exists() else None
    records = []

    def get(name):
        p = out / name
        b = (
            p.read_bytes()
            if p.exists()
            else urllib.request.urlopen(BASE + name, timeout=120).read()
        )
        digest = hashlib.sha256(b).hexdigest()
        if prior:
            expected = next(f["sha256"] for f in prior["files"] if f["path"] == name)
            if expected != digest:
                raise ValueError("Joga resource checksum changed: " + name)
        p.write_bytes(b)
        records.append({"path": name, "sha256": digest, "size": len(b)})
        print(name, len(b), flush=True)
        return b

    spec = json.loads(get("skills.json"))
    get("scene.xml")
    for name in ["locomotion", "receiving", "passing"]:
        get(spec["modules"][name]["weights"])
    data = get("meshes_v2.bin")
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    cursor = 0
    while cursor < len(data):
        kind = data[cursor]
        cursor += 1
        n = struct.unpack_from("<I", data, cursor)[0]
        cursor += 4
        name = data[cursor : cursor + n].decode()
        cursor += n
        n = struct.unpack_from("<I", data, cursor)[0]
        cursor += 4
        blob = data[cursor : cursor + n]
        cursor += n
        if kind == 1:
            nv, nf = struct.unpack_from("<II", blob)
            vertices = np.frombuffer(blob, dtype="<f4", count=3 * nv, offset=8).reshape(
                -1, 3
            )
            ids = np.cumsum(
                np.frombuffer(blob, dtype="<i4", count=3 * nf, offset=8 + 12 * nv)
            ).reshape(-1, 3)
            triangles = np.zeros(
                nf,
                dtype=np.dtype(
                    [
                        ("normal", "<f4", (3,)),
                        ("vertices", "<f4", (3, 3)),
                        ("attribute", "<u2"),
                    ]
                ),
            )
            triangles["vertices"] = vertices[ids]
            blob = bytes(80) + struct.pack("<I", nf) + triangles.tobytes()
        target = out / "meshes" / name
        # Container paths are filenames, not executable source.
        if not target.resolve().is_relative_to((out / "meshes").resolve()):
            raise ValueError(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    record = {
        "project": "https://joga.csail.mit.edu/",
        "interface": "https://joga.csail.mit.edu/demo/js/sim.js?v=36",
        "purpose": "G1 native policy baseline, local research; training code marked Code (soon) on public demo",
        "files": records,
    }
    pin.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
