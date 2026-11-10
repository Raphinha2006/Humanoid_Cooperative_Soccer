"""Download pinned upstream inference assets; no SDK, training stack or pickle weights."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fetch_assets import request

ROOT = Path(__file__).resolve().parents[1] / "policies"
SOURCES = {
    "booster_mjlab": (
        "IntelligentRoboticsLab/booster_mjlab",
        "b6d7639c69410a247f74ba61d0034bfc277a3922",
    ),
    "sonic_k1": ("Kyk0/sonic-k1", "47920074916699148abc722f94193d2859739755"),
}
WEIGHTS_REVISION = "1b8092be48e99a1b3c202e5ffa807536a3abeaec"
WEIGHTS = {
    "k1_plant.json": "12a4646d603ea4a4f34dc625b6e152eb352064eb855dc95dfaa9b30689a0c89e",
    "onnx/policy_g1.onnx": "5166065099985c0de4fe791a489e7177d3a8574f7aaa3c5dce086d0013c30781",
    "README.md": "28dda6a7058f0bd2d6f2079287287906889a0bf82ac9582f796c83ff519d0d70",
}


def fetch_sonic_weights():
    destination = ROOT / "sonic_k1/checkpoints/sonic-k1"
    records = []
    for name, expected in WEIGHTS.items():
        target = destination / name
        data = target.read_bytes() if target.exists() else b""
        if hashlib.sha256(data).hexdigest() != expected:
            data = request(
                f"https://huggingface.co/Kyry10/sonic-k1/resolve/{WEIGHTS_REVISION}/{name}?download=true"
            )
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Weight checksum mismatch: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        records.append({"path": name, "sha256": expected, "size": len(data)})
        print("sonic weights", name, len(data), flush=True)
    (destination / "WEIGHTS_SOURCE.json").write_text(
        json.dumps(
            {
                "repository": "Kyry10/sonic-k1",
                "revision": WEIGHTS_REVISION,
                "files": records,
            },
            indent=2,
        )
    )


def main():
    for name, (repo, rev) in SOURCES.items():
        tree = json.loads(
            request(f"https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1")
        )

        def include(p, name=name):
            if name == "booster_mjlab":
                return p.startswith("website/static/demo/") and not p.endswith(
                    "robot.glb"
                )
            return p in [
                "LICENSE",
                "NOTICE",
                "gear_sonic/scripts/deploy_k1_booster.py",
                "gear_sonic/scripts/play_k1.py",
                "gear_sonic/data/assets/robot_description/mjcf/k1.xml",
                "examples/k1/motions/README.md",
            ] or p.startswith(
                (
                    "gear_sonic/data/assets/robot_description/urdf/k1/",
                    "examples/k1/motions/",
                )
            )

        files = [r for r in tree["tree"] if r["type"] == "blob" and include(r["path"])]

        def fetch(r, name=name, repo=repo, rev=rev):
            p = ROOT / name / r["path"]
            if p.exists():
                data = p.read_bytes()
            else:
                data = b""

            def valid(d):
                return (
                    hashlib.sha1(
                        b"blob " + str(len(d)).encode() + b"\0" + d
                    ).hexdigest()
                    == r["sha"]
                )

            if not valid(data):
                data = request(
                    f"https://raw.githubusercontent.com/{repo}/{rev}/{r['path']}"
                )
                if not valid(data):
                    raise ValueError("Checksum mismatch " + r["path"])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            print(name, r["path"], flush=True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, files))
        (ROOT / name / "SOURCE.json").write_text(
            json.dumps({"repository": repo, "revision": rev, "files": files}, indent=2)
        )
        if name == "sonic_k1":
            link = ROOT / name / "gear_sonic/data/assets/robot_description/mjcf/meshes"
            if not link.exists():
                link.symlink_to("../urdf/k1/meshes", target_is_directory=True)
    fetch_sonic_weights()


if __name__ == "__main__":
    main()
