"""A reused independent test must match weights, protocol and all output bytes."""

from soccer_sim.gpu_receive import receipt


def test_independent_artifact_reuse_rejects_changes(tmp_path, monkeypatch):
    provenance = {"source": "frozen"}
    monkeypatch.setattr(receipt, "sources", lambda: dict(provenance))
    weights = tmp_path / "best.pt"
    result = tmp_path / "result.json"
    weights.write_bytes(b"weights")
    result.write_text("[]")
    receipt.write_receipt(tmp_path, "best.pt", 230000, "gpu", 1024, [result])
    check = lambda count=1024: receipt.verified(
        tmp_path, "best.pt", 230000, "gpu", count
    )
    assert check()
    assert not check(512)
    result.write_text("[1]")
    assert not check()
    result.write_text("[]")
    weights.write_bytes(b"different weights")
    assert not check()
    weights.write_bytes(b"weights")
    provenance["source"] = "changed"
    assert not check()
