import os
import time

from scripts.snoo_prune import prune


def _session(root, name, age_s):
    s = root / name
    (s / "000").mkdir(parents=True)
    (s / "000" / "meta.json").write_text("{}")
    (s / "000" / "recon.npz").write_bytes(b"x" * 1000)
    (s / "vggt.npz").write_bytes(b"y" * 500)
    (s / "slam" / "frames").mkdir(parents=True)
    (s / "slam" / "frames" / "a.jpg").write_bytes(b"z" * 200)
    (s / "room.json").write_text("{}")
    t = time.time() - age_s
    for p in s.rglob("*"):
        os.utime(p, (t, t))
    return s


def test_prune_keeps_the_newest_sessions_and_only_heavy_files(tmp_path):
    old = _session(tmp_path, "explore_old", 3600)
    new = _session(tmp_path, "explore_new", 10)
    n, freed = prune(tmp_path, keep=1)
    assert n == 3 and freed == 1700
    assert not (old / "000" / "recon.npz").exists() and not (old / "vggt.npz").exists()
    assert not (old / "slam" / "frames").exists()
    assert (old / "000" / "meta.json").exists() and (old / "room.json").exists()
    assert (new / "000" / "recon.npz").exists() and (new / "slam" / "frames" / "a.jpg").exists()
    assert prune(tmp_path, keep=1) == (0, 0)  # 2 回目は何もしない
