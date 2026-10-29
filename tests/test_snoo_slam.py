"""app/snoo_slam.py の置き場の形・座標合わせ・書き出しを、作り物の SLAM で試す（GPU なし）。"""

import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.snoo_slam import (
    KeyFrame,
    SlamService,
    align_session,
    box_mask,
    camera_up,
    fit_floor,
    frame_poses,
    mask_boxes,
    read_trajectory,
    station_anchors,
    umeyama,
    voxel_down,
)
from app.snoo_slam_tap import file_stem

H = 0.08  # カメラの高さ（m）
T0 = float(int(time.time()))


def cam_rot(yaw, pan_deg, tilt_deg):
    """車の向き yaw・パン（右が +）・チルト（上が +）のカメラの向き
    （カメラ → 世界、OpenCV の軸）。
    """
    a = yaw - math.radians(pan_deg)
    t = math.radians(tilt_deg)
    f = np.array([math.cos(a) * math.cos(t), math.sin(a) * math.cos(t), math.sin(t)])
    r = np.array([math.sin(a), -math.cos(a), 0.0])
    d = np.cross(f, r)
    return np.stack([r, d, f], 1)


def rot3(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


class Scene:
    """セッションの座標の本当の位置と、それを大きさ・向き・位置を変えた SLAM の座標。"""

    def __init__(self, stations, s=0.3, rng=None):
        rng = rng or np.random.default_rng(1)
        self.q = rot3([0.3, -1.0, 0.5], 1.1)  # SLAM の座標 = s * q @ 本当 + t
        self.s, self.t = s, np.array([0.5, -2.0, 1.0])
        self.frames, cams, rots = [], [], []
        for sid, (x, y, yaw) in stations.items():
            for pan in (-30, 0, 30):
                for tilt in (-15, 0):
                    self.frames.append(KeyFrame(0, Path("x"), pan, tilt, station=sid))
                    cams.append([x, y, H])
                    rots.append(cam_rot(yaw, pan, tilt))
        self.true_cams, self.true_rots = np.array(cams), np.array(rots)
        floor = np.c_[rng.uniform(-2, 2, (4000, 2)), np.zeros(4000)]
        wall = np.c_[np.full(800, 1.5), rng.uniform(-2, 2, 800), rng.uniform(0.05, 0.5, 800)]
        self.true_points = np.r_[floor, wall]

    def slam(self, p):
        return self.s * np.asarray(p) @ self.q.T + self.t

    def cams(self):
        return self.slam(self.true_cams)

    def rots(self):
        return np.einsum("ij,njk->nik", self.q, self.true_rots)

    def points(self):
        return self.slam(self.true_points)


STATIONS = {0: (0.0, 0.0, 0.0), 1: (0.4, -0.1, -0.5), 2: (0.1, 0.35, 1.2), 3: (-0.2, 0.1, 2.5)}


def test_umeyama_finds_the_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    r = rot3([1, 2, 3], 0.7)
    dst = 2.5 * src @ r.T + [1, 2, 3]
    s, r2, t = umeyama(src, dst)
    assert s == pytest.approx(2.5) and np.allclose(r2, r) and np.allclose(t, [1, 2, 3])
    s, r2, _ = umeyama(src[:, :2], 2.5 * src[:, :2], with_scale=False)
    assert s == 1.0 and np.allclose(r2, np.eye(2))


def test_camera_up_is_the_floor_normal():
    rots = np.stack([cam_rot(0.3, 20, -15), cam_rot(-1.0, 0, 0)])
    assert np.allclose(camera_up(rots, [-15, 0]), [[0, 0, 1], [0, 0, 1]])


def test_fit_floor_finds_the_plane_under_the_cameras():
    sc = Scene(STATIONS)
    floor = fit_floor(sc.points(), sc.q @ [0, 0, 1.0], sc.cams())
    assert floor is not None
    assert np.allclose(floor.normal, sc.q @ [0, 0, 1.0], atol=1e-6)
    h = sc.cams() @ floor.normal + floor.offset
    assert np.allclose(h, sc.s * H, atol=1e-6)


def test_align_recovers_the_session_frame_from_floor_height_and_stations():
    sc = Scene(STATIONS)
    ref = dict(STATIONS)
    al = align_session(sc.cams(), sc.rots(), sc.frames, sc.points(), ref)
    assert al.method == "stations" and al.stations == [0, 1, 2, 3]
    assert al.frame.scale == pytest.approx(1 / sc.s)  # 大きさは床とカメラの高さだけで決まる
    assert al.rmse_m == pytest.approx(0, abs=1e-6)
    assert al.sim2_scale == pytest.approx(1.0)
    poses = frame_poses(al, sc.cams(), sc.rots(), sc.frames)
    for f, (x, y, yaw, z) in zip(sc.frames, poses, strict=True):
        tx, ty, tyaw = STATIONS[f.station]
        assert (x, y, z) == pytest.approx((tx, ty, H), abs=1e-6)
        assert math.remainder(yaw - tyaw, 2 * math.pi) == pytest.approx(0, abs=1e-6)
    # 点は z が床からの高さになる
    p = al.frame.apply(sc.points())
    assert np.allclose(p, sc.true_points, atol=1e-6)


def test_align_with_one_station_uses_its_heading():
    sc = Scene({0: (0.0, 0.0, 0.0)})
    al = align_session(sc.cams(), sc.rots(), sc.frames, sc.points(), {0: (0.0, 0.0, 0.0)})
    assert al.method == "heading"
    assert np.allclose(al.frame.apply(sc.points()), sc.true_points, atol=1e-6)
    # poses.json が無ければ最初のコマ（パン -30 のコマでも車の向きに直す）を原点・x にする
    al = align_session(sc.cams(), sc.rots(), sc.frames, sc.points(), {})
    assert al.method == "first_frame"
    assert np.allclose(al.frame.apply(sc.points()), sc.true_points, atol=1e-6)


def test_align_uses_station_poses_with_a_different_height():
    sc = Scene(STATIONS)
    al = align_session(sc.cams(), sc.rots(), sc.frames, sc.points(), STATIONS, height_m=0.10)
    assert al.frame.scale == pytest.approx(0.10 / 0.08 / sc.s)
    assert al.sim2_scale == pytest.approx(0.08 / 0.10)  # poses.json との大きさの比が分かる


def test_station_anchor_falls_back_to_the_last_moving_frame():
    fr = [
        KeyFrame(1_000_000_000, Path("a"), moving=True),
        KeyFrame(2_000_000_000, Path("b"), moving=True),
        KeyFrame(9_000_000_000, Path("c"), moving=True),
        KeyFrame(3_000_000_000, Path("d"), station=1),
    ]
    a = station_anchors(fr, {1: 3_000_000_000, 2: 9_500_000_000, 3: 20_000_000_000})
    assert a == {1: [3], 2: [2]}  # 3 は 3 秒以内に走ったコマが無い
    assert station_anchors(fr[:3], {1: 2_500_000_000}) == {1: [1]}


def test_voxel_down_averages_points_in_a_voxel():
    xyz = np.array([[0.01, 0.01, 0.0], [0.02, 0.02, 0.0], [0.5, 0.5, 0.1]])
    rgb = np.array([[0, 0, 0], [10, 20, 30], [255, 255, 255]], np.uint8)
    p, c, n = voxel_down(xyz, rgb, 0.03)
    order = np.argsort(p[:, 0])
    assert np.allclose(p[order], [[0.015, 0.015, 0.0], [0.5, 0.5, 0.1]])
    assert c[order].tolist() == [[5, 10, 15], [255, 255, 255]] and n[order].tolist() == [2, 1]


def test_cat_boxes_are_painted_and_masked():
    img = np.zeros((480, 640, 3), np.uint8)
    out = mask_boxes(img, [[0.5, 0.5, 1.0, 1.0]])
    assert (out[240:, 320:] == 127).all() and (out[:240] == 0).all() and img.max() == 0
    m = box_mask([[0.5, 0.5, 1.0, 1.0]], (392, 518))
    assert m[196:, 259:].all() and not m[:196].any() and not m[:, :259].any()


# --- サービス（作り物の SLAM） ---


class FakeSlam:
    """VggtSlam の代わり。コマの名前から本当の位置を引いて、Scene の SLAM の座標で返す。"""

    def __init__(self, scene: Scene, truth: dict[str, int], fail_at: int | None = None):
        self.sc, self.truth = scene, truth
        self.names: list[str] = []
        self.segment = self.submaps = self.loops = self.failures = 0
        self.timing: list[dict] = []
        self.calls: list[list[str]] = []
        self.fail_at = fail_at

    def add(self, paths, boxes=None):
        self.calls.append([Path(p).name for p in paths])
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            self.segment += 1
            self.failures += 1
            self.names = []
            return {"ok": False, "frames": len(paths), "submap_ms": 1.0, "vram_gb": 1.0}
        self.names += [str(p) for p in paths]
        self.submaps += 1
        info = {"ok": True, "frames": len(paths) + 1, "submap_ms": 10.0, "vram_gb": 1.0}
        self.timing.append(info)
        return info

    def poses(self):
        cams, rots = self.sc.cams(), self.sc.rots()
        return {
            n: (cams[self.truth[Path(n).name]], rots[self.truth[Path(n).name]]) for n in self.names
        }

    def points(self):
        p = self.sc.points()
        if not self.names:
            return p[:0], np.zeros((0, 3), np.uint8), np.zeros(0, np.int32), []
        return p, np.full((len(p), 3), 100, np.uint8), np.zeros(len(p), np.int32), self.names[:1]


def _jpeg(path: Path):
    cv2.imwrite(str(path), np.full((480, 640, 3), 200, np.uint8))


def make_session(root: Path, sc: Scene) -> tuple[Path, dict[str, int], list[int]]:
    """Scene のステーションを、地図のサーバーと同じ形で置く。返すのはコマ → Scene の番号。"""
    session = root / "data" / "explore_test"
    truth, starts = {}, []
    t = 100 * 10**9
    for sid in sorted({f.station for f in sc.frames}):
        d = session / f"{sid:03d}"
        d.mkdir(parents=True)
        frames = []
        starts.append(t)
        for i, f in enumerate(sc.frames):
            if f.station != sid:
                continue
            name = f"{i:02d}_pan{int(f.pan_deg):+04d}_tilt{int(f.tilt_deg):+03d}.jpg"
            _jpeg(d / name)
            frames.append(
                {
                    "file": name,
                    "t_capture_ns": t,
                    "pan_deg": f.pan_deg,
                    "tilt_deg": f.tilt_deg,
                    "moving": False,
                    "steer_deg": 0.0,
                    "speed": 0.0,
                }
            )
            truth[f"{t:020d}.jpg"] = i
            t += 10**9
        (d / "meta.json").write_text(json.dumps({"station_id": sid, "frames": frames}))
        t += 20 * 10**9
    poses = {
        str(k): {"x": v[0], "y": v[1], "yaw_deg": math.degrees(v[2])} for k, v in STATIONS.items()
    }
    (session / "poses.json").write_text(json.dumps(poses))
    return session, truth, starts


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


def test_service_reads_stations_and_queue_and_writes_the_outputs(tmp_path):
    sc = Scene(STATIONS)
    session, truth, starts = make_session(tmp_path, sc)
    # タップの置き場に、ステーション 1 の真上を走りながら撮ったコマ 2 枚（猫が 1 匹）
    q = tmp_path / "queue" / "192.168.0.188"
    q.mkdir(parents=True)
    i1 = next(i for i, f in enumerate(sc.frames) if f.station == 1 and f.pan_deg == 0)
    for k, t in enumerate((starts[1] - 2 * 10**9, starts[1] - 10**9)):
        _jpeg(q / f"{file_stem(t)}.jpg")
        info = {
            "t_capture_ns": t,
            "image": f"{file_stem(t)}.jpg",
            "pan_deg": 0.0,
            "tilt_deg": float(sc.frames[i1].tilt_deg),
            "steer_deg": 5.0,
            "speed": 12.0,
            "moving": True,
            "t_recv": T0 - 1.0,
            "dets": [{"box": [0.0, 0.0, 0.5, 0.5], "cat": 0.9, "snoo": 0.8}] if k == 0 else [],
        }
        (q / f"{file_stem(t)}.json").write_text(json.dumps(info))
        truth[f"{t:020d}.jpg"] = i1
    # 最初のステーションより前のコマは別のセッションのものなので使わない
    _jpeg(q / f"{file_stem(starts[0] - 10**9)}.jpg")
    (q / f"{file_stem(starts[0] - 10**9)}.json").write_text(
        json.dumps(
            {"t_capture_ns": starts[0] - 10**9, "image": f"{file_stem(starts[0] - 10**9)}.jpg"}
        )
    )
    # ステーションを受けた時刻を、作り物の時計にそろえる（それより前に届いたタップは使わない）
    for meta in (tmp_path / "data").glob("*/*/meta.json"):
        os.utime(meta, (T0 - 10.0, T0 - 10.0))
    clock = Clock()
    fake = FakeSlam(sc, truth)
    svc = SlamService(
        tmp_path / "data",
        tmp_path / "queue",
        submap_size=8,
        flush_s=1.5,
        slam_factory=lambda: fake,
        clock=clock,
    )
    while svc.poll():
        pass
    assert len(svc.kfs) == 24 + 2
    clock.t += 2.0  # 新しいコマが来ないので、残りも回す
    while svc.poll():
        pass
    assert not svc.pending
    # 撮った順に submap_size ずつ
    assert [len(c) for c in fake.calls] == [8, 8, 8, 2]
    assert all(a < b for c in fake.calls for a, b in zip(c, c[1:], strict=False))

    out = session / "slam"
    rows = read_trajectory(out)
    assert len(rows) == 26
    for r in rows:
        tx, ty, tyaw = STATIONS[sc.frames[truth[r["file"]]].station]
        assert (r["x"], r["y"], r["z"]) == pytest.approx((tx, ty, H), abs=1e-3)
        assert math.remainder(r["yaw"] - tyaw, 2 * math.pi) == pytest.approx(0, abs=1e-3)
    tap_rows = [r for r in rows if r["source"] == "tap"]
    assert len(tap_rows) == 2 and tap_rows[0]["station"] is None
    assert tap_rows[0]["latency_s"] == pytest.approx(1.0)  # 届いて 1 秒後に回した

    pts = np.load(out / "points.npz")
    assert pts["xyz"].dtype == np.float32 and pts["rgb"].dtype == np.uint8
    assert float(pts["voxel_m"]) == 0.03
    assert np.abs(pts["xyz"][:, 2]).min() < 0.02  # 床の点は z ≈ 0

    status = json.loads((out / "status.json").read_text())
    assert status["session"] == "explore_test" and status["state"] == "ok"
    assert status["keyframes"] == 26 and status["submaps"] == 4 and status["segments"] == 1
    assert status["alignment"]["method"] == "stations"
    assert status["alignment"]["rmse_m"] == pytest.approx(0, abs=1e-3)
    assert status["pose"]["t_capture_ns"] == rows[-1]["t_capture_ns"]

    # 猫の所は灰色で塗ってから SLAM に渡す
    cat = cv2.imread(str(out / "frames" / tap_rows[0]["file"]))
    assert abs(int(cat[10, 10, 0]) - 127) <= 2 and abs(int(cat[400, 600, 0]) - 200) <= 2
    lines = (out / "frames.jsonl").read_text().splitlines()
    assert len(lines) == 26 and json.loads(lines[1])["boxes"] == []

    # 使わないので終えて起動し直しても、同じセッションのそれまでのコマを読み直す
    # （タップの置き場からは消えていても、frames/ に写したものがある）
    for p in (tmp_path / "queue").rglob("*"):
        if p.is_file():
            p.unlink()
    again = SlamService(
        tmp_path / "data", tmp_path / "queue", submap_size=8, flush_s=1.5,
        slam_factory=lambda: FakeSlam(sc, truth), clock=clock,
    )  # fmt: skip
    again.start(session)
    assert len(again.kfs) == 26 and len(again.pending) == 26


def test_service_starts_a_new_segment_when_tracking_is_lost(tmp_path):
    sc = Scene(STATIONS)
    session, truth, _ = make_session(tmp_path, sc)
    clock = Clock()
    fake = FakeSlam(sc, truth, fail_at=2)
    svc = SlamService(
        tmp_path / "data", tmp_path / "queue", submap_size=6, slam_factory=lambda: fake, clock=clock
    )
    svc.poll()
    svc.poll()
    status = json.loads((session / "slam" / "status.json").read_text())
    assert status["segments"] == 2 and status["failures"] == 1
    assert status["state"] == "lost"
    svc.drain()
    rows = read_trajectory(session / "slam")
    assert sorted({r["segment"] for r in rows}) == [0, 1]
    assert len(rows) == 24 - 6  # 崩れたサブマップの 6 コマは位置が無い
    status = json.loads((session / "slam" / "status.json").read_text())
    assert status["state"] == "ok"  # 新しい区間もステーションで合わせられた


def test_slam_waits_while_the_map_server_runs_vggt(tmp_path, monkeypatch):
    from app import snoo_map_server, snoo_slam

    busy = tmp_path / "map_busy"
    monkeypatch.setattr(snoo_map_server, "GPU_BUSY_FILE", str(busy))
    monkeypatch.setattr(snoo_slam, "MAP_BUSY_FILE", str(busy))
    assert not snoo_slam.map_is_busy()
    with snoo_map_server.gpu_busy():
        assert busy.exists() and snoo_slam.map_is_busy()
    assert not busy.exists() and not snoo_slam.map_is_busy()
    busy.write_text("1")  # 落ちて残った古いファイルは見ない
    old = time.time() - snoo_slam.MAP_BUSY_MAX_S - 1
    os.utime(busy, (old, old))
    assert not snoo_slam.map_is_busy()
