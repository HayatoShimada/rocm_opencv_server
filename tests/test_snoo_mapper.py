"""app/snoo_mapper.py を、VGGT の代わりに作り物の部屋（箱の部屋の点とカメラ）で試す。
GPU は使わない。
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.roommap import FREE, OCCUPIED, Pose
from app.snoo_mapper import (
    PNG_VALUE,
    FrameRef,
    MapParams,
    Station,
    StationResult,
    build_map,
    decide,
    select_frames,
    write_outputs,
)

CAM_H = 0.10
# VGGT の長さ 1 が何 m か（VGGT は大きさを知らない。これを当て直せるか見る）
TRUE_SCALE = 2.7
STATIONS = [Pose(0.0, 0.0, 0.0), Pose(0.3, 0.1, math.radians(20))]
PANS = [-35, -15, 5, 25]
TILTS = [-15, 0]
BOX = (1.0, 1.2, -0.6, -0.4, 0.08)  # x0, x1, y0, y1, 高さ。床に置いた箱
TABLE = (1.5, 2.2, 0.3, 0.9, 0.7)  # 机の天板（車は下をくぐれる）


def _room_points(rng) -> np.ndarray:
    """4 m x 3 m の部屋（x -1..3、y -1.5..1.5）の床・壁・箱・天板の点（m、z が上）。"""
    s = 0.0125
    gx, gy = np.meshgrid(np.arange(-1, 3, s), np.arange(-1.5, 1.5, s), indexing="ij")
    parts = [np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)]
    wz = np.arange(0, 0.5, s)
    for y in (-1.5, 1.5):
        a, b = np.meshgrid(np.arange(-1, 3, s), wz, indexing="ij")
        parts.append(np.stack([a.ravel(), np.full(a.size, y), b.ravel()], 1))
    a, b = np.meshgrid(np.arange(-1.5, 1.5, s), wz, indexing="ij")
    parts.append(np.stack([np.full(a.size, 3.0), a.ravel(), b.ravel()], 1))
    x0, x1, y0, y1, h = BOX
    a, b = np.meshgrid(np.arange(x0, x1, s), np.arange(y0, y1, s), indexing="ij")
    for z in np.arange(0.03, h + 1e-9, 0.025):
        parts.append(np.stack([a.ravel(), b.ravel(), np.full(a.size, z)], 1))
    x0, x1, y0, y1, h = TABLE
    a, b = np.meshgrid(np.arange(x0, x1, s), np.arange(y0, y1, s), indexing="ij")
    parts.append(np.stack([a.ravel(), b.ravel(), np.full(a.size, h)], 1))
    pts = np.concatenate(parts)
    return pts + rng.normal(0, 0.002, pts.shape)


def _cam_rotation(yaw: float, tilt: float) -> np.ndarray:
    """世界（x 前・y 左・z 上）→ カメラ（x 右・y 下・z 前）。
    yaw は光軸の向き（左回り +）、tilt は上が +。
    """
    f = np.array([math.cos(yaw) * math.cos(tilt), math.sin(yaw) * math.cos(tilt), math.sin(tilt)])
    r = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    return np.stack([r, np.cross(f, r), f])


def _scene(stations=STATIONS):
    """VGGT が返しそうなもの。最初のコマのカメラを世界にし、大きさを 1/TRUE_SCALE にした
    点とカメラ。
    """
    rng = np.random.default_rng(1)
    world = _room_points(rng)
    refs, rots, centers = [], [], []
    for si, st in enumerate(stations):
        for tilt in TILTS:
            for pan in PANS:
                # パンは右が +。車の向きから右へ pan° 回した向きを見る
                rots.append(_cam_rotation(st.yaw - math.radians(pan), math.radians(tilt)))
                centers.append(np.array([st.x, st.y, CAM_H]))
                refs.append(FrameRef(None, si, float(pan), float(tilt)))
    R0, C0 = rots[0], centers[0]
    ext = np.stack(
        [
            np.concatenate([R @ R0.T, (R @ (C0 - C) / TRUE_SCALE)[:, None]], 1)
            for R, C in zip(rots, centers, strict=True)
        ]
    )
    # 各コマに写る点：そのカメラの水平の画角（±27°）の内側
    per_frame = []
    for R, C, ref in zip(rots, centers, refs, strict=True):
        cam = (world - C) @ R.T
        seen = (cam[:, 2] > 0.05) & (np.abs(np.arctan2(cam[:, 0], cam[:, 2])) < math.radians(27))
        seen &= np.abs(np.arctan2(cam[:, 1], cam[:, 2])) < math.radians(20)
        per_frame.append((world[seen] - C0) @ R0.T / TRUE_SCALE)
        assert ref.station in (0, 1)
    n = max(len(p) for p in per_frame)
    points = np.full((len(refs), n, 3), np.nan, np.float32)
    conf = np.zeros((len(refs), n), np.float32)
    for i, p in enumerate(per_frame):
        points[i, : len(p)] = p
        conf[i, : len(p)] = 1 + rng.random(len(p)) * 5
    station_objs = [Station(Path(f"st{i}"), []) for i in range(len(stations))]
    return station_objs, refs, ext, points, conf


@pytest.fixture(scope="module")
def result():
    stations, refs, ext, points, conf = _scene()
    return build_map(stations, refs, ext, points, conf, MapParams(camera_height_m=CAM_H))


def test_scale_and_floor(result):
    assert result.world.scale == pytest.approx(TRUE_SCALE, rel=0.02)
    for s in result.stations:
        assert s.height_m == pytest.approx(CAM_H, abs=0.005)
        assert s.spread_m < 0.01


def test_station_poses(result):
    for got, want in zip(result.stations, STATIONS, strict=True):
        assert (got.pose.x, got.pose.y) == pytest.approx((want.x, want.y), abs=0.02)
        assert math.degrees(got.pose.yaw - want.yaw) == pytest.approx(0, abs=1.5)


def test_grid_layers(result):
    g, state = result.grid, result.grid.state()
    x0, x1, y0, y1, _ = BOX
    assert state[g.cell((x0 + x1) / 2, (y0 + y1) / 2)] == OCCUPIED
    assert state[g.cell(0.7, 0.0)] == FREE
    assert state[g.cell(1.0, 0.5)] == FREE
    x0, x1, y0, y1, _ = TABLE
    tx, ty = g.cell((x0 + x1) / 2, (y0 + y1) / 2)
    assert state[tx, ty] != OCCUPIED  # 天板は走れるかには使わない
    assert result.high[tx, ty] > 0
    assert result.high[g.cell(0.7, 0.0)] == 0


def test_next_step(result):
    a = result.action
    assert a["action"] == "move"
    assert a["speed"] == 12
    assert 0 < a["distance_cm"] <= 40
    assert -30 < a["steer_deg"] < 30
    assert a["pose"]["x"] == pytest.approx(0.3, abs=0.02)


def test_outputs(result, tmp_path):
    write_outputs(result, tmp_path)
    room = json.loads((tmp_path / "room.json").read_text())
    assert room["resolution_cm"] == 5
    assert room["size"] == [result.grid.n, result.grid.n]
    assert room["origin"] == [result.grid.origin, result.grid.origin]
    assert room["start_pose"] == {"x": 0, "y": 0, "yaw_deg": 0}
    assert [s["name"] for s in room["stations"]] == ["st0", "st1"]
    layers = room["layers"]
    assert layers["static"]["png"] == "grid.png"
    for k in ("movable", "occluders", "snoo_spots"):
        assert layers[k] == []
    img = cv2.imread(str(tmp_path / "grid.png"), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (result.grid.n, result.grid.n)
    # 行 0 が +y の端、列 0 が -x の端
    ix, iy = result.grid.cell(1.1, -0.5)
    assert img[result.grid.n - 1 - iy, ix] == PNG_VALUE[OCCUPIED]
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["action"] == "move" and len(plan["plan"]["path"]) > 1
    assert cv2.imread(str(tmp_path / "topdown.png")) is not None


def test_floor_ignores_a_big_table_above_the_camera():
    """天板の点がいくら多くても、カメラより上の平面は床にしない。"""
    stations, refs, ext, points, conf = _scene(STATIONS[:1])
    r = build_map(stations, refs, ext, points, conf, MapParams(camera_height_m=CAM_H))
    assert r.world.scale == pytest.approx(TRUE_SCALE, rel=0.02)


def test_unknown_height_station_is_measured():
    """高さの分からないステーション（None）は、ほかで決めた大きさで高さを測る。"""
    stations, refs, ext, points, conf = _scene()
    r = build_map(stations, refs, ext, points, conf, MapParams(camera_height_m=[CAM_H, None]))
    assert r.stations[1].height_m == pytest.approx(CAM_H, abs=0.005)


def _scan(tilts, pans):
    return [
        {"file": f"{i:02d}.jpg", "pan_deg": float(p), "tilt_deg": float(t)}
        for i, (t, p) in enumerate((t, p) for t in tilts for p in pans)
    ]


PICARX_PANS = [-35, -25, -15, -5, 5, 15, 25, 35]


def test_select_frames_drops_the_upward_row_and_thins_pans():
    st = Station(Path("a"), _scan([-15, 0, 15], PICARX_PANS))
    refs = select_frames([st], 48)
    assert len(refs) == 16 and all(r.tilt_deg <= 0 for r in refs)
    four = select_frames([Station(Path(str(i)), st.frames) for i in range(4)], 48)
    assert len(four) <= 48
    first = [r for r in four if r.station == 0]
    assert len(first) == 8 and {r.tilt_deg for r in first} == {-15.0, 0.0}
    # 向きを出すのに使うパン 0 に近いコマ（-5）は残る
    assert any(r.tilt_deg == 0 and r.pan_deg == -5 for r in first)


def test_select_frames_keeps_the_latest_station_whole_and_thins_the_older_ones():
    pans = [-30, -15, 0, 15, 30]  # 3 回目の実走の首振り
    frames = _scan([-15, 0], pans)
    sts = [Station(Path(str(i)), frames) for i in range(4)]
    refs = select_frames(sts, 48, old_frames=6)
    for i in range(3):
        old = [r for r in refs if r.station == i]
        assert sorted({r.pan_deg for r in old}) == [-30, 0, 30] and len(old) == 6
    assert len([r for r in refs if r.station == 3]) == 10  # 最後は全部


def test_ultrasonic_marks_an_obstacle_the_camera_did_not_see(result):
    """VGGT が見落とした物（白い壁など）を、超音波で格子に足す。"""
    stations, refs, ext, points, conf = _scene()
    params = MapParams(camera_height_m=CAM_H, ultrasonic_cm=[None, 40])
    r = build_map(stations, refs, ext, points, conf, params)
    g, state = r.grid, r.grid.state()
    last = r.stations[-1].pose
    d = 0.05 + 0.40  # 超音波（カメラの 5 cm 前）から 40 cm
    ahead = g.cell(last.x + d * math.cos(last.yaw), last.y + d * math.sin(last.yaw))
    # 作り物の部屋ではカメラがそこに床を 2 回見ているので、障害物 1 回と足して「分からない」になる。
    # それでも床ではないので、車はそこへ行かない
    assert state[ahead] != FREE
    assert result.grid.state()[ahead] == FREE  # 超音波が無ければ床
    with pytest.raises(ValueError):
        build_map(stations, refs, ext, points, conf, MapParams(CAM_H, ultrasonic_cm=[1.0]))


def test_decide_returns_segments_only_when_asked(result):
    heights = [CAM_H, CAM_H]
    a, plan = decide(result.grid, result.stations, MapParams(segments=True), heights)
    assert a["action"] == "move"
    segs = a["segments"]
    assert 1 <= len(segs) <= 3
    assert (a["steer_deg"], a["distance_cm"]) == (segs[0]["steer_deg"], segs[0]["distance_cm"])
    assert sum(x["distance_cm"] for x in segs) <= 40.05
    assert "segments" not in result.action  # 頼まなければ今までどおり


def test_decide_stops_on_step_limit_and_bad_registration(result):
    g = result.grid
    st = result.stations
    a, _ = decide(g, st, MapParams(max_stations=2), [CAM_H, CAM_H])
    assert a == {"action": "done", "reason": "step_limit"}
    bad = [StationResult("x", Pose(0, 0, 0), CAM_H, 0.3, 8)]
    a, _ = decide(g, bad, MapParams(), [CAM_H])
    assert a["action"] == "done" and a["reason"].startswith("bad_registration")


def test_observations_add_up_to_the_same_grid_and_move_with_the_anchor(tmp_path):
    """ステーションごとに見たもの（セッションの座標）を足すと、窓で作った格子と同じになる。
    窓の原点が (1, 2, 90°) なら、見たものも同じだけずれて回る。
    """
    from app.snoo_mapper import Observation, session_grid

    stations, refs, ext, points, conf = _scene()
    params = MapParams(camera_height_m=CAM_H, ultrasonic_cm=[80.0, None])
    r = build_map(stations, refs, ext, points, conf, params, anchor=Pose(0, 0, 0), plan=False)
    assert r.action == {} and r.plan is None
    grid, _ = session_grid(r.observations)
    assert (grid.state() == r.grid.state()).all()
    assert r.observations[0].sonar_m == pytest.approx(0.8)
    r.observations[1].save(tmp_path / "obs.npz")
    back = Observation.load(tmp_path / "obs.npz")
    assert back.pose == pytest.approx(r.observations[1].pose) and back.sonar_m is None
    assert np.allclose(back.free, r.observations[1].free)

    moved = build_map(
        stations, refs, ext, points, conf, params, anchor=Pose(1.0, 2.0, math.pi / 2), plan=False
    )
    o, p = moved.observations[1], r.observations[1]
    assert (o.pose.x, o.pose.y) == pytest.approx((1.0 - p.pose.y, 2.0 + p.pose.x), abs=1e-6)
    assert len(o.free) == pytest.approx(len(p.free), rel=0.05)  # 回してもセルが抜けない


def test_decide_turns_instead_of_looping_when_the_target_is_behind():
    """見ていない所が真後ろだけなら、前へ輪を描かず、切り返す（Pi が reverse を受けるとき）。"""
    from app.roommap import Grid

    g = Grid(size_m=6.0)
    xs = np.arange(-0.2, 1.6, 0.025)
    gx, gy = np.meshgrid(xs, np.arange(-0.8, 0.8, 0.025))
    g.mark_free(gx.ravel(), gy.ravel())  # 前と足元だけ床。後ろ（x < -0.2）は見ていない
    wall = np.arange(-0.8, 0.8, 0.02)
    g.mark_occupied(np.full_like(wall, 1.6), wall)
    g.mark_occupied(np.arange(-0.2, 1.6, 0.02), np.full(90, 0.8))
    g.mark_occupied(np.arange(-0.2, 1.6, 0.02), np.full(90, -0.8))
    st = [StationResult("000", Pose(0.3, 0.0, 0.0), CAM_H, 0.0, 8)]
    a, plan = decide(g, st, MapParams(segments=True, reverse=True, random_p=0.0), [CAM_H])
    assert a["action"] == "move" and a.get("turn") and a["reverse"] is True and plan is None
    a, _ = decide(g, st, MapParams(segments=True, random_p=0.0), [CAM_H])
    assert not a.get("turn")  # reverse を受けない Pi には今までどおり


def _slam_dir(tmp_path, state="ok", rmse=0.03, method="stations"):
    import json as _json

    d = tmp_path / "slam"
    d.mkdir()
    (d / "status.json").write_text(
        _json.dumps({"state": state, "alignment": {"method": method, "rmse_m": rmse}})
    )
    xs = np.arange(0.0, 1.0, 0.03)
    floor = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1)
    wall = np.array([[1.2, 0.0, 0.06], [1.2, 0.03, 0.06]])
    table = np.array([[0.5, 0.5, 0.6]])
    xyz = np.concatenate([floor, wall, table]).astype(np.float32)
    count = np.full(len(xyz), 3)
    count[-1] = 1  # 点が 1 つだけのボクセルは使わない
    np.savez(d / "points.npz", xyz=xyz, count=count)
    (d / "trajectory.jsonl").write_text(_json.dumps({"x": -0.3, "y": 0.0}) + "\n")
    return d


def test_mark_slam_adds_floor_obstacles_and_the_driven_path(tmp_path):
    from app.roommap import FREE, OCCUPIED, UNKNOWN, Grid
    from app.snoo_mapper import MapParams, mark_slam

    g = Grid()
    high = np.zeros_like(g.logodds, np.int32)
    info = mark_slam(g, high, _slam_dir(tmp_path), MapParams())
    s = g.state()
    assert info["used"] and info["obstacle"] == 2
    assert s[g.cell(0.5, 0.0)] == FREE
    assert s[g.cell(1.2, 0.0)] == OCCUPIED
    assert s[g.cell(-0.3, 0.0)] == FREE  # 走った所
    assert high.sum() == 0  # 点が少ないボクセルは使わない
    assert s[g.cell(0.5, 1.0)] == UNKNOWN


def test_mark_slam_ignores_a_lost_or_misaligned_slam(tmp_path):
    from app.roommap import Grid
    from app.snoo_mapper import MapParams, mark_slam

    for state, rmse, method in (("lost", 0.03, "stations"), ("ok", 0.2, "stations"),
                                ("ok", 0.0, "heading")):  # fmt: skip
        g = Grid()
        sub = tmp_path / f"{state}{rmse}{method}"
        sub.mkdir()
        info = mark_slam(g, np.zeros_like(g.logodds, np.int32), _slam_dir(sub, state, rmse, method),
                         MapParams())  # fmt: skip
        assert not info["used"] and not g.logodds.any()
    assert not mark_slam(Grid(), np.zeros((240, 240), np.int32), tmp_path / "none", MapParams())[
        "used"
    ]
