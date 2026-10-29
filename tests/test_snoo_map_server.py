"""app/snoo_map_server.py を、VGGT の代わりに作り物の部屋（tests/test_snoo_mapper.py の _scene）で
試す。GPU は使わない。
"""

import json
import math
from functools import cache

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_snoo_mapper import PANS, STATIONS, TILTS, _scene

from app.roommap import Pose
from app.snoo import parse_allow
from app.snoo_map_server import (
    Recon,
    choose_window,
    create_app,
    save_station,
    to_session,
    ultrasonic_cm,
)

PI = ("192.168.0.188", 40000)
JPEG = cv2.imencode(".jpg", np.full((48, 64, 3), 128, np.uint8))[1].tobytes()


@cache
def _scene_for(n: int):
    return _scene(STATIONS[:n])


def fake_reconstruct(refs, npz, marks):
    """VGGT の代わり。作り物の部屋の点を、サーバーが選んだコマと同じ並びで返す。"""
    n = max(r.station for r in refs) + 1
    _, want, ext, points, conf = _scene_for(n)
    got = [(r.station, r.pan_deg, r.tilt_deg) for r in refs]
    assert got == [(w.station, w.pan_deg, w.tilt_deg) for w in want]
    marks["vggt"] = 0.0
    return Recon(ext, points, conf, None)


def no_reconstruct(refs, npz, marks):
    raise AssertionError("VGGT を回してはいけない")


def _client(tmp_path, reconstruct=fake_reconstruct, client=PI):
    app = create_app(tmp_path, parse_allow("192.168.0.188,127.0.0.1"), reconstruct)
    return TestClient(app, client=client)


def _frames():
    """作り物の部屋と同じパン・チルトのコマと、走りながら撮ったコマ 1 つ。"""
    frames = [
        {
            "file": f"{i:02d}_pan{p:+04d}_tilt{t:+03d}.jpg",
            "t_capture_ns": 1000 + i,
            "pan_deg": float(p),
            "tilt_deg": float(t),
            "ultrasonic_cm": None,
        }
        for i, (t, p) in enumerate((t, p) for t in TILTS for p in PANS)
    ]
    frames.append(
        {
            "file": "drive_000.jpg",
            "t_capture_ns": 999,
            "pan_deg": 0.0,
            "tilt_deg": 0.0,
            "ultrasonic_cm": None,
            "steer_deg": 5.0,
            "speed": 12,
            "moving": True,
        }
    )
    return frames


def _meta(station_id=0, session="explore_20261029_201500", **kw):
    meta = {
        "schema_version": 2,
        "camera": "picarx-ov5647",
        "tilt_offset_deg": 0.0,
        "frames": _frames(),
        "session_id": session,
        "station_id": station_id,
        "prev_step": None,
        "battery_v": 7.8,
        "ultrasonic_cm": None,
        "camera_height_cm": 10,
    }
    meta.update(kw)
    return meta


def _post(c, meta, images=None):
    if images is None:
        images = _all_images(meta)
    files = [("images", (name, JPEG, "image/jpeg")) for name in images]
    data = {"meta": json.dumps(meta)}
    return c.post("/v1/snoo/map/station", data=data, files=files)


def test_two_stations_return_the_next_step(tmp_path):
    c = _client(tmp_path)
    r = _post(c, _meta(0))
    assert r.status_code == 200, r.text
    assert r.json()["action"] == "move"
    assert r.json()["station_index"] == 0

    r = _post(c, _meta(1, prev_step={"steer_deg": 0, "speed": 12, "seconds": 1.2}))
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["action"] == "move"
    assert isinstance(a["steer_deg"], float) and -30 < a["steer_deg"] < 30
    assert a["speed"] == 12
    assert 0 < a["distance_cm"] <= 40
    assert a["station_index"] == 1
    assert a["map_url"] == "/v1/snoo/map/explore_20261029_201500/topdown.png"
    # 2 か所目の位置（セッションの座標。最初のステーションが原点）
    assert (a["pose"]["x"], a["pose"]["y"]) == pytest.approx((0.3, 0.1), abs=0.02)
    assert a["pose"]["yaw_deg"] == pytest.approx(20, abs=1.5)

    r = c.get(a["map_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR) is not None
    room = c.get("/v1/snoo/map/explore_20261029_201500/room.json").json()
    assert [s["name"] for s in room["stations"]] == ["000", "001"]
    assert room["window"]["stations"] == [0, 1]
    plan = c.get("/v1/snoo/map/explore_20261029_201500/plan.json").json()
    assert plan["action"] == "move" and "total" not in plan["timing_s"]
    assert "vggt" in plan["timing_s"]


def test_ultrasonic_is_the_median_of_the_station_readings(tmp_path):
    session = tmp_path / "s"
    m = _meta(0, ultrasonic_cm=34.5)
    for f, v in zip(m["frames"], [34.4, 61.3, 34.5, -1, 34.6, None, 34.5, 34.4], strict=False):
        f["ultrasonic_cm"] = v
    m["frames"][-1]["ultrasonic_cm"] = 150.0  # 走りながら測った値は使わない
    d = save_station(session, 0, m, {})
    assert ultrasonic_cm(d) == pytest.approx(34.5)
    d = save_station(session, 1, _meta(1), {})
    assert ultrasonic_cm(d) is None


def test_station_is_saved(tmp_path):
    c = _client(tmp_path)
    meta = _meta(3)
    assert _post(c, meta).status_code == 200
    d = tmp_path / "explore_20261029_201500" / "003"
    saved = json.loads((d / "meta.json").read_text())
    assert saved == meta
    # 走りながら撮ったコマも残す（VGGT には入れない。入れれば fake_reconstruct が落ちる）
    for f in meta["frames"]:
        assert (d / f["file"]).read_bytes() == JPEG
    poses = json.loads((tmp_path / "explore_20261029_201500" / "poses.json").read_text())
    assert poses == {"3": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}}
    # 同じ station_id を送り直したら置き換える
    assert _post(c, meta, [f["file"] for f in meta["frames"][:-1]]).status_code == 200
    assert not (d / "drive_000.jpg").exists()


def _set(**kw):
    return lambda m: {**m, **kw}


def _all_images(m):
    return [f["file"] for f in m["frames"]]


@pytest.mark.parametrize(
    "change, images, words",
    [
        (lambda m: "{not json", [], "JSON"),
        (lambda m: {"station_id": 0}, [], "session_id"),
        (_set(frames=[]), [], "frames"),
        (_set(station_id="3"), _all_images, "station_id"),
        (_set(station_id=1000), _all_images, "station_id"),
        (_set(battery_v="7.8"), _all_images, "battery_v"),
        (_set(camera_height_cm=0), _all_images, "camera_height_cm"),
        (_set(ultrasonic_cm="34"), _all_images, "ultrasonic_cm"),
        (lambda m: {**m, "frames": m["frames"][-1:]}, _all_images, "止まって撮った"),
        (lambda m: m, lambda m: ["00_pan-035_tilt-15.jpg"], "画像が無い"),
        (lambda m: m, lambda m: _all_images(m) + ["other.jpg"], "frames に無い"),
    ],
)
def test_bad_input_is_400(tmp_path, change, images, words):
    c = _client(tmp_path, no_reconstruct)
    m = _meta()
    names = images(m) if callable(images) else images
    files = [("images", (name, JPEG, "image/jpeg")) for name in names]
    meta = change(m)
    data = {"meta": meta if isinstance(meta, str) else json.dumps(meta)}
    r = c.post("/v1/snoo/map/station", data=data, files=files)
    assert r.status_code == 400, r.text
    assert words in r.json()["detail"]
    assert not any(tmp_path.iterdir())


def test_missing_images_and_broken_image_are_400(tmp_path):
    c = _client(tmp_path, no_reconstruct)
    r = c.post("/v1/snoo/map/station", files=[("images", ("a.jpg", JPEG, "image/jpeg"))])
    assert r.status_code == 400 and "meta が無い" in r.json()["detail"]
    r = c.post("/v1/snoo/map/station", data={"meta": json.dumps(_meta())})
    assert r.status_code == 400 and "images" in r.json()["detail"]
    m = _meta()
    files = [("images", (f["file"], b"not a jpeg", "image/jpeg")) for f in m["frames"]]
    r = c.post("/v1/snoo/map/station", data={"meta": json.dumps(m)}, files=files)
    assert r.status_code == 400 and "読めない" in r.json()["detail"]


def test_too_large_image_is_413(tmp_path):
    c = _client(tmp_path, no_reconstruct)
    m = _meta()
    big = JPEG + b"\0" * (2 * 1024 * 1024)
    files = [("images", (f["file"], big, "image/jpeg")) for f in m["frames"][:1]]
    r = c.post("/v1/snoo/map/station", data={"meta": json.dumps(m)}, files=files)
    assert r.status_code == 413


def test_other_sources_get_403(tmp_path):
    c = _client(tmp_path, no_reconstruct, client=("192.168.0.50", 40000))
    assert _post(c, _meta()).status_code == 403
    assert c.get("/v1/snoo/map/explore_20261029_201500/topdown.png").status_code == 403
    assert not any(tmp_path.iterdir())
    # IP でない送り元も断る
    c = _client(tmp_path, no_reconstruct, client=("testclient", 50000))
    assert c.get("/v1/snoo/map/x/room.json").status_code == 403


@pytest.mark.parametrize("session", ["../evil", "a/b", "..", "", "a" * 65])
def test_session_id_cannot_leave_the_data_dir(tmp_path, session):
    root = tmp_path / "sessions"
    c = _client(root, no_reconstruct)
    r = _post(c, _meta(session=session))
    assert r.status_code == 400 and "session_id" in r.json()["detail"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["../evil.jpg", "/tmp/evil.jpg", ".hidden.jpg", "meta.json"])
def test_image_names_cannot_leave_the_station_dir(tmp_path, name):
    c = _client(tmp_path, no_reconstruct)
    m = _meta()
    m["frames"][0]["file"] = name
    r = _post(c, m)
    assert r.status_code == 400 and "file" in r.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_outputs_check_the_session_id(tmp_path):
    c = _client(tmp_path, no_reconstruct)
    assert c.get("/v1/snoo/map/..%2E/room.json").status_code in (400, 404)
    assert c.get("/v1/snoo/map/a.b/room.json").status_code == 400
    assert c.get("/v1/snoo/map/abc/grid.png").status_code == 404
    assert c.get("/v1/snoo/map/abc/topdown.png").status_code == 404


def test_low_battery_stops_without_vggt(tmp_path):
    c = _client(tmp_path, no_reconstruct)
    r = _post(c, _meta(battery_v=6.7))
    assert r.json() == {"action": "done", "reason": "low_battery"}
    assert (tmp_path / "explore_20261029_201500" / "000" / "meta.json").exists()


def test_step_limit_after_30_stations(tmp_path):
    c = _client(tmp_path, no_reconstruct)
    session = tmp_path / "explore_20261029_201500"
    for i in range(29):
        save_station(session, i, _meta(i), {})
    r = _post(c, _meta(29))
    assert r.json() == {"action": "done", "reason": "step_limit"}


def test_window_keeps_stations_near_the_car():
    poses = {i: Pose(0.4 * i, 0.0, 0.0) for i in range(10)}
    poses[1] = Pose(3.2, 0.0, 0.0)  # 遠くから戻ってきた所の近く
    w = choose_window(list(range(11)), 10, poses, k=4)
    assert w == [1, 8, 9, 10]
    # 位置の分からないステーションは入れない
    assert choose_window([0, 1, 2], 2, {0: Pose(0, 0, 0)}, k=4) == [0, 2]
    assert choose_window([0], 0, {}, k=4) == [0]


def test_to_session_turns_and_shifts():
    anchor = Pose(1.0, 2.0, math.radians(90))
    p = to_session(anchor, Pose(0.5, 0.0, math.radians(-30)))
    assert (p.x, p.y) == pytest.approx((1.0, 2.5))
    assert math.degrees(p.yaw) == pytest.approx(60)
    p = to_session(Pose(0, 0, math.radians(170)), Pose(0, 0, math.radians(20)))
    assert math.degrees(p.yaw) == pytest.approx(-170)
