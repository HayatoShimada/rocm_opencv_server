"""app/snoo_map_server.py を、VGGT の代わりに作り物の部屋（tests/test_snoo_mapper.py の _scene）で
試す。GPU は使わない。
"""

import json
import math
import time
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
    steps_driven,
    to_session,
    ultrasonic_cm,
)

PI = ("192.168.0.188", 40000)
JPEG = cv2.imencode(".jpg", np.full((48, 64, 3), 128, np.uint8))[1].tobytes()


@cache
def _scene_for(n: int):
    return _scene(STATIONS[:n])


def fake_reconstruct(refs, npz, marks):
    """VGGT の代わり。作り物の部屋の点を、サーバーが選んだコマと同じ並びで返す
    （前のステーションはコマを間引くので、選んだコマだけ）。
    """
    n = max(r.station for r in refs) + 1
    _, want, ext, points, conf = _scene_for(n)
    index = {(w.station, w.pan_deg, w.tilt_deg): i for i, w in enumerate(want)}
    got = [index[(r.station, r.pan_deg, r.tilt_deg)] for r in refs]
    assert sorted(set(r.station for r in refs)) == list(range(n))
    marks["vggt"] = 0.0
    return Recon(ext[got], points[got], conf[got], None)


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


@pytest.fixture
def window_poses(monkeypatch):
    """位置のグラフ（SNOO_POSE_GRAPH）を使わず、窓の VGGT の位置だけで決める（今までの決め方）。
    ここのテストの prev_step はわざと本当の 1 歩と違えてあり、グラフは走った辺も重ねるので、
    窓の位置そのものを見るテストではグラフを止める（グラフは test_snoo_map_server_graph.py）。
    """
    import app.snoo_map_server as server

    monkeypatch.setattr(server, "POSE_GRAPH", False)


def test_two_stations_return_the_next_step(tmp_path, window_poses):
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
    # 間取りはセッションの座標で、全部のステーションの見たもの（obs.npz）を足したもの
    assert room["window"]["map"] == "session"
    assert room["window"]["origin_pose"] == {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    session = tmp_path / "explore_20261029_201500"
    assert (session / "000" / "obs.npz").is_file() and (session / "001" / "obs.npz").is_file()
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


def test_segments_only_for_a_pi_that_supports_them(tmp_path):
    old = _client(tmp_path / "old")
    _post(old, _meta(0))
    a_old = _post(old, _meta(1, prev_step={"steer_deg": 0, "speed": 12, "seconds": 1.2})).json()
    assert a_old["action"] == "move"
    assert "segments" not in a_old and "reverse" not in a_old  # 古い Pi への返事は今までどおり

    new = _client(tmp_path / "new")
    _post(new, _meta(0, supports=["reverse", "segments"]))
    m = _meta(1, prev_step={"steer_deg": 0, "speed": 12, "seconds": 1.2})
    a = _post(new, {**m, "supports": ["reverse", "segments"]}).json()
    assert a["action"] == "move"
    segs = a["segments"]
    assert 1 <= len(segs) <= 3
    assert sum(x["distance_cm"] for x in segs if not a.get("reverse")) <= 40.05
    assert (a["steer_deg"], a["distance_cm"]) == (segs[0]["steer_deg"], segs[0]["distance_cm"])
    assert all("reverse" not in x for x in segs)  # gear_segments を書かない Pi には 1 歩 1 方向

    gear = _client(tmp_path / "gear")
    can = ["reverse", "segments", "long_step", "gear_segments"]
    _post(gear, _meta(0, supports=can))
    a = _post(gear, {**m, "supports": can}).json()
    assert a["action"] == "move" and 1 <= len(a["segments"]) <= 5
    for x in a["segments"]:
        assert isinstance(x["reverse"], bool) and set(x["end_pose"]) == {"x", "y", "yaw_deg"}
        assert x["min_clear_cm"] is None or x["min_clear_cm"] > 0
    assert a["reverse_speed"] < a["speed"] or a.get("reverse")


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
        (_set(supports="reverse"), _all_images, "supports"),
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


def test_step_limit_after_max_stations(tmp_path, monkeypatch):
    import app.snoo_map_server as server

    # 1 セッションの上限は SESSION_MAX_STATIONS（止まらずに探り続けるので 2000）。小さくして確かめる
    MAX_STATIONS = 5
    monkeypatch.setattr(server, "SESSION_MAX_STATIONS", MAX_STATIONS)
    c = _client(tmp_path, no_reconstruct)
    session = tmp_path / "explore_20261029_201500"
    for i in range(MAX_STATIONS - 1):
        save_station(session, i, _meta(i), {})
    r = _post(c, _meta(MAX_STATIONS - 1))
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


# --- 走った 1 歩（prev_step）を読む ---


# 作り物の部屋の 1 か所目 (0, 0, 0°) から 2 か所目 (0.3, 0.1, 20°) へは、ハンドル -8.9° の
# 弧で 31 cm ほど。Pi の見積もり（est_distance_cm）はわざと短くしておく
PREV = {"steer_deg": -8.9, "speed": 12, "seconds": 0.9, "est_distance_cm": 18.0, "reverse": False}


def _line(tmp_path, steps, replies=None):
    """x 軸の上を走ったセッション。steps は (x, prev_step) の並び（最初は prev_step 無し）。
    replies はステーションの番号と返した返事（reply.json）。"""
    session = tmp_path / "line"
    poses = {}
    for i, (x, prev) in enumerate(steps):
        save_station(session, i, _meta(i, session="line", prev_step=prev), {})
        if x is not None:
            poses[i] = Pose(x, 0.0, 0.0)
    for i, reply in (replies or {}).items():
        (session / f"{i:03d}" / "reply.json").write_text(json.dumps(reply))
    return session, poses


def _step(cm, reverse=None, **kw):
    p = {"steer_deg": 0.0, "speed": 12, "seconds": 1.0, "est_distance_cm": cm, **kw}
    if reverse is not None:
        p["reverse"] = reverse
    return p


def test_driven_steps_use_what_the_car_actually_drove(tmp_path):
    # VGGT では 12 cm しか進んでいない（Pi は 40 cm と見積もった）→ 12 cm
    session, poses = _line(tmp_path / "a", [(0.0, None), (0.12, _step(40, False))])
    ((_, moves),) = steps_driven(session, 1, poses)[1:]
    assert moves == [(0.0, pytest.approx(0.12, abs=0.011), False)]
    # 位置が分からなければ Pi の見積もり（est_distance_cm、無ければ 秒 × speed × 1.7 cm/s）
    session, _ = _line(tmp_path / "b", [(0.0, None), (None, _step(None, False, seconds=0.5))])
    ((_, moves),) = steps_driven(session, 1, {0: Pose(0, 0, 0)})[1:]
    assert moves == [(0.0, pytest.approx(0.5 * 12 * 0.017), False)]
    # どう動いたか分からなければ None
    session, poses = _line(tmp_path / "c", [(0.0, None), (0.3, None)])
    assert steps_driven(session, 1, poses)[1][1] is None


def test_mixed_gear_steps_are_integrated_segment_by_segment(tmp_path):
    """前 20 cm → 後ろ 10 cm（gear_segments）。走った区切りごとの向きで位置と床を足す。"""
    from app import snoo_posegraph as pg
    from app.snoo_map_server import mark_steps_driven, odom_edge_for

    prev = {"steer_deg": 0.0, "speed": 12, "reverse": False, "est_distance_cm": 30.0,
            "segments": [{"steer_deg": 0.0, "distance_cm": 20.0, "reverse": False},
                         {"steer_deg": 0.0, "distance_cm": 10.0, "reverse": True}]}  # fmt: skip
    session, poses = _line(tmp_path, [(0.0, None), (0.10, prev)])
    ((_, moves),) = steps_driven(session, 1, poses)[1:]
    assert [b for _, _, b in moves] == [False, True]
    e = odom_edge_for(session, 1)
    assert (e.z.x, e.z.y) == pytest.approx((0.10, 0.0), abs=1e-3)
    assert e.z.x == pytest.approx(pg.odom_edge_moves(0, 1, moves).z.x, abs=1e-6)
    from app.roommap import FREE, Grid

    g = Grid()
    mark_steps_driven(g, session, 1, poses)
    assert g.state()[g.cell(0.18, 0.0)] == FREE  # 前へ 20 cm 走ったときの車の下


def test_old_pi_without_the_reverse_flag_uses_the_reply_it_got(tmp_path):
    steps = [(0.0, None), (0.4, _step(40)), (0.2, _step(20))]
    session, poses = _line(tmp_path, steps, replies={1: {"action": "move", "reverse": True}})
    out = steps_driven(session, 2, poses)
    assert out[1][1][0][2] is False and out[2][1][0][2] is True


def test_each_station_keeps_its_points_and_reply_and_can_be_replayed(tmp_path):
    from app.snoo_map_server import RECON_FILE, load_recon
    from scripts.snoo_replay import replay, same_reply

    c = _client(tmp_path)
    replies = [_post(c, _meta(i)).json() for i in range(2)]
    session = tmp_path / "explore_20261029_201500"
    for i, reply in enumerate(replies):
        d = session / f"{i:03d}"
        keys, recon = load_recon(d / RECON_FILE)
        assert len(keys) == len(recon.extrinsic) and recon.points.dtype == np.float32
        assert json.loads((d / "reply.json").read_text()) == reply
    # 最後でないステーションも、残した点で決め直せる
    station, again, before, _ = replay(session, tmp_path / "out", station=0)
    assert station == 0 and before == replies[0] and same_reply(again, before)


def test_no_floor_returns_done_instead_of_500(tmp_path, monkeypatch):
    import app.snoo_mapper as mapper

    monkeypatch.setattr(mapper, "fit_floor", lambda *a, **k: None)
    c = _client(tmp_path)
    r = _post(c, _meta(0))
    assert r.status_code == 200 and r.json()["action"] == "done"


def test_idle_exit_waits_for_requests_in_flight():
    from app.snoo_map_server import VggtReconstruct

    v = VggtReconstruct(idle_s=0)  # 監視のスレッドは立てない
    v.idle_s = 0.01
    v.model = object()  # 読んだことにする
    assert v.enter()  # 受けている途中
    v.last_used -= 1.0
    assert not v._should_exit()  # 途中のリクエストがあるので終えない
    v.leave()
    v.last_used -= 1.0
    assert v._should_exit()
    assert not v.enter()  # 終えると決めた後は受けない（503）


def test_resending_the_same_station_does_not_break_the_first(tmp_path):
    import threading as _threading

    c = _client(tmp_path)
    out = []
    ts = [
        _threading.Thread(target=lambda: out.append(_post(c, _meta(0)).status_code))
        for _ in range(3)
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out == [200, 200, 200]


def test_read_meta_rereads_when_the_station_is_resent(tmp_path):
    import os as _os

    from app.snoo_map_server import read_meta

    d = tmp_path / "000"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"battery_v": 7.8}))
    assert read_meta(d)["battery_v"] == 7.8
    (d / "meta.json").write_text(json.dumps({"battery_v": 7.25}))
    st = (d / "meta.json").stat()
    _os.utime(d / "meta.json", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert read_meta(d)["battery_v"] == 7.25


def _write_meta(session, i, prev, reply=None):
    d = session / f"{i:03d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"station_id": i, "prev_step": prev}))
    if reply is not None:
        (d / "reply.json").write_text(json.dumps(reply))


def test_a_bad_registration_retraces_the_step_just_driven(tmp_path):
    from app.snoo_map_server import retrace_after_bad
    from app.snoo_mapper import MapParams

    params = MapParams(segments=True, reverse=True)
    session = tmp_path / "explore_20261030_184931"
    driven = {
        "steer_deg": 26.6, "speed": 12.0, "reverse": False, "est_distance_cm": 25.7,
        "segments": [{"steer_deg": 26.6, "distance_cm": 25.7}], "stop_reason": None,
    }  # fmt: skip
    _write_meta(session, 3, None, {"action": "move"})
    _write_meta(session, 4, driven)
    bad = {"action": "done", "reason": "bad_registration:004"}
    out = retrace_after_bad(session, 4, bad, "004", params)
    assert np.allclose([o[:2] for o in out], [(26.6, 0.257)]) and out[0][2] is True
    # ほかの理由・ほかのステーションの崩れ・続けて崩れたら戻らない
    nf = {"action": "done", "reason": "no_frontier"}
    assert retrace_after_bad(session, 4, nf, "004", params) is None
    other = {**bad, "reason": "bad_registration:002"}
    assert retrace_after_bad(session, 4, other, "004", params) is None
    _write_meta(session, 3, None, {"action": "move", "reverse": True, "retrace": True})
    assert retrace_after_bad(session, 4, bad, "004", params) is None


def test_a_retrace_inverts_a_mixed_gear_step(tmp_path):
    from app.snoo_map_server import retrace_after_bad
    from app.snoo_mapper import MapParams

    session = tmp_path / "s"
    segs = [{"steer_deg": 20.0, "distance_cm": 10.0, "reverse": True},
            {"steer_deg": 0.0, "distance_cm": 40.0, "reverse": False}]  # fmt: skip
    prev = {"steer_deg": 0.0, "speed": 12.0, "reverse": False, "est_distance_cm": 30.0,
            "segments": segs, "stop_reason": "clear"}  # fmt: skip
    _write_meta(session, 1, prev)
    bad = {"action": "done", "reason": "bad_registration:001"}
    gear = MapParams(segments=True, reverse=True, gear_segments=True)
    out = retrace_after_bad(session, 1, bad, "001", gear)
    assert np.allclose([o[:2] for o in out], [(0.0, 0.2), (20.0, 0.1)])
    assert [o[2] for o in out] == [True, False]  # 止まった所まで、後ろから、向きを逆に
    assert retrace_after_bad(session, 1, bad, "001", MapParams(segments=True, reverse=True)) is None


def test_busy_reports_the_last_reply_for_live_deploys(tmp_path):
    c = _client(tmp_path)
    b = c.get("/v1/snoo/map/busy").json()
    assert b == {"in_flight": 0, "since_reply_s": None, "last": None}
    _post(c, _meta(0))
    b = c.get("/v1/snoo/map/busy").json()
    assert b["in_flight"] == 0 and b["last"] == "move" and 0 <= b["since_reply_s"] < 5


def test_an_agent_step_reserved_for_the_next_station_is_used_once(tmp_path):
    from app.snoo_map_server import override_path

    c = _client(tmp_path)
    assert _post(c, _meta(0)).json()["action"] == "move"
    path = override_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"steer_deg": 0.0, "distance_cm": 10.0, "t": time.time(),
                                "note": "左のソファの下を見たい"}))  # fmt: skip
    a = _post(c, _meta(1)).json()
    assert a["agent"] is True and a["distance_cm"] == 10.0 and not path.exists()
    log = [json.loads(x) for x in (path.parent / "override_result.jsonl").read_text().splitlines()]
    assert log[-1]["used"] is True


def test_a_bad_agent_step_is_not_used(tmp_path):
    from app.snoo_map_server import override_path

    c = _client(tmp_path)
    _post(c, _meta(0))
    path = override_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"steer_deg": 0.0, "distance_cm": 80.0, "t": time.time()}))
    a = _post(c, _meta(1)).json()
    assert not a.get("agent")
    log = [json.loads(x) for x in (path.parent / "override_result.jsonl").read_text().splitlines()]
    assert log[-1]["used"] is False and log[-1]["why"]


# --- 家の地図（SNOO_HOME_MAP） ---

STEP = {"steer_deg": 0, "speed": 12, "seconds": 1.2}


def test_home_map_off_changes_nothing(tmp_path, monkeypatch):
    """SNOO_HOME_MAP が無ければ、家の地図を読まず・書かず、返事も今までと同じ。"""
    import app.snoo_map_server as _server

    monkeypatch.setattr(_server, "HOME_MAP", False)
    monkeypatch.setattr(_server, "HOME_DIR", str(tmp_path / "home"))
    c = _client(tmp_path / "sessions")
    replies = [_post(c, _meta(0)).json(), _post(c, _meta(1, prev_step=STEP)).json()]
    session = tmp_path / "sessions" / "explore_20261029_201500"
    assert not (session / "home.json").exists() and not (tmp_path / "home").exists()
    assert c.get("/v1/snoo/map/home.json").status_code == 404
    assert c.get("/v1/snoo/map/home.png").status_code == 404
    plan = json.loads((session / "plan.json").read_text())
    assert "home" not in plan and "home" not in plan["timing_s"]
    # 家の地図を使っても、家が空なら 1 つ目のセッションが家になり、返事は同じ
    monkeypatch.setattr(_server, "HOME_MAP", True)
    c2 = _client(tmp_path / "sessions2")
    again = [_post(c2, _meta(0)).json(), _post(c2, _meta(1, prev_step=STEP)).json()]
    assert again == replies
    assert (tmp_path / "home" / "home.json").is_file()


def test_home_map_on_registers_sessions_and_writes_home_json(tmp_path, monkeypatch):
    import app.snoo_map_server as _server

    monkeypatch.setattr(_server, "HOME_MAP", True)
    monkeypatch.setattr(_server, "HOME_DIR", str(tmp_path / "home"))
    c = _client(tmp_path / "sessions")
    for sid in ("explore_20261029_201500", "explore_20261029_203000"):
        assert _post(c, _meta(0, session=sid)).status_code == 200
        r = _post(c, _meta(1, session=sid, prev_step=STEP))
        assert r.status_code == 200, r.text
    root = tmp_path / "sessions"
    first = json.loads((root / "explore_20261029_201500" / "home.json").read_text())
    assert first["registered"] and first["reason"] == "anchor"
    # 2 つ目は同じ所から走った（作り物の部屋は同じ）ので、家の座標でもほぼ (0, 0, 0)
    second = json.loads((root / "explore_20261029_203000" / "home.json").read_text())
    assert second["registered"], second
    assert abs(second["pose"]["x"]) < 0.05 and abs(second["pose"]["y"]) < 0.05
    assert abs(second["pose"]["yaw_deg"]) < 3
    home = c.get("/v1/snoo/map/home.json")
    assert home.status_code == 200
    assert [s["id"] for s in home.json()["sessions"]] == [
        "explore_20261029_201500", "explore_20261029_203000"]  # fmt: skip
    png = c.get("/v1/snoo/map/home.png")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    plan = c.get("/v1/snoo/map/explore_20261029_203000/plan.json").json()
    assert plan["home"]["registered"] and "home" in plan["timing_s"]


def test_blocked_ahead_uses_the_segment_the_old_pi_stopped_in(tmp_path):
    """古い Pi は返した区切りを全部返す。止まったのは、走った長さ（10 cm）までの最後の区切り
    （前への 1 つ目）なので、前の先に足す（後ろへの 2 つ目の先ではない）。"""
    from app.roommap import Grid
    from app.snoo_map_server import mark_steps_driven

    prev = {"steer_deg": 0.0, "speed": 12, "reverse": False, "est_distance_cm": 10.0,
            "stop_reason": "clear",
            "segments": [{"steer_deg": 0.0, "distance_cm": 20.0, "reverse": False},
                         {"steer_deg": 0.0, "distance_cm": 20.0, "reverse": True}]}  # fmt: skip
    session, poses = _line(tmp_path, [(0.0, None), (0.10, prev)])
    g = Grid()
    mark_steps_driven(g, session, 1, poses)
    assert g.logodds[g.cell(0.20, 0.0)] > 0  # 止まった所（カメラ x = 0.10）の前の端の先


def test_blocked_ahead_expires_after_a_few_stations(tmp_path):
    from app.roommap import Grid
    from app.snoo_map_server import BLOCKED_KEEP_STATIONS, mark_steps_driven

    stop = _step(5, False, stop_reason="clear")
    n = BLOCKED_KEEP_STATIONS + 1
    steps = [(0.0, None), (0.05, stop)] + [(0.05, None)] * n
    session, poses = _line(tmp_path, steps)
    g = Grid()
    mark_steps_driven(g, session, 1, poses)
    assert g.logodds[g.cell(0.15, 0.0)] > 0
    g = Grid()
    mark_steps_driven(g, session, 1 + BLOCKED_KEEP_STATIONS, poses)
    assert g.logodds[g.cell(0.15, 0.0)] == 0


def test_a_retrace_stops_where_the_reverse_budget_runs_out(tmp_path):
    """戻る後ろへを 30 cm で切ったら、その先の区切りは足さない（走った道から外れた所から走る）。"""
    from app.snoo_map_server import retrace_after_bad
    from app.snoo_mapper import MapParams

    session = tmp_path / "s"
    segs = [{"steer_deg": 0.0, "distance_cm": 10.0, "reverse": True},
            {"steer_deg": 20.0, "distance_cm": 35.0, "reverse": False}]  # fmt: skip
    prev = {"steer_deg": 0.0, "speed": 12.0, "reverse": True, "est_distance_cm": 45.0,
            "segments": segs, "stop_reason": None}  # fmt: skip
    _write_meta(session, 1, prev)
    bad = {"action": "done", "reason": "bad_registration:001"}
    gear = MapParams(segments=True, reverse=True, gear_segments=True)
    out = retrace_after_bad(session, 1, bad, "001", gear)
    assert np.allclose([o[:2] for o in out], [(20.0, 0.30)]) and out[0][2] is True
