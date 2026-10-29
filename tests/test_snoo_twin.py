"""app/snoo_twin.py（/twin の画面と JSON）を、作り物のセッションで試す。GPU は使わない。"""

import json
import math
import os
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.snoo import parse_allow
from app.snoo_map_server import create_app

JPEG = cv2.imencode(".jpg", np.full((48, 64, 3), 128, np.uint8))[1].tobytes()
ME = ("127.0.0.1", 50000)


def no_reconstruct(refs, npz, marks):
    raise AssertionError("VGGT を回してはいけない")


def _station(session, sid, battery=7.9, sonar=50.0, prev=None):
    d = session / f"{sid:03d}"
    d.mkdir(parents=True)
    frames = [
        {"file": f"{sid}0_pan+030_tilt-15.jpg", "pan_deg": 30.0, "tilt_deg": -15.0},
        {"file": f"{sid}1_pan-030_tilt-00.jpg", "pan_deg": -30.0, "tilt_deg": 0.0},
        {"file": f"d{sid}_pan+000_tilt+00.jpg", "pan_deg": 0.0, "tilt_deg": 0.0, "moving": True},
    ]
    for f in frames:
        (d / f["file"]).write_bytes(JPEG)
    meta = {
        "session_id": session.name,
        "station_id": sid,
        "battery_v": battery,
        "ultrasonic_cm": sonar,
        "prev_step": prev,
        "supports": ["reverse", "segments"],
        "frames": frames,
    }
    (d / "meta.json").write_text(json.dumps(meta))


def _session(root, name="explore_a", reply=True):
    """2 か所のステーションと、窓の原点を (1, 0, 90°) にした間取り・返事。"""
    s = root / name
    _station(s, 0)
    _station(s, 1, prev={"steer_deg": 0.0, "speed": 12, "seconds": 1.0})
    poses = {"0": {"x": 1.0, "y": 0.0, "yaw_deg": 90.0}, "1": {"x": 1.0, "y": 0.3, "yaw_deg": 90.0}}
    (s / "poses.json").write_text(json.dumps(poses))
    window = {"stations": [0, 1], "origin_pose": {"x": 1.0, "y": 0.0, "yaw_deg": 90.0}}
    grid = np.full((240, 240), 128, np.uint8)
    grid[100:110, 120:130] = 255
    grid[100, 120] = 0
    cv2.imwrite(str(s / "grid.png"), grid)
    cv2.imwrite(str(s / "high.png"), np.zeros((240, 240), np.uint8))
    cv2.imwrite(str(s / "topdown.png"), np.zeros((8, 8, 3), np.uint8))
    room = {
        "resolution_cm": 5.0,
        "origin": [-6.0, -6.0],
        "size": [240, 240],
        "layers": {"high": {"png": "high.png", "legend": {"min_height_m": 0.15}}},
        "window": window,
    }
    (s / "room.json").write_text(json.dumps(room))
    move = {
        "action": "move",
        "steer_deg": -28.3,
        "speed": 12,
        "distance_cm": 4.0,
        "pose": {"x": 1.0, "y": 0.3, "yaw_deg": 90.0},
        "target": {"x": 0.5, "y": 0.8, "yaw_deg": 180.0},
        "segments": [
            {"steer_deg": -28.3, "distance_cm": 4.0},
            {"steer_deg": 0.0, "distance_cm": 20.0},
        ],
    }
    plan = {
        **move,
        # 窓の座標（原点は (1, 0)、向きは 90°）。セッションの座標では x と y が入れ替わる
        "plan": {
            "target": {"x": 0.8, "y": 0.5, "yaw_deg": 90.0},
            "gain_cells": 10,
            "length_m": 1.0,
            "path": [[0.3, 0.0], [0.8, 0.5]],
        },
        "next": move,
        "window": window,
        "timing_s": {"vggt": 6.0, "map": 2.0},
    }
    if reply:
        plan["reply"] = move
    (s / "plan.json").write_text(json.dumps(plan))
    return s


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "sessions"
    r.mkdir()
    return r


def _client(root, live=None, client=ME):
    app = create_app(root, parse_allow("192.168.0.188,127.0.0.1"), no_reconstruct, live)
    return TestClient(app, client=client)


def test_page_and_static_files_are_served(root):
    c = _client(root)
    r = c.get("/twin")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "/twin/static/twin.js" in r.text
    for name, kind in (("twin.js", "javascript"), ("view3d.js", "javascript"), ("twin.css", "css")):
        r = c.get(f"/twin/static/{name}")
        assert r.status_code == 200 and kind in r.headers["content-type"]
    assert c.get("/twin/static/nothing.js").status_code == 404
    assert c.get("/twin/static/..%2Fsnoo_twin.py").status_code == 404


def test_page_refuses_senders_not_in_the_allow_list(root):
    c = _client(root, client=("10.1.2.3", 1))
    assert c.get("/twin").status_code == 403
    assert c.get("/twin/api/sessions").status_code == 403


def test_sessions_are_listed_newest_first(root):
    old = _session(root, "explore_old")
    new = _session(root, "explore_new")
    os.utime(old / "plan.json", (time.time() - 100, time.time() - 100))
    for p in (old, old / "001"):
        os.utime(p, (time.time() - 100, time.time() - 100))
    (root / "not a session").mkdir()  # 名前が合わないものは出さない
    r = _client(root).get("/twin/api/sessions")
    assert r.headers["cache-control"] == "no-store"
    got = r.json()["sessions"]
    assert [s["id"] for s in got] == ["explore_new", "explore_old"]
    assert got[0]["stations"] == 2 and got[0]["latest"] == 1
    assert new.exists()


def test_sessions_are_empty_without_data(tmp_path):
    assert _client(tmp_path / "none").get("/twin/api/sessions").json() == {"sessions": []}


def test_state_puts_everything_in_session_coordinates(root):
    _session(root)
    st = _client(root).get("/twin/api/explore_a/state").json()
    assert st["session"] == "explore_a" and st["reply_station"] == 1
    assert [s["id"] for s in st["stations"]] == [0, 1]
    s1 = st["stations"][1]
    assert s1["pose"] == {"x": 1.0, "y": 0.3, "yaw_deg": 90.0}
    assert s1["battery_v"] == 7.9 and s1["ultrasonic_cm"] == 50.0
    assert s1["prev_step"]["seconds"] == 1.0
    # 走りながら撮ったコマは入れない
    assert [f["file"] for f in s1["frames"]] == ["10_pan+030_tilt-15.jpg", "11_pan-030_tilt-00.jpg"]
    assert st["grid"]["origin_pose"] == {"x": 1.0, "y": 0.0, "yaw_deg": 90.0}
    assert st["grid"]["high"] and st["grid"]["high_min_m"] == 0.15
    # 窓の (0.3, 0) は、原点 (1, 0)・向き 90° のセッションの座標で (1, 0.3)
    assert st["plan"]["path"][0] == [1.0, 0.3]
    assert st["plan"]["path"][1] == [0.5, 0.8]
    assert st["plan"]["target"] == {"x": 0.5, "y": 0.8, "yaw_deg": 180.0}
    assert st["reply"]["action"] == "move" and st["timing_s"]["vggt"] == 6.0
    assert st["car"]["wheelbase_m"] == 0.14 and st["car"]["hfov_deg"] == 54.0


def test_state_drives_the_segments_with_the_bicycle_model(root):
    _session(root)
    step = _client(root).get("/twin/api/explore_a/state").json()["step"]
    assert not step["reverse"]
    a, b = step["segments"]
    assert a["path"][0] == [1.0, 0.3]  # Pi に返した位置から走り出す
    assert a["path"][-1] == b["path"][0]
    assert (a["steer_deg"], b["distance_cm"]) == (-28.3, 20.0)

    # 区間の長さは後ろの車軸が走る長さ。描くのはカメラの真下（16 cm 前）なので、曲がる区間は
    # 回る半径が 0.26 → 0.31 m になる分（1.17 倍）長い。まっすぐなら同じ
    def length(seg):
        return np.linalg.norm(np.diff(np.array(seg["path"]), axis=0), axis=1).sum()

    assert length(a) == pytest.approx(0.04 * math.hypot(0.26, 0.16) / 0.26, rel=0.03)
    assert length(b) == pytest.approx(0.20, abs=0.002)
    # 左へ曲がる（steer は右が +）ので、+y を向いた車は -x へ寄る
    assert b["path"][-1][0] < 1.0
    end = step["end"]
    assert math.hypot(end["x"] - b["path"][-1][0], end["y"] - b["path"][-1][1]) < 1e-3


def test_old_plan_without_reply_uses_next(root):
    _session(root, reply=False)
    st = _client(root).get("/twin/api/explore_a/state").json()
    assert st["reply"]["action"] == "move" and st["step"] is not None


def test_state_of_a_session_without_a_map_yet(root):
    _station(root / "explore_b", 0)
    st = _client(root).get("/twin/api/explore_b/state").json()
    assert st["grid"] is None and st["reply"] is None and st["step"] is None
    assert st["stations"][0]["pose"] is None


def test_state_errors(root):
    c = _client(root)
    assert c.get("/twin/api/nothing/state").status_code == 404
    assert c.get("/twin/api/bad.id/state").status_code == 400


def test_map_images(root):
    _session(root)
    c = _client(root)
    r = c.get("/twin/api/explore_a/map/grid.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (240, 240) and img[100, 120] == 0
    assert c.get("/twin/api/explore_a/map/high.png").status_code == 200
    assert c.get("/twin/api/explore_a/map/room.json").status_code == 404
    (root / "explore_a" / "high.png").unlink()
    assert c.get("/twin/api/explore_a/map/high.png").status_code == 404


def test_station_images(root):
    _session(root)
    c = _client(root)
    r = c.get("/twin/api/explore_a/station/1/10_pan+030_tilt-15.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content == JPEG
    assert c.get("/twin/api/explore_a/station/1/nothing.jpg").status_code == 404
    assert c.get("/twin/api/explore_a/station/7/10_pan+030_tilt-15.jpg").status_code == 404
    assert c.get("/twin/api/explore_a/station/1/meta.json").status_code == 404
    assert c.get("/twin/api/explore_a/station/-1/x.jpg").status_code == 404
    assert c.get("/twin/api/explore_a/station/abc/x.jpg").status_code == 422


@pytest.mark.parametrize(
    "path",
    [
        "/twin/api/explore_a/station/1/..%2F..%2Fsecret.jpg",
        "/twin/api/explore_a/station/1/%2E%2E%2Fsecret.jpg",
        "/twin/api/explore_a/station/1/../../secret.jpg",
        "/twin/api/..%2Fexplore_a/station/1/10_pan+030_tilt-15.jpg",
        "/twin/api/explore_a/map/..%2F..%2Fsecret.jpg",
        "/twin/api/live/..%2Fsecret.jpg",
        "/twin/api/live/.secret.jpg",
    ],
)
def test_files_outside_the_session_cannot_be_read(root, tmp_path, path):
    _session(root)
    (tmp_path / "secret.jpg").write_bytes(b"secret")
    (root / "secret.jpg").write_bytes(b"secret")
    live = tmp_path / "live"
    live.mkdir()
    r = _client(root, live).get(path)
    assert r.status_code in (400, 404, 422)
    assert b"secret" not in r.content


def test_links_pointing_outside_are_not_followed(root, tmp_path):
    s = _session(root)
    (tmp_path / "secret.jpg").write_bytes(b"secret")
    (s / "001" / "link.jpg").symlink_to(tmp_path / "secret.jpg")
    live = tmp_path / "live"
    live.mkdir()
    (live / "evil.jpg").symlink_to(tmp_path / "secret.jpg")
    c = _client(root, live)
    assert c.get("/twin/api/explore_a/station/1/link.jpg").status_code == 404
    assert c.get("/twin/api/live/evil.jpg").status_code == 404


def test_live_frames(root, tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    now = time.time()
    for name, t in (("192.168.0.188", now - 1), ("100.73.14.8", now - 30)):
        (live / f"{name}.jpg").write_bytes(JPEG)
        info = {"seq": 3, "t_recv": t, "dets": [], "cmd": {"pan_deg": 10.0}}
        (live / f"{name}.json").write_text(json.dumps(info))
    (live / ".192.168.0.188.json.tmp").write_text("{")  # 書きかけは読まない
    (live / "broken.json").write_text("{")
    c = _client(root, live)
    got = c.get("/twin/api/live").json()
    assert got["enabled"]
    senders = got["senders"]
    assert [s["sender"] for s in senders] == ["192.168.0.188", "100.73.14.8"]  # 新しい順
    assert 0.5 < senders[0]["age_s"] < 10 and senders[0]["has_image"]
    r = c.get("/twin/api/live/192.168.0.188.jpg")
    assert r.status_code == 200 and r.content == JPEG
    assert c.get("/twin/api/live/10.0.0.1.jpg").status_code == 404


def test_live_frames_when_turned_off(root):
    c = _client(root, None)
    assert c.get("/twin/api/live").json() == {"enabled": False, "senders": []}
    assert c.get("/twin/api/live/192.168.0.188.jpg").status_code == 404
