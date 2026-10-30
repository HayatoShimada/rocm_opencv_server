"""app/snoo_agent_api.py（エージェントの API）を、作り物のセッションと今のフレームで試す。
GPU も動いているサービスも使わない。
"""

import io
import json
import sys
import time
import wave

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.roommap import FREE, OCCUPIED, Grid
from app.snoo import parse_allow
from app.snoo_agent_api import PREFIX, cat_place, create_router, resample_to, wav_bytes
from app.snoo_mapper import grid_png

SESSION = "explore_20261030_000000"
SENDER = "192.168.0.188"
JPEG = cv2.imencode(".jpg", np.full((48, 64, 3), 128, np.uint8))[1].tobytes()
# 箱の下の端が 0.9（足もとが近い。約 26 cm）と 0.52（遠い。約 5 m）の猫
NEAR_CAT = {"box": [0.4, 0.5, 0.6, 0.9], "cat": 0.9, "snoo": 0.8}
FAR_CAT = {"box": [0.45, 0.45, 0.55, 0.52], "cat": 0.9, "snoo": 0.8}


def write_session(root, wall_x=0.4, grid=True):
    """車は (0, 0) で +x を向いている。床は x -1〜1.5・y -1〜1、x = wall_x に幅 60 cm の壁。"""
    d = root / SESSION
    st = d / "000"
    st.mkdir(parents=True)
    meta = {
        "session_id": SESSION,
        "station_id": 0,
        "mode": "explore",
        "battery_v": 7.9,
        "ultrasonic_cm": 55.0,
        "frames": [{"file": "a.jpg", "pan_deg": 0, "tilt_deg": 0}],
    }
    (st / "meta.json").write_text(json.dumps(meta))
    (st / "a.jpg").write_bytes(JPEG)
    (st / "reply.json").write_text(json.dumps({"action": "done", "reason": "explored"}))
    (d / "poses.json").write_text(json.dumps({"0": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}}))
    if not grid:
        return d
    g = Grid(size_m=4.0)
    xs, ys = g.center(np.arange(g.n)[:, None], np.arange(g.n)[None, :])
    state = np.zeros((g.n, g.n), np.uint8)
    state[(xs > -1) & (xs < 1.5) & (ys > -1) & (ys < 1)] = FREE
    state[(np.abs(xs - wall_x) < 0.05) & (np.abs(ys) < 0.3)] = OCCUPIED
    cv2.imwrite(str(d / "grid.png"), grid_png(state))
    cv2.imwrite(str(d / "high.png"), np.zeros((g.n, g.n), np.uint8))
    room = {
        "resolution_cm": 5.0,
        "origin": [g.origin, g.origin],
        "size": [g.n, g.n],
        "window": {"map": "session", "stations": [0]},
    }
    (d / "room.json").write_text(json.dumps(room))
    return d


def write_live(live, dets=(), age_s=0.2, moving=False):
    live.mkdir(exist_ok=True)
    info = {
        "sender": SENDER,
        "seq": 7,
        "t_recv": time.time() - age_s,
        "size": [640, 480],
        "image": f"{SENDER}.jpg",
        "infer_ms": 9.0,
        "dets": list(dets),
        "clear_cm": 80.0,
        "cmd": {
            "steer_deg": 0.0,
            "speed": 0.0,
            "pan_deg": 0.0,
            "tilt_deg": 0.0,
            "moving": moving,
            "flags": int(moving),
        },
    }
    (live / f"{SENDER}.jpg").write_bytes(JPEG)
    (live / f"{SENDER}.json").write_text(json.dumps(info))


class FakeTts:
    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return wav_bytes(np.zeros(1600), 16000)


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "snoo_map" / "sessions"
    root.mkdir(parents=True)
    write_session(root)
    live = tmp_path / "live"
    write_live(live)
    tts = FakeTts()
    app = FastAPI()
    app.include_router(create_router(root, live, tmp_path / "agent", tts=tts))
    client = TestClient(app)
    return {
        "client": client,
        "root": root,
        "live": live,
        "agent": tmp_path / "agent",
        "tmp": tmp_path,
        "tts": tts,
    }


def move(client, **kw):
    body = {"kind": "move", "distance_cm": 15, "speed": 8, "steer_deg": 0, **kw}
    r = client.post(f"{PREFIX}/command", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def failed(reply):
    return {c["name"] for c in reply["checks"] if not c["ok"]}


# --- state・frame・map ---


def test_state_collects_live_station_slam_and_reply(env):
    write_live(env["live"], [FAR_CAT])
    s = env["client"].get(f"{PREFIX}/state").json()
    assert s["session"] == SESSION and s["station"] == 0
    assert s["battery_v"] == 7.9 and s["sonar_cm"] == 55.0 and s["mode"] == "explore"
    assert s["sources"]["battery_v"] == "station_meta"
    assert s["last_reply"] == {"action": "done", "reason": "explored"}
    assert s["sender"] == SENDER and s["frame_age_s"] < 2 and s["clear_cm"] == 80.0
    assert s["pan_deg"] == 0.0 and s["moving"] is False
    assert s["snoo"]["distance_cm"] > 300 and s["pose"]["source"] == "station:0"
    assert s["slam"] is None and s["pi_support"] is False and s["pi_status"] is None


def test_state_prefers_slam_pose_when_fresh(env):
    slam = env["root"] / SESSION / "slam"
    slam.mkdir()
    status = {
        "state": "ok",
        "last_update": time.time(),
        "last_keyframe_age_s": 1.0,
        "alignment": {"rmse_m": 0.01},
        "pose": {"x": 0.2, "y": 0.1, "yaw": 0.5, "t_capture_ns": 1},
    }
    (slam / "status.json").write_text(json.dumps(status))
    s = env["client"].get(f"{PREFIX}/state").json()
    assert s["pose"]["source"] == "slam" and s["pose"]["x"] == 0.2
    assert s["slam"]["yaw_deg"] == pytest.approx(28.65, abs=0.01)


def test_state_without_any_files_is_all_null(tmp_path):
    app = FastAPI()
    app.include_router(create_router(tmp_path / "none", None, tmp_path / "agent"))
    s = TestClient(app).get(f"{PREFIX}/state").json()
    assert s["session"] is None and s["frame"] is None and s["battery_v"] is None


def test_cat_distance_and_bearing_from_the_box_bottom():
    live = {"size": [640, 480], "cmd": {"pan_deg": 0.0, "tilt_deg": 0.0}}
    near = cat_place(NEAR_CAT, live)
    assert 20 < near["distance_cm"] < 32 and abs(near["bearing_deg"]) < 1
    right = cat_place({"box": [0.8, 0.5, 0.9, 0.9], "cat": 0.9}, live)
    assert right["bearing_deg"] > 10  # 右が +
    panned = cat_place(NEAR_CAT, {"size": [640, 480], "cmd": {"pan_deg": -30, "tilt_deg": 0}})
    assert panned["bearing_deg"] == pytest.approx(-30, abs=1)
    sky = cat_place({"box": [0.4, 0.1, 0.6, 0.3], "cat": 0.9}, live)
    assert sky["distance_cm"] is None and sky["bearing_deg"] is not None


def test_frame_jpg_and_json(env):
    write_live(env["live"], [NEAR_CAT])
    r = env["client"].get(f"{PREFIX}/frame.jpg")
    assert r.status_code == 200 and r.content == JPEG and r.headers["x-frame-seq"] == "7"
    j = env["client"].get(f"{PREFIX}/frame.json").json()
    assert j["sender"] == SENDER and j["snoo"]["distance_cm"] < 40
    assert env["client"].get(f"{PREFIX}/frame.jpg", params={"sender": "../x"}).status_code == 400


def test_map_png_and_json(env):
    env["client"].post(f"{PREFIX}/annotate", json={"x": 1.0, "y": 0.5, "label": "sofa"})
    r = env["client"].get(f"{PREFIX}/map.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    j = env["client"].get(f"{PREFIX}/map.json").json()
    assert list(img.shape[1::-1]) == j["size_px"]
    assert j["robot"]["source"] == "station:0" and j["frontier_cells"] > 0
    assert j["annotations"][0]["mark"] == "A1" and j["annotations"][0]["label"] == "sofa"
    # 画素と座標の対応：車（0, 0）の画素は赤
    u = round((0.0 - j["x_at_u0"]) * j["px_per_m"])
    v = round((j["y_at_v0"] - 0.0) * j["px_per_m"])
    b, g, rr = img[v, u].tolist()
    assert rr > 150 and g < 80


def test_map_without_grid_is_404(tmp_path):
    root = tmp_path / "s"
    write_session(root, grid=False)
    app = FastAPI()
    app.include_router(create_router(root, None, tmp_path / "agent"))
    assert TestClient(app).get(f"{PREFIX}/map.png").status_code == 404


# --- 印・困りごと ---


def test_annotate_keeps_session_frame_and_snapshot(env):
    c = env["client"]
    a = c.post(
        f"{PREFIX}/annotate",
        json={
            "x": 0.8,
            "y": -0.2,
            "label": "cable",
            "note": "床に寝たケーブル",
            "image_ref": "live",
        },
    ).json()
    assert a["id"] == 1 and a["session"] == SESSION and a["frame"] == "session"
    assert a["image_ref"] == "snapshot:annotation_1.jpg"
    assert c.get(f"{PREFIX}/snapshots/annotation_1.jpg").content == JPEG
    c.post(f"{PREFIX}/annotate", json={"x": 0, "y": 0, "label": "snoo_spot"})
    lines = (env["root"].parent / "annotations.jsonl").read_text().splitlines()
    assert [json.loads(x)["id"] for x in lines] == [1, 2]
    got = c.get(f"{PREFIX}/annotations", params={"label": "cable"}).json()
    assert [x["note"] for x in got["annotations"]] == ["床に寝たケーブル"]
    assert c.post(f"{PREFIX}/annotate", json={"x": 1e9, "y": 0, "label": "x"}).status_code == 400
    bad = c.post(f"{PREFIX}/annotate", json={"x": 0, "y": 0, "label": "x", "session": "../a"})
    assert bad.status_code == 400


def test_issue_writes_a_draft_with_images(env):
    import base64

    body = {
        "title": "Sofa leg missed",
        "body": "ソファの脚が地図に無い",
        "images": [
            "live",
            "map",
            {"name": "crop.jpg", "data_b64": base64.b64encode(JPEG).decode()},
            f"station:{SESSION}/0/a.jpg",
        ],
    }
    r = env["client"].post(f"{PREFIX}/issue", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["posted"] is False
    d = env["agent"] / "issues"
    (draft,) = d.iterdir()
    assert draft.name.endswith("sofa-leg-missed")
    md = (draft / "issue.md").read_text()
    assert "# Sofa leg missed" in md and "ソファの脚" in md and "![map.png](map.png)" in md
    assert (draft / "map.png").stat().st_size > 100 and (draft / "crop.jpg").read_bytes() == JPEG
    assert (draft / "station_0_a.jpg").is_file()
    bad = env["client"].post(f"{PREFIX}/issue", json={"title": "x", "images": ["../../etc"]})
    assert bad.status_code == 400
    esc = env["client"].post(
        f"{PREFIX}/issue", json={"title": "x", "images": [f"station:{SESSION}/0/../../x"]}
    )
    assert esc.status_code == 400


# --- 声 ---


def test_say_returns_wav_caches_and_queues(env):
    c, tts = env["client"], env["tts"]
    r = c.post(f"{PREFIX}/say", json={"text": "こんにちは"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
    c.post(f"{PREFIX}/say", json={"text": "こんにちは"})
    assert tts.calls == ["こんにちは"]  # 2 回目は作らない
    first = c.get(f"{PREFIX}/say/next").json()["say"]
    assert first["id"] == 1 and first["text"] == "こんにちは"
    assert c.get(f"{PREFIX}/say/next", params={"since": 1}).json()["say"]["id"] == 2
    assert c.get(f"{PREFIX}/say/next", params={"since": 2}).json()["say"] is None
    assert c.get(first["url"]).content == r.content
    c.post(f"{PREFIX}/say", json={"text": "鳴らさない", "queue": False})
    assert c.get(f"{PREFIX}/say/next", params={"since": 2}).json()["say"] is None
    assert c.post(f"{PREFIX}/say", json={"text": "あ" * 201}).status_code == 422


def test_resample_48k_to_16k_keeps_a_low_tone():
    t = np.arange(48000) / 48000
    y = resample_to(np.sin(2 * np.pi * 440 * t) * 10000, 48000)
    assert len(y) == 16000
    spec = np.abs(np.fft.rfft(y))
    assert abs(np.argmax(spec) - 440) <= 1


def test_say_with_pyopenjtalk(tmp_path):
    pytest.importorskip("pyopenjtalk")
    root = tmp_path / "s"
    root.mkdir()
    app = FastAPI()
    app.include_router(create_router(root, None, tmp_path / "agent"))
    r = TestClient(app).post(f"{PREFIX}/say", json={"text": "スヌー、遊ぼう"})
    assert r.status_code == 200
    with wave.open(io.BytesIO(r.content)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert 0.3 < w.getnframes() / 16000 < 5


# --- 指令 ---


def test_short_slow_move_is_queued_without_pi_support(env):
    r = move(env["client"])
    assert r["queued"] and r["id"] == 1 and r["pi_support"] is False
    assert "pi_support: false" in r["note"]
    assert {c["name"] for c in r["checks"]} >= {"clip_pieces", "moves_are_clear", "cat_far"}
    row = json.loads((env["agent"] / "commands.jsonl").read_text())
    assert row["kind"] == "move" and row["distance_cm"] == 15 and row["from_pose"]["x"] == 0


@pytest.mark.parametrize(
    "kw, why",
    [
        ({"distance_cm": 40}, "distance"),
        ({"speed": 20}, "speed"),
        ({"steer_deg": 45}, "steer"),
        ({"distance_cm": 30}, "clip_pieces"),  # 40 cm 先の壁に当たる
    ],
)
def test_move_is_refused_when_too_long_fast_or_blocked(env, kw, why):
    r = move(env["client"], **kw)
    assert not r["queued"] and r["id"] is None and why in failed(r)
    assert not (env["agent"] / "commands.jsonl").exists()


def test_reverse_into_unseen_is_refused(env):
    assert move(env["client"], reverse=True, distance_cm=10)["queued"]
    env["client"].post(f"{PREFIX}/command", json={"kind": "stop"})
    env["client"].post(f"{PREFIX}/command/1/result", json={"status": "cancelled"})
    # 床は x -1 まで。30 cm 下がっても入るが、この後ろの見ていない所へは入らない
    poses_file = env["root"] / SESSION / "poses.json"
    poses_file.write_text(json.dumps({"0": {"x": -0.75, "y": 0.0, "yaw_deg": 0.0}}))
    r = move(env["client"], reverse=True, distance_cm=30)
    assert not r["queued"] and "moves_are_clear" in failed(r)


def test_move_is_refused_with_a_cat_near_or_unknown(env):
    write_live(env["live"], [NEAR_CAT])
    assert "cat_far" in failed(move(env["client"]))
    write_live(env["live"], [{"box": [0.4, 0.1, 0.6, 0.3], "cat": 0.9, "snoo": None}])
    assert "cat_far" in failed(move(env["client"]))  # 距離が分からない猫も
    write_live(env["live"], [{**NEAR_CAT, "cat": 0.2}])  # 猫らしくない箱は気にしない
    assert move(env["client"])["queued"]


def test_move_is_refused_without_fresh_frame_grid_or_while_moving(env, tmp_path):
    write_live(env["live"], age_s=10)
    assert "fresh_frame" in failed(move(env["client"]))
    write_live(env["live"], moving=True)
    assert "robot_still" in failed(move(env["client"]))
    root = tmp_path / "nogrid"
    write_session(root, grid=False)
    app = FastAPI()
    write_live(env["live"])
    app.include_router(create_router(root, env["live"], tmp_path / "agent2"))
    r = TestClient(app).post(
        f"{PREFIX}/command", json={"kind": "move", "distance_cm": 10, "speed": 5}
    )
    assert not r.json()["queued"] and "grid" in failed(r.json())


def test_only_one_move_at_a_time_until_a_result(env):
    c = env["client"]
    assert move(c)["queued"]
    assert "one_move_at_a_time" in failed(move(c))
    c.post(f"{PREFIX}/command/1/result", json={"status": "done", "detail": {"moved_cm": 14}})
    assert move(c)["queued"]


def test_look_scan_stop_and_bad_kind(env):
    c = env["client"]
    assert c.post(f"{PREFIX}/command", json={"kind": "look", "pan_deg": 20}).json()["queued"]
    r = c.post(f"{PREFIX}/command", json={"kind": "look", "pan_deg": 90}).json()
    assert not r["queued"] and "pan_deg" in failed(r)
    assert c.post(f"{PREFIX}/command", json={"kind": "look"}).status_code == 400
    assert c.post(f"{PREFIX}/command", json={"kind": "scan"}).json()["queued"]
    assert c.post(f"{PREFIX}/command", json={"kind": "dance"}).status_code == 400


def test_pi_poll_gets_commands_reports_status_and_stop_goes_first(env):
    c = env["client"]
    move(c)
    c.post(f"{PREFIX}/command", json={"kind": "look", "pan_deg": 10})
    q = {"after": 0, "mode": "agent", "battery_v": 7.6, "sonar_cm": 42}
    nxt = c.get(f"{PREFIX}/commands/next", params=q).json()
    assert nxt["command"]["id"] == 1 and nxt["command"]["kind"] == "move"
    assert "checks" not in nxt["command"] and nxt["heartbeat_s"] == 3.0
    s = c.get(f"{PREFIX}/state").json()
    assert s["pi_support"] and s["mode"] == "agent" and s["battery_v"] == 7.6
    assert s["sources"]["battery_v"] == "pi"
    c.post(f"{PREFIX}/command/1/result", json={"status": "running"})
    c.post(f"{PREFIX}/command", json={"kind": "stop"})
    nxt = c.get(f"{PREFIX}/commands/next", params={"after": 1}).json()
    assert nxt["command"]["kind"] == "stop" and nxt["command"]["id"] == 3
    c.post(f"{PREFIX}/command/1/result", json={"status": "cancelled", "reason": "stop"})
    got = c.get(f"{PREFIX}/command/1").json()
    assert got["status"] == "cancelled" and got["result"]["reason"] == "stop"
    assert c.post(f"{PREFIX}/command/9/result", json={"status": "done"}).status_code == 404
    assert c.post(f"{PREFIX}/command/1/result", json={"status": "yay"}).status_code == 400
    r = move(c)
    assert r["queued"] and r["pi_support"] is True
    # Pi が explore で動いているときは並べない
    c.get(f"{PREFIX}/commands/next", params={"after": 3, "mode": "explore"})
    c.post(f"{PREFIX}/command/{r['id']}/result", json={"status": "done"})
    assert "pi_mode" in failed(move(c))


def test_stop_cancels_commands_the_pi_has_not_taken(env):
    c = env["client"]
    move(c)
    c.post(f"{PREFIX}/command", json={"kind": "stop"})
    got = c.get(f"{PREFIX}/command/1").json()
    assert got["status"] == "cancelled" and got["result"]["reason"] == "stop"
    assert move(c)["queued"]  # 取り消した move は待たない


def test_commands_expire(env):
    now = [1000.0]
    app = FastAPI()
    app.include_router(
        create_router(env["root"], env["live"], env["tmp"] / "a3", clock=lambda: now[0])
    )
    c = TestClient(app)
    c.post(f"{PREFIX}/command", json={"kind": "scan"})
    now[0] += 16
    assert c.get(f"{PREFIX}/commands/next").json()["command"] is None
    assert c.get(f"{PREFIX}/command/1").json()["status"] == "expired"


# --- 地図のサーバーに足す ---


def test_map_server_mounts_the_agent_api(tmp_path):
    from app.snoo_map_server import create_app

    app = create_app(tmp_path, parse_allow("127.0.0.1"), lambda *a: None, None)
    c = TestClient(app, client=("127.0.0.1", 5000))
    assert c.get(f"{PREFIX}/state").status_code == 200
    assert TestClient(app, client=("10.0.0.9", 5000)).get(f"{PREFIX}/state").status_code == 403


def test_map_server_runs_without_the_agent_api(tmp_path, monkeypatch):
    from app.snoo_map_server import create_app

    monkeypatch.setitem(sys.modules, "app.snoo_agent_api", None)  # import が ImportError になる
    app = create_app(tmp_path, parse_allow("127.0.0.1"), lambda *a: None, None)
    c = TestClient(app, client=("127.0.0.1", 5000))
    assert c.get(f"{PREFIX}/state").status_code == 404
    assert c.get("/twin/api/sessions").status_code == 200
