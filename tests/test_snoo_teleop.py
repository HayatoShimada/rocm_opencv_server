"""app/snoo_teleop.py（手で動かす API と /drive）を、時計を差し替えて試す。GPU も Pi も使わない。"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.snoo import parse_allow
from app.snoo_map_server import create_app
from app.snoo_teleop import (
    DEADMAN_S,
    PREFIX,
    TELEOP_MAX_REVERSE,
    TELEOP_MAX_SPEED,
    TELEOP_MAX_STEER_DEG,
    TELEOP_PAN_DEG,
    TELEOP_TILT_MAX_DEG,
    TELEOP_TILT_MIN_DEG,
    Teleop,
    add_teleop,
    to_pi,
)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def env():
    clock = Clock()
    app = FastAPI()
    add_teleop(app, Teleop(clock))
    return TestClient(app), clock


def send(c, seq, **kw):
    return c.post(PREFIX, json={"seq": seq, **kw}).json()


def test_mapping_and_caps():
    assert to_pi(1, 1, 1, 1) == {
        "steer_deg": TELEOP_MAX_STEER_DEG,
        "speed": TELEOP_MAX_SPEED,
        "pan_deg": TELEOP_PAN_DEG,
        "tilt_deg": TELEOP_TILT_MAX_DEG,
    }
    back = to_pi(-1, -1, -1, -1)
    assert back["speed"] == -TELEOP_MAX_REVERSE  # 後ろへは遅く
    assert back["steer_deg"] == -TELEOP_MAX_STEER_DEG
    assert back["pan_deg"] == -TELEOP_PAN_DEG
    assert back["tilt_deg"] == TELEOP_TILT_MIN_DEG  # 下は -20 まで
    half = to_pi(0.5, 0.5, 0, -0.5)
    assert half["speed"] == 7.5 and half["steer_deg"] == 14.0 and half["tilt_deg"] == -10.0
    # 外れた値は端に寄せる
    assert to_pi(3, 9, -4, 2)["speed"] == TELEOP_MAX_SPEED
    assert (TELEOP_MAX_SPEED, TELEOP_MAX_REVERSE, TELEOP_MAX_STEER_DEG) == (15, 10, 28)
    assert (TELEOP_PAN_DEG, TELEOP_TILT_MIN_DEG, TELEOP_TILT_MAX_DEG) == (35, -20, 30)


def test_out_of_range_input_is_refused(env):
    c, _ = env
    assert c.post(PREFIX, json={"seq": 1, "speed": 1.5}).status_code == 422
    assert c.post(PREFIX, json={"seq": 1, "mode": "dance"}).status_code == 400


def test_pi_gets_the_latest_command_with_its_age(env):
    c, clock = env
    r = send(c, 1, speed=1.0, steer=-0.5, pan=0.2, tilt=0.5)
    assert r["accepted"] and r["command"]["speed"] == 15.0
    clock.t += 0.1
    got = c.get(f"{PREFIX}/latest", params={"after": 0}).json()
    assert got["changed"] and not got["stop"] and not got["stale"]
    assert got["command"] == {"steer_deg": -14.0, "speed": 15.0, "pan_deg": 7.0, "tilt_deg": 15.0}
    assert got["age_s"] == pytest.approx(0.1)
    # 同じ seq を渡せば changed は False（同じ指令をまた受けている）
    again = c.get(f"{PREFIX}/latest", params={"after": got["seq"]}).json()
    assert not again["changed"] and again["command"]["speed"] == 15.0


def test_deadman_turns_an_old_command_into_stop_but_keeps_the_head(env):
    c, clock = env
    send(c, 1, speed=1.0, pan=1.0, tilt=-1.0)
    clock.t += DEADMAN_S + 0.01
    got = c.get(f"{PREFIX}/latest").json()
    assert got["stale"] and got["stop"] and got["reason"] == "deadman"
    assert got["command"]["speed"] == 0.0
    assert got["command"]["pan_deg"] == 35.0 and got["command"]["tilt_deg"] == -20.0


def test_nothing_received_yet_is_stop(env):
    c, _ = env
    got = c.get(f"{PREFIX}/latest").json()
    assert got["stop"] and got["command"]["speed"] == 0.0 and got["age_s"] is None


def test_stop_endpoint_stops_until_the_next_command(env):
    c, clock = env
    send(c, 5, speed=0.8)
    r = c.post(f"{PREFIX}/stop", json={"reason": "button"}).json()
    assert r["stopped"] and r["command"]["speed"] == 0.0
    got = c.get(f"{PREFIX}/latest").json()
    assert got["stop"] and not got["stale"] and got["reason"] == "button"
    # 体（JSON）が無くても止まる
    assert c.post(f"{PREFIX}/stop").status_code == 200
    clock.t += 0.05
    send(c, 6, speed=0.4)
    got = c.get(f"{PREFIX}/latest").json()
    assert not got["stop"] and got["command"]["speed"] == 6.0


def test_older_seq_does_not_overwrite_a_newer_command(env):
    c, _ = env
    send(c, 10, speed=1.0)
    r = send(c, 9, speed=-1.0)  # 遅れて着いた古い指令
    assert not r["accepted"] and r["reason"] == "old_seq"
    assert send(c, 10, speed=0.0)["accepted"] is False
    got = c.get(f"{PREFIX}/latest").json()
    assert got["command"]["speed"] == 15.0
    # 受けた順の通し番号は、stop でも増える
    s0 = got["seq"]
    c.post(f"{PREFIX}/stop")
    assert c.get(f"{PREFIX}/latest").json()["seq"] == s0 + 1


def test_status_shows_operator_and_whether_the_pi_accepts_teleop(env):
    c, clock = env
    st = c.get(f"{PREFIX}/status").json()
    assert st["pi"] is None and not st["operator"]["present"]
    send(c, 1, speed=0.5, mode="teleop")
    q = {"mode": "explore", "battery_v": 7.6, "clear_cm": 42.0, "speed": 0.0, "limit": "mode"}
    got = c.get(f"{PREFIX}/latest", params=q).json()
    assert got["request_mode"] == "teleop"  # Pi が ui_mode を受けるなら、これで teleop に入る
    assert got["request_id"] == 1
    send(c, 2, mode="teleop")
    assert c.get(f"{PREFIX}/latest", params=q).json()["request_id"] == 2
    st = c.get(f"{PREFIX}/status").json()
    assert st["operator"]["present"]
    assert st["pi"]["connected"] and not st["pi"]["accepted"]
    assert st["pi"]["battery_v"] == 7.6 and st["pi"]["clear_cm"] == 42.0
    assert st["pi"]["limit"] == ["mode"]
    c.get(f"{PREFIX}/latest", params={"mode": "teleop", "speed": 7.5, "steer_deg": 0})
    st = c.get(f"{PREFIX}/status").json()
    assert st["pi"]["accepted"] and st["pi"]["applied"]["speed"] == 7.5
    # 時間が経てば、人もいない・Pi も見えない・頼んだモードも渡さない
    clock.t += 10
    st = c.get(f"{PREFIX}/status").json()
    assert not st["operator"]["present"] and not st["pi"]["connected"]
    assert st["request_mode"] is None


def test_drive_page_and_controller_js_are_served(env):
    c, _ = env
    page = c.get("/drive")
    assert page.status_code == 200 and "/controller/controller.js" in page.text
    assert c.get("/controller/controller.js").status_code == 200
    assert c.get("/drive/static/drive.js").status_code == 200
    assert c.get("/controller/..%2Fsnoo_teleop.py").status_code == 404
    assert c.get("/controller/nope.js").status_code == 404


def test_map_server_has_teleop_and_explains_the_allow_list(tmp_path):
    app = create_app(tmp_path, parse_allow("127.0.0.1"), lambda *a: None, None)
    ok = TestClient(app, client=("127.0.0.1", 5))
    assert ok.get("/drive").status_code == 200
    assert ok.post(PREFIX, json={"seq": 1, "speed": 0.2}).json()["accepted"]
    phone = TestClient(app, client=("192.168.0.77", 5))
    r = phone.get("/drive", headers={"accept": "text/html"})
    assert r.status_code == 403 and "SNOO_MAP_ALLOW" in r.text and "192.168.0.77" in r.text
    # API には今まで通り JSON で断る
    r = phone.post(PREFIX, json={"seq": 2, "speed": 1.0})
    assert r.status_code == 403 and r.json()["detail"]
