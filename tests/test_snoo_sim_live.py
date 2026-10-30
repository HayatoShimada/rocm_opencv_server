"""手で動かす猫（app/snoo_sim.ControlledSnoo）・作った家の部屋（room_from_home）と、実機の速さで回す
シミュレーターのサーバー（scripts/snoo_sim_live.py）を試す。GPU の計算は使わない（絵は
MUJOCO_GL=egl）。
サーバーは光線を少なく・速く回して、数秒で終える。
"""

import math
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.roommap import Pose
from app.snoo import parse_allow
from app.snoo_sim import (
    CAT_MAX_SPEED_M_S,
    ControlledSnoo,
    RandomWalkSnoo,
    Room,
    SnooPose,
    _box,
    room_from_home,
    synthetic_room,
)


def open_room() -> Room:
    """4 x 4 m の空の部屋（壁だけ）に、x = 1.0〜1.2 の箱を 1 つ。"""
    wall = (0.8, 0.8, 0.8, 1.0)
    blocks = [
        _box(-2.05, -2.05, 2.05, -2.0, 0, 0.5, wall),
        _box(-2.05, 2.0, 2.05, 2.05, 0, 0.5, wall),
        _box(-2.05, -2.0, -2.0, 2.0, 0, 0.5, wall),
        _box(2.0, -2.0, 2.05, 2.0, 0, 0.5, wall),
        _box(1.0, -0.5, 1.2, 0.5, 0, 0.5, wall, "box"),
    ]
    return Room(blocks, (-2.0, -2.0, 2.0, 2.0), start=Pose(-1.5, 1.5, 0.0))


def run(cat, seconds, robot=None, dt=0.1):
    for _ in range(int(round(seconds / dt))):
        cat.step(dt, robot)
    return cat.pose


def test_controlled_snoo_walks_and_turns():
    cat = ControlledSnoo(open_room(), SnooPose(-1.0, -1.0, 0.0))
    cat.command(1.0, 0.0)
    p = run(cat, 1.0)
    assert p.x == pytest.approx(-1.0 + CAT_MAX_SPEED_M_S, abs=0.02) and p.y == pytest.approx(-1.0)
    assert p.moving and p.posture == "walk" and cat.controlled
    cat.command(0.0, 1.0)  # 左回り
    p = run(cat, 0.5)
    assert p.yaw > 1.0 and p.x == pytest.approx(-1.0 + CAT_MAX_SPEED_M_S, abs=0.02)
    cat.command(-1.0, 0.0)  # 後ずさりは遅い
    before = (p.x, p.y)
    p = run(cat, 1.0)
    assert 0.1 < math.dist(before, (p.x, p.y)) < 0.3


def test_controlled_snoo_does_not_walk_through_furniture():
    room = open_room()
    cat = ControlledSnoo(room, SnooPose(0.0, 0.0, 0.0))
    cat.command(1.0, 0.0)
    p = run(cat, 3.0)  # まっすぐ x = 1.0 の箱へ
    assert p.x < 1.0 - 0.1 and cat.blocked
    probe = RandomWalkSnoo(room, p)
    assert probe.free_at(p.x, p.y)
    # 斜めに当たれば、箱の面に沿って滑り、端を回って先へ行く（箱の中には一度も入らない）
    cat2 = ControlledSnoo(room, SnooPose(0.0, 0.2, 0.3))
    cat2.command(1.0, 0.0)
    for _ in range(30):
        q = cat2.step(0.1, None)
        assert probe.free_at(q.x, q.y)
    assert q.x > 1.2 and q.y > 0.6


def test_controlled_snoo_keeps_away_from_the_car_but_can_leave():
    cat = ControlledSnoo(open_room(), SnooPose(-1.0, 0.0, 0.0))
    robot = Pose(-0.5, 0.0, math.pi)  # カメラ。車の真ん中は 8 cm 後ろ（x = -0.42）
    cat.command(1.0, 0.0)
    p = run(cat, 2.0, robot)
    assert math.dist((p.x, p.y), (-0.42, 0.0)) >= 0.3 - 1e-6
    cat.command(-1.0, 0.0)  # 離れる向きには動ける
    q = run(cat, 1.0, robot)
    assert q.x < p.x - 0.1


def test_controlled_snoo_postures_and_fallback_after_timeout():
    room = open_room()
    fb = RandomWalkSnoo(room, SnooPose(-1.0, -1.0, 0.0), np.random.default_rng(3))
    cat = ControlledSnoo(room, SnooPose(-1.0, -1.0, 0.0), fb, timeout_s=5.0)
    cat.command(0.0, 0.0, "sit")
    p = run(cat, 1.0)
    assert p.posture == "sit" and not p.moving and cat.controlled
    with pytest.raises(ValueError):
        cat.command(0.0, 0.0, "dance")
    cat.command(0.5, 0.0)  # 動かせば立って歩く
    assert run(cat, 0.2).posture == "walk"
    # 5 秒指令が来なければ、自分で動く猫に任せる（任せた猫は今の所から動き出す）
    here = cat.pose
    run(cat, 5.0)
    assert not cat.controlled
    assert fb.pose is not None
    run(cat, 20.0)
    assert cat.pose is fb.pose
    moved = math.dist((here.x, here.y), (cat.pose.x, cat.pose.y))
    assert moved > 0.05
    # また指令が来れば、任せた猫の今の所から手で動かす
    at = cat.pose
    cat.command(0.0, 0.0)
    assert cat.controlled and run(cat, 0.1).x == pytest.approx(at.x)


def test_room_from_home_centres_the_house_on_the_map():
    from app.snoo_homes import make_home

    room = room_from_home(make_home(3))
    x0, y0, x1, y1 = room.bounds
    assert -6.0 <= x0 < x1 <= 6.0 and -6.0 <= y0 < y1 <= 6.0
    assert x0 <= room.start.x <= x1 and y0 <= room.start.y <= y1
    assert len(room.blocks) > 10
    probe = RandomWalkSnoo(room, SnooPose(room.start.x, room.start.y, 0.0))
    assert probe.free_at(room.start.x, room.start.y)


# --- サーバー ---


@pytest.fixture
def live():
    from scripts.snoo_sim_live import Live

    # 猫を見守る（near_cm）と猫しだいで探索が進まないので、ここでは切る（見守るのは下で試す）
    lv = Live("synthetic", seed=1, rays=(16, 12), speed=50.0, server_s=1.0, near_cm=0)
    yield lv
    lv.close()


def _client(live, host="127.0.0.1"):
    from scripts.snoo_sim_live import create_app

    return TestClient(create_app(live, parse_allow("127.0.0.1")), client=(host, 1))


def _wait(c, cond, timeout=20.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        st = c.get("/sim/state").json()
        if cond(st):
            return st
        time.sleep(0.05)
    raise AssertionError(f"待っても来なかった: {st}")


def test_sim_live_runs_and_takes_cat_commands(live):
    c = _client(live)
    st = _wait(c, lambda s: s["t"] > 1.0)
    assert st["running"] and st["error"] is None and st["room"] == "synthetic"
    assert set(st["cat"]) >= {"x", "y", "posture", "controlled"}
    r = c.post("/sim/cat", json={"forward": 1.0, "turn": 0.0, "posture": "walk"})
    assert r.status_code == 200 and r.json()["posture"] == "walk"
    st = _wait(c, lambda s: s["cat"]["controlled"])
    assert st["cat"]["since_command_s"] is not None
    assert c.post("/sim/cat", json={"forward": 2.0, "turn": 0}).status_code == 422
    assert c.post("/sim/cat", json={"posture": "dance"}).status_code == 400
    c.post("/sim/cat", json={"forward": 0, "turn": 0, "posture": "sleep"})
    _wait(c, lambda s: s["cat"]["posture"] == "sleep")
    # 車は撮って次の 1 歩を決める（地図のサーバーと同じ決め方）
    # 車は A（POST /sim/robot）でマッピングを始めるまで、撮ったところで待つ
    assert c.post("/sim/robot", json={"mapping": True}).json() == {"mapping": True}
    st = _wait(c, lambda s: s["stations"] >= 1 and s["action"], timeout=40.0)
    assert st["action_text"] and st["phase_text"]
    assert "reaction" in st and st["coverage"] >= 0


def test_sim_live_frames_and_page(live):
    c = _client(live)
    for view in ("fpv", "top", "3d"):
        r = c.get("/sim/frame.jpg", params={"view": view})
        assert r.status_code == 200 and r.content[:2] == b"\xff\xd8", view
    assert c.get("/sim/frame.jpg", params={"view": "side"}).status_code == 400
    page = c.get("/sim")
    assert page.status_code == 200 and "/controller/controller.js" in page.text
    assert c.get("/sim/static/sim.js").status_code == 200
    assert c.get("/controller/controller.css").status_code == 200
    rooms = c.get("/sim/rooms").json()
    assert rooms["now"] == "synthetic" and any(
        x["room"].startswith("home:") for x in rooms["rooms"]
    )
    blocked = _client(live, "10.9.9.9").get("/sim", headers={"accept": "text/html"})
    assert blocked.status_code == 403 and "SNOO_SIM_ALLOW" in blocked.text


def test_sim_live_reset_changes_the_room(live):
    c = _client(live)
    from scripts.snoo_sim_live import home_seeds

    h = home_seeds()[0]
    r = c.post("/sim/reset", json={"room": f"home:{h}", "seed": 4})
    assert r.status_code == 200
    st = r.json()
    assert st["room"] == f"home:{h}" and st["seed"] == 4 and st["room_name"] == f"home{h}"
    _wait(c, lambda s: s["running"] and s["t"] > 0.5)
    assert c.get("/sim/frame.jpg", params={"view": "top"}).status_code == 200
    bad = c.post("/sim/reset", json={"room": "session:no_such_session"})
    assert bad.status_code == 400
    assert c.get("/sim/state").json()["room"] == f"home:{h}"  # 作れなければ今のまま回す
    r = c.post("/sim/reset", json={"room": "synthetic"})
    assert r.json()["room_name"] == synthetic_room().name


def test_sim_live_robot_stops_and_watches_a_near_cat():
    """Pi と同じく、snoo が 150 cm より近くに見えたら止まってカメラで追う。"""
    from scripts.snoo_sim_live import Live

    lv = Live("synthetic", seed=1, rays=(16, 12), speed=50.0, server_s=1.0, fallback="none")
    try:
        c = _client(lv)
        _wait(c, lambda s: s["t"] > 0.2)
        # 猫を車の 60 cm 前へ置き直す（車は始めの所で首を振っている）
        with lv.lock:
            cam = lv.sim.car.camera_pose()
            cat = lv.sim.snoo
            cat.pose = SnooPose(cam.x + 0.6, cam.y, math.pi, False, "sit")
            cat.command(0.0, 0.0, "sit")
        st = _wait(c, lambda s: s["reaction"]["watching"])
        assert st["phase"] == "watch" and st["reaction"]["snoo_seen"]
        assert st["reaction"]["snoo_dist_cm"] < 150
        assert "見守" in st["reaction"]["text"]
    finally:
        lv.close()
