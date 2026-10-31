"""地図のサーバーの位置のグラフ（SNOO_POSE_GRAPH、app/snoo_posegraph.py）を、作り物の部屋で試す。
GPU は使わない。
"""

import json
import logging
import math

import numpy as np
import pytest
from test_snoo_map_server import PREV, _client, _meta, _post, fake_reconstruct

import app.snoo_map_server as server
from app import snoo_posegraph as pg
from app.roommap import Pose
from app.snoo_map_server import read_observations, read_poses
from app.snoo_mapper import Observation

SESSION = "explore_20261029_201500"
CAN = ["reverse", "segments", "long_step"]


@pytest.fixture(autouse=True)
def graph_on(monkeypatch):
    monkeypatch.setattr(server, "POSE_GRAPH", True)
    monkeypatch.setattr(server, "_graph_failed", set())


def _rows(tmp_path):
    p = tmp_path / SESSION / "graph_edges.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()]


def scattered(station: int, cm: float = 40.0):
    """station のコマのカメラの中心を cm ほど散らす（首振りで位置合わせが崩れた VGGT の代わり）。"""

    def reconstruct(refs, npz, marks):
        rec = fake_reconstruct(refs, npz, marks)
        ext = rec.extrinsic.copy()
        rng = np.random.default_rng(3)
        for k, r in enumerate(refs):
            if r.station == station:
                # VGGT の長さ 1 は 2.7 m（tests/test_snoo_mapper.TRUE_SCALE）
                ext[k, :3, 3] += rng.normal(0, cm / 100 / 2.7, 3)
        return server.Recon(ext, rec.points, rec.conf, rec.colors)

    return reconstruct


def test_edges_are_written_once_per_station_and_a_resend_replaces_them(tmp_path):
    c = _client(tmp_path)
    assert _post(c, _meta(0)).status_code == 200
    assert _rows(tmp_path) == []  # 1 か所だけでは辺が無い
    assert _post(c, _meta(1, prev_step=PREV)).status_code == 200
    rows = _rows(tmp_path)
    assert sorted(r["kind"] for r in rows) == ["odom", "window"]
    assert {r["station"] for r in rows} == {1}
    # 送り直し（同じ station_id）は、前の辺を置き換える
    assert _post(c, _meta(1, prev_step=PREV)).status_code == 200
    assert len(_rows(tmp_path)) == 2
    state = json.loads((tmp_path / SESSION / "graph_poses.json").read_text())
    assert state["localized"] == [0, 1] and state["fixed"] == 0
    plan = json.loads((tmp_path / SESSION / "plan.json").read_text())
    assert plan["pose_graph"]["station_localized"] is True
    assert "graph" in plan["timing_s"]
    # poses.json は今までの形のまま、グラフの位置
    poses = read_poses(tmp_path / SESSION)
    g = pg.read_graph_poses(tmp_path / SESSION)
    assert math.hypot(poses[1].x - g[1].x, poses[1].y - g[1].y) < 0.002


def test_odom_edge_falls_back_to_the_reply_or_reverse_json(tmp_path):
    session = tmp_path / "s"
    for i in (0, 1):
        (session / f"{i:03d}").mkdir(parents=True)
    prev = {"steer_deg": 0.0, "speed": 12, "est_distance_cm": 20.0}
    (session / "000" / "meta.json").write_text(json.dumps({"prev_step": None, "frames": []}))
    (session / "001" / "meta.json").write_text(json.dumps({"prev_step": prev, "frames": []}))
    assert server.odom_edge_for(session, 1).z.x == pytest.approx(0.2, abs=1e-3)
    (session / "reverse.json").write_text(json.dumps({"0": {"reverse": True}}))
    assert server.odom_edge_for(session, 1).z.x == pytest.approx(-0.2, abs=1e-3)
    assert server.odom_edge_for(session, 0) is None


def test_a_bad_last_station_is_placed_by_odometry_and_keeps_exploring(tmp_path):
    c = _client(tmp_path, scattered(1))
    assert _post(c, _meta(0, supports=CAN)).json()["action"] == "move"
    a = _post(c, _meta(1, prev_step=PREV, supports=CAN)).json()
    session = tmp_path / SESSION
    plan = json.loads((session / "plan.json").read_text())
    # この窓で 001 の位置合わせは崩れた（グラフが無ければ戻る）
    assert plan["pose_graph"]["station_localized"] and plan["pose_graph"]["station_odom_only"]
    assert not str(plan["next"].get("reason", "")).startswith("bad_registration")
    assert a["action"] == "move" and not a.get("retrace")
    # 位置は走った 1 歩（ハンドル -8.9° で 18 cm）から
    odom = pg.odom_pose([(-8.9, 0.18)])
    assert (a["pose"]["x"], a["pose"]["y"]) == pytest.approx((odom.x, odom.y), abs=0.01)
    assert 1 in read_poses(session)
    # 崩れたステーションの見たものは足さない
    assert not (session / "001" / "obs.npz").exists()
    assert json.loads((session / "graph_poses.json").read_text())["bad"] == [1]


def test_without_the_graph_the_same_bad_station_retraces(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "POSE_GRAPH", False)
    c = _client(tmp_path, scattered(1))
    _post(c, _meta(0, supports=CAN))
    a = _post(c, _meta(1, prev_step=PREV, supports=CAN)).json()
    assert a["action"] == "move" and a["retrace"] is True
    assert not (tmp_path / SESSION / "graph_edges.jsonl").exists()


def test_a_bad_station_the_graph_cannot_place_still_retraces(tmp_path):
    # 001 が無い（走った辺がつながらない）ので、002 はグラフで位置が決まらない
    c = _client(tmp_path, scattered(1))
    _post(c, _meta(0, supports=CAN))
    a = _post(c, _meta(2, prev_step=PREV, supports=CAN)).json()
    plan = json.loads((tmp_path / SESSION / "plan.json").read_text())
    assert plan["pose_graph"]["station_localized"] is False
    assert a["action"] == "move" and a["retrace"] is True


def test_observations_are_replaced_when_the_pose_changes(tmp_path):
    session = tmp_path / "s"
    (session / "000").mkdir(parents=True)
    old = Pose(1.0, 0.0, 0.0)
    cells = np.array([[1.5, 0.0], [1.0, 0.5]], np.float32)
    Observation(cells, cells[:1], np.zeros((0, 2)), old, 0.4).save(session / "000" / "obs.npz")
    same = read_observations(session, 0)[0]
    assert np.allclose(same.free, cells) and same.pose == old
    new = Pose(0.0, 1.0, math.pi / 2)
    o = read_observations(session, 0, {0: new})[0]
    # old から見て前へ 0.5 m だったセルは、new から見ても前へ 0.5 m（+y）
    assert np.allclose(o.free, [[0.0, 1.5], [-0.5, 1.0]], atol=1e-6)
    assert np.allclose(o.occupied, [[0.0, 1.5]], atol=1e-6)
    assert o.pose == new and o.sonar_m == 0.4
    assert len(o.high) == 0
    # グラフに位置の無いステーションは置き直さない
    assert np.allclose(read_observations(session, 0, {5: new})[0].free, cells)


def test_flag_off_and_a_failing_graph_answer_like_before(tmp_path, monkeypatch, caplog):
    """SNOO_POSE_GRAPH=0 と、グラフが落ちたとき（今までの決め方に戻る）は、同じ返事と位置。"""

    def run(root):
        c = _client(root)
        out = [_post(c, _meta(0, supports=CAN)).json()]
        out.append(_post(c, _meta(1, prev_step=PREV, supports=CAN)).json())
        return out, read_poses(root / SESSION)

    monkeypatch.setattr(server, "POSE_GRAPH", False)
    off, off_poses = run(tmp_path / "off")
    assert not list((tmp_path / "off" / SESSION).glob("graph_*"))
    plan = json.loads((tmp_path / "off" / SESSION / "plan.json").read_text())
    assert "pose_graph" not in plan and "graph" not in plan["timing_s"]
    # 今までの決め方：2 か所目は窓の VGGT の位置そのまま
    assert (off[1]["pose"]["x"], off[1]["pose"]["y"]) == pytest.approx((0.3, 0.1), abs=0.02)

    monkeypatch.setattr(server, "POSE_GRAPH", True)

    def boom(*a, **k):
        raise RuntimeError("作り物の失敗")

    monkeypatch.setattr(pg, "optimize", boom)
    with caplog.at_level(logging.ERROR, logger="snoo_map"):
        failed, failed_poses = run(tmp_path / "failed")
    assert failed == off
    assert failed_poses == off_poses
    # 同じセッションでは 1 回だけ書く
    assert sum("位置のグラフ" in r.getMessage() for r in caplog.records) == 1


def test_a_failure_while_replacing_observations_falls_back(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ValueError("作り物の失敗")

    monkeypatch.setattr(pg, "replace_observation", boom)
    c = _client(tmp_path)
    _post(c, _meta(0))
    r = _post(c, _meta(1, prev_step=PREV))
    assert r.status_code == 200 and r.json()["action"] == "move"


def _result(name, pose, spread=0.01):
    from app.snoo_mapper import StationResult

    return StationResult(name, pose, 0.1, spread, 10)


def test_a_graph_started_mid_session_holds_a_known_station_and_places_a_bad_last(tmp_path):
    """サーバーを入れ替えて途中から使い始めても、位置の分かったステーションに合わせて解く。
    最後のステーションの窓が崩れても、走った辺で位置が決まる。"""
    session = tmp_path / "s"
    step = {"steer_deg": 0.0, "speed": 12, "est_distance_cm": 30.0, "reverse": False}
    for i in range(4):
        (session / f"{i:03d}").mkdir(parents=True)
        meta = {"prev_step": step if i else None, "frames": []}
        (session / f"{i:03d}" / "meta.json").write_text(json.dumps(meta))
    known = {i: Pose(0.3 * i, 0.0, 0.0) for i in range(3)}
    # 窓の座標（原点は 001）。003 は首振りでカメラが 30 cm 散った
    results = [_result("001", Pose(0, 0, 0)), _result("002", Pose(0.3, 0, 0)),
               _result("003", Pose(2.0, 1.0, 1.0), spread=0.3)]  # fmt: skip
    g = server.solve_graph(session, 3, [1, 2, 3], results, [None] * 3, known)
    assert g.info["fixed"] == 1 and g.bad == {3}
    assert g.info["station_localized"] and g.info["station_odom_only"]
    assert (g.poses[1].x, g.poses[1].y) == pytest.approx((0.3, 0.0), abs=1e-6)
    assert (g.poses[3].x, g.poses[3].y) == pytest.approx((0.9, 0.0), abs=0.01)
    # 次のステーションも同じ所を止める（graph_poses.json の fixed・base）
    g2 = server.solve_graph(session, 3, [1, 2, 3], results, [None] * 3, {})
    assert g2.info["fixed"] == 1 and g2.poses[3].x == pytest.approx(0.9, abs=0.01)


def test_the_home_map_reads_the_replaced_observations(tmp_path):
    from app.snoo_home import load_session_grid

    session = tmp_path / "s"
    (session / "000").mkdir(parents=True)
    cells = np.array([[0.5, 0.0]], np.float32)
    empty = np.zeros((0, 2), np.float32)
    Observation(empty, cells, empty, Pose(0, 0, 0)).save(session / "000" / "obs.npz")
    before, _, _ = load_session_grid(session)
    state = {"localized": [0], "poses": {"0": [0.0, 1.0, 0.0]}}
    (session / "graph_poses.json").write_text(json.dumps(state))
    after, _, _ = load_session_grid(session)
    ix, iy = before.cell(np.array([0.5]), np.array([1.0]))
    assert after.logodds[ix[0], iy[0]] > 0 and before.logodds[ix[0], iy[0]] == 0


def _fake_roma(truth: Pose, calls: list):
    """RoMa の代わり：コマの名前のパン・チルトから、truth だけ動いた作り物の対応を返す。"""
    import re

    from test_snoo_roma import _matches

    def match(pairs):
        calls.append(pairs)
        out = []
        for pa, pb in pairs:
            (p1, t1), (p2, t2) = (
                map(int, re.search(r"pan([+-]\d+)_tilt([+-]\d+)", x).groups()) for x in (pa, pb)
            )
            ka, kb = _matches(truth, p1, t1, p2, t2)
            out.append({"ka": ka, "kb": kb, "cert": np.ones(len(ka))})
        return out

    return match


def test_roma_edges_are_added_only_with_the_flag(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(server, "roma_match", _fake_roma(Pose(0.25, 0.0, 0.1), calls))
    for on in (False, True):
        root = tmp_path / str(on)
        monkeypatch.setattr(server, "ROMA_EDGES", on)
        c = _client(root)
        assert _post(c, _meta(0)).status_code == 200
        assert _post(c, _meta(1, prev_step=PREV)).status_code == 200
        if not on:  # 既定は使わない
            assert calls == [] and "roma" not in {r["kind"] for r in _rows(root)}
    roma = [r for r in _rows(root) if r["kind"] == "roma"]
    assert len(calls) == 1 and len(calls[0]) == 4  # 前のステーションとだけ、コマ 4 組
    assert [(r["a"], r["b"], r["station"]) for r in roma] == [(0, 1, 1)]
    x, y, yaw = roma[0]["z"]
    assert abs(x - 0.25) < 0.02 and abs(y) < 0.02 and abs(yaw - 0.1) < 0.02


def test_a_failing_roma_keeps_the_station_going(tmp_path, monkeypatch, caplog):
    def broken(pairs):
        raise RuntimeError("GPU が無い")

    monkeypatch.setattr(server, "ROMA_EDGES", True)
    monkeypatch.setattr(server, "roma_match", broken)
    monkeypatch.setattr(server, "_roma_failed", set())
    c = _client(tmp_path)
    assert _post(c, _meta(0)).status_code == 200
    with caplog.at_level(logging.ERROR, logger="snoo_map"):
        assert _post(c, _meta(1, prev_step=PREV)).status_code == 200
    assert "RoMa の辺で落ちた" in caplog.text
    assert sorted(r["kind"] for r in _rows(tmp_path)) == ["odom", "window"]


def test_roma_revisit_candidate_is_near_and_far_in_order():
    from app import snoo_roma as sr

    prior = {i: Pose(0.3 * i, 0.0, 0.0) for i in range(6)}
    prior[6] = Pose(0.1, 0.05, 0.0)  # 0 の近くに戻ってきた
    assert sr.revisit_candidate(6, prior) == 0
    assert sr.revisit_candidate(3, prior) is None  # 番号が近いものは数えない
    prior[6] = Pose(5.0, 5.0, 0.0)
    assert sr.revisit_candidate(6, prior) is None  # 遠い
