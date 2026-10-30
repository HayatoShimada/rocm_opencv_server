"""app/snoo_home.py（家の地図）を、作り物の部屋の格子で試す。GPU は使わない。"""

import json
import math

import numpy as np
import pytest

from app import snoo_home
from app.roommap import FREE, OCCUPIED, Grid, Pose
from app.snoo_home import HomeMap, on_station, register, warp
from app.snoo_mapper import to_frame


def _l_room(x, y):
    """L 字の部屋（4 x 3 m に 1.5 x 1.5 m の出っ張り）と、箱と丸い台。
    0 見ていない・1 床・2 障害物。"""
    inside = ((-2 <= x) & (x <= 2) & (-1.5 <= y) & (y <= 1.5)) | (
        (0.5 <= x) & (x <= 2) & (1.5 <= y) & (y <= 3.0)
    )
    near = np.zeros_like(inside)
    for dx in (-0.08, 0, 0.08):
        for dy in (-0.08, 0, 0.08):
            near |= ((-2 <= x + dx) & (x + dx <= 2) & (-1.5 <= y + dy) & (y + dy <= 1.5)) | (
                (0.5 <= x + dx) & (x + dx <= 2) & (1.5 <= y + dy) & (y + dy <= 3.0)
            )
    wall = near & ~inside
    box = (-1.2 <= x) & (x <= -0.6) & (-0.8 <= y) & (y <= -0.4)
    disk = (x - 1.0) ** 2 + (y + 0.9) ** 2 <= 0.2**2
    out = np.zeros(np.shape(x), np.uint8)
    out[inside] = 1
    out[wall | (inside & (box | disk))] = 2
    return out


def _rect_room(x, y):
    """何も置いていない 4 x 2 m の部屋（180° 回すと同じに見える）。"""
    inside = (-2 <= x) & (x <= 2) & (-1 <= y) & (y <= 1)
    wall = (-2.1 <= x) & (x <= 2.1) & (-1.1 <= y) & (y <= 1.1) & ~inside
    out = np.zeros(np.shape(x), np.uint8)
    out[inside] = 1
    out[wall] = 2
    return out


def _grid(room, pose=None, seen=None, size=12.0, noise=0.0, seed=0):
    """room（家の座標の関数）を、pose（このグリッドの座標 → 家の座標）で見た格子。
    seen（家の座標の関数）が真のセルだけを見たことにする。noise の割合のセルを入れ替える。"""
    pose = pose or Pose(0, 0, 0)
    g = Grid(size)
    ix, iy = np.meshgrid(np.arange(g.n), np.arange(g.n), indexing="ij")
    x, y = g.center(ix, iy)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    hx, hy = pose.x + c * x - s * y, pose.y + s * x + c * y
    lab = room(hx, hy)
    if seen is not None:
        lab[~seen(hx, hy)] = 0
    if noise:
        rng = np.random.default_rng(seed)
        known = np.argwhere(lab > 0)
        flip = known[rng.random(len(known)) < noise]
        lab[flip[:, 0], flip[:, 1]] = 3 - lab[flip[:, 0], flip[:, 1]]
    g.logodds[lab == 1] = -1.5
    g.logodds[lab == 2] = 2.0
    return g


def _home(room):
    h = Grid(snoo_home.HOME_SIZE_M)
    full = _grid(room, size=snoo_home.HOME_SIZE_M)
    h.logodds[:] = full.logodds
    return h


def _err(a: Pose, b: Pose):
    d = math.degrees(a.yaw - b.yaw)
    return math.hypot(a.x - b.x, a.y - b.y), abs((d + 180) % 360 - 180)


@pytest.mark.parametrize(
    "truth",
    [
        Pose(0.8, -0.4, math.radians(37)),
        Pose(-0.3, 0.6, math.radians(-128)),
        Pose(0.0, 0.0, 0.0),
    ],
)
def test_registration_recovers_a_cropped_rotated_shifted_room(truth):
    home = _home(_l_room)
    # セッションは部屋の右上の一部（家の (0.8, 0.8) から 1.8 m 以内）だけを見た
    seen = lambda x, y: (x - 0.8) ** 2 + (y - 0.8) ** 2 <= 1.8**2  # noqa: E731
    sess = _grid(_l_room, truth, seen, noise=0.03)
    m, _ = register(sess, home)
    pos, deg = _err(m.pose, truth)
    print(
        f"truth={truth} got={m.pose} err={pos * 100:.1f}cm {deg:.2f}deg score={m.score:.2f} "
        f"margin={m.margin:.2f} overlap={m.overlap:.2f} {m.seconds:.2f}s"
    )
    assert m.ok, m
    assert pos <= 0.05 and deg <= 3.0
    assert m.overlap > 0.8


def test_a_symmetric_room_is_rejected():
    home = _home(_rect_room)
    truth = Pose(0.5, 0.2, math.radians(70))
    sess = _grid(_rect_room, truth)
    m, cands = register(sess, home)
    assert not m.ok, m
    assert len(cands) == 2 and m.margin < snoo_home.MIN_MARGIN


def test_a_session_from_another_room_is_not_registered():
    home = _home(_l_room)
    sess = _grid(lambda x, y: _rect_room(x * 0.6, y * 1.4), Pose(0, 0, 0.3))
    m, _ = register(sess, home)
    assert not m.ok


def test_warp_moves_a_cell_with_the_pose():
    g = Grid(12.0)
    ix, iy = g.cell(0.33, 0.12)
    g.logodds[ix, iy] = 2.0
    x, y = g.center(ix, iy)
    pose = Pose(1.0, -0.5, math.radians(90))
    out = warp(g.logodds, g.origin, pose, (480, 480), -12.0, g.res)
    p = to_frame(pose, Pose(float(x), float(y), 0))
    jx, jy = int((p.x + 12) // g.res), int((p.y + 12) // g.res)
    assert out[jx, jy] == pytest.approx(2.0)
    assert (out != 0).sum() == 1


def test_merge_adds_cells_and_can_be_redone(tmp_path):
    home = HomeMap(tmp_path / "home")
    sess = _grid(_l_room, Pose(0, 0, 0), lambda x, y: x < 0)
    assert (home.grid.state() != 0).sum() == 0
    home.merge("a", sess, None, Pose(0.0, 0.0, 0.0))
    n1 = int((home.grid.state() == FREE).sum())
    assert n1 > 500
    # 同じセッションを足し直しても 1 回分（ステーションごとにサーバーが足し直す）
    before = home.grid.logodds.copy()
    home.merge("a", sess, None, Pose(0.0, 0.0, 0.0))
    assert np.array_equal(before, home.grid.logodds)
    # 別のセッション（右半分）を足すと床が増える
    pose = Pose(0.4, 0.2, math.radians(30))
    other = _grid(_l_room, pose, lambda x, y: x > -0.5)
    home.merge("b", other, None, pose)
    assert int((home.grid.state() == FREE).sum()) > n1 * 1.5
    home.save()
    again = HomeMap.load(tmp_path / "home")
    assert set(again.sessions) == {"a", "b"} and again.open_session == "b"
    assert np.allclose(again.grid.logodds, home.grid.logodds, atol=0.01)
    assert (tmp_path / "home" / "home.png").is_file()


def test_moved_furniture_is_updated_by_newer_sessions(tmp_path):
    home = HomeMap(tmp_path / "home")
    ix, iy = home.grid.cell(0.5, 0.5)
    for k in range(5):  # 何回も障害物と見たセル
        g = Grid(12.0)
        g.logodds[g.cell(0.5, 0.5)] = 4.0
        home.merge(f"s{k}", g, None, Pose(0, 0, 0))
    assert home.grid.logodds[ix, iy] == pytest.approx(snoo_home.CAP)
    for k in range(2):  # 動かした後に 2 回床と見た
        g = Grid(12.0)
        g.logodds[g.cell(0.5, 0.5)] = -4.0
        home.merge(f"t{k}", g, None, Pose(0, 0, 0))
    assert home.grid.state()[ix, iy] == FREE


def test_islands_get_connected_when_a_bridging_session_is_merged(tmp_path):
    home = HomeMap(tmp_path / "home")
    # 家（a）は左上の角のまわりだけ。島（c）は部屋の右の 3/4 を見た。a は小さすぎて（角と壁
    # しか無く、c のどの角にも合う）合わせない
    a = _grid(_l_room, Pose(0, 0, 0), lambda x, y: (x + 1.6) ** 2 + (y - 1.0) ** 2 < 0.6**2)
    home.merge("a", a, None, Pose(0, 0, 0))
    truth_c = Pose(1.2, 0.3, math.radians(-50))
    c = _grid(_l_room, truth_c, lambda x, y: x > -0.8)
    m, _ = home.register(c)
    assert not m.ok, m
    home.add_island("c", c, None, m)
    home.save()
    assert set(HomeMap.load(tmp_path / "home").islands) == {"c"}
    # 部屋の左の大きな所を見たセッション（b）で家が広がると、島が合う
    truth_b = Pose(-0.2, -0.1, math.radians(25))
    b = _grid(_l_room, truth_b, lambda x, y: x < 1.3)
    mb, _ = home.register(b)
    home.merge("b", b, None, truth_b, mb)  # b の合わせ方はここでは見ない（a が小さいので）
    joined = home.connect_islands(skip="b")
    assert joined == ["c"]
    assert not home.islands and set(home.sessions) == {"a", "b", "c"}
    assert not (tmp_path / "home" / "islands" / "c.npz").exists()
    got = home.sessions["c"]["pose"]
    pos, deg = _err(Pose(got["x"], got["y"], math.radians(got["yaw_deg"])), truth_c)
    assert pos <= 0.05 and deg <= 3.0
    # 島は base にも足すので、いま足している b を足し直しても、c だけが見た所（右の端）は消えない
    only_c = home.grid.cell(1.7, 2.5)
    assert home.grid.state()[only_c] == FREE
    home.merge("b", b, None, truth_b, mb)
    assert home.grid.state()[only_c] == FREE


def test_union_uses_home_where_the_session_has_not_looked(tmp_path):
    home = HomeMap(tmp_path / "home")
    home.merge("a", _grid(_l_room), None, Pose(0, 0, 0))
    pose = Pose(0.5, 0.0, math.radians(90))
    sess = _grid(_l_room, pose, lambda x, y: (x - 0.5) ** 2 + y**2 < 0.5**2)
    u = home.union(sess, pose)
    st, s0 = u.state(), sess.state()
    assert (st == FREE).sum() > 3 * (s0 == FREE).sum()
    # 家の障害物はセッションの座標でも障害物（家の (1.0, -0.9) の丸い台）
    p = to_frame(snoo_home.inverse(pose), Pose(1.0, -0.9, 0))
    assert st[u.cell(p.x, p.y)] == OCCUPIED


def test_on_station_anchors_then_registers_and_writes_home_json(tmp_path):
    root = tmp_path / "home"
    s1, s2 = tmp_path / "sessions" / "s1", tmp_path / "sessions" / "s2"
    s1.mkdir(parents=True)
    s2.mkdir(parents=True)
    union, info = on_station(root, s1, 0, _grid(_l_room), None)
    assert info["registered"] and union is not None
    st = json.loads((s1 / "home.json").read_text())
    assert st["registered"] and st["reason"] == "anchor"
    truth = Pose(0.6, -0.3, math.radians(140))
    g = _grid(_l_room, truth, lambda x, y: (x - 0.8) ** 2 + (y - 0.8) ** 2 <= 1.8**2)
    union, info = on_station(root, s2, 0, g, None)
    assert info["registered"], json.loads((s2 / "home.json").read_text())
    st = json.loads((s2 / "home.json").read_text())
    got = Pose(st["pose"]["x"], st["pose"]["y"], math.radians(st["pose"]["yaw_deg"]))
    pos, deg = _err(got, truth)
    assert pos <= 0.05 and deg <= 3.0
    home = json.loads((root / "home.json").read_text())
    assert [s["id"] for s in home["sessions"]] == ["s1", "s2"]
    assert home["open_session"] == "s2"


def test_on_station_retries_then_leaves_an_island(tmp_path):
    root = tmp_path / "home"
    s1, s2 = tmp_path / "s1", tmp_path / "s2"
    s1.mkdir()
    s2.mkdir()
    on_station(root, s1, 0, _grid(_l_room), None)
    rng = np.random.default_rng(3)
    sym = Grid(12.0)  # どこにも合わない、でたらめな障害物だらけの所
    sym.logodds[100:140, 100:140] = rng.choice([-1.5, 2.0], (40, 40), p=[0.6, 0.4])
    for sid in range(6):
        union, info = on_station(root, s2, sid, sym, None)
        assert union is None and not info["registered"]
    st = json.loads((s2 / "home.json").read_text())
    assert [a["station"] for a in st["attempts"]] == list(snoo_home.RETRY_STATIONS)
    home = json.loads((root / "home.json").read_text())
    assert [s["id"] for s in home["islands"]] == ["s2"]
    assert (root / "islands" / "s2.npz").is_file()
