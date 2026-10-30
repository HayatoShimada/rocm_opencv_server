"""保存した探索のセッションで、ステーションの位置の出し方を比べる（正しい位置は無いので、代わりの物差しで）。

    .venv/bin/python -m scripts.snoo_pose_bakeoff data/snoo_map/sessions/explore_20261030_190244 \\
        --out /tmp/bakeoff [--gpu] [--traj vggt_slam=<trajectory.jsonl>] [--traj 名前=<traj.jsonl>]

比べるもの（--traj で足せる）:
- current：今の地図のサーバーが書いた poses.json（窓 4 か所の VGGT、最後のステーションだけ）
- odom：走った 1 歩（prev_step の区間と Pi の見積もりの長さ）を自転車モデルでつないだもの
- pg_live：位置のグラフ（app/snoo_posegraph.py）。辺は、地図のサーバーがもう回した窓
  （各ステーションの recon.npz）の全部の組と、走った辺。足す GPU の時間は無い
- pg_full：pg_live に、--gpu で別に VGGT に通した窓（続く 2 か所 + 走りながらのコマ、
  近くに戻ってきた 2 か所）を足したもの。戻ってきた組は半分（奇数番目）だけ使い、残りで確かめる
- pg_vision：pg_full の走った辺の確からしさを 5 倍ゆるくしたもの（走った辺との合い方を、
  走った辺にほぼ頼らずに測る）
- --traj 名前=ファイル：SLAM の道筋。行が {path, c2w}（4x4、OpenCV の向き。外の SLAM）か、
  app/snoo_slam.py の trajectory.jsonl の形（station・x・y・yaw）

物差し（ステーションごと）:
- consistent：首振りのカメラの中心のばらつきが 5 cm 未満で位置がある割合
- odom_t・odom_yaw：続く 2 か所の相対の位置と、走った 1 歩（自転車モデル）との差の中央値・90 %
- scale：1 歩の長さ ÷ 走った長さ の中央値と、そのばらつき（四分位の幅）
- loop：近くに戻ってきた組（ORB で選び、VGGT に 2 か所だけ通したもの。pg_full が使っていない半分）の
  相対の位置との差
- 地図：同じステーションの点（いちばんばらつきの小さい窓のもの）を、その出し方の位置に置いて
  格子を作る。ぽつんと 1 つだけの障害物のセルの割合（speckle）・床の広さ・ほかの 2 か所以上が床と
  見たセルに障害物を置いた割合（conflict）

GPU を使うのは --gpu のときだけ（VGGT。1 つずつ）。窓の答えは --out/extra/ に残し、
次からは読むだけ。
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from app import snoo_posegraph as pg
from app.roommap import FREE, OCCUPIED, Grid, Pose
from app.snoo_map_server import odom_edge_for, read_poses
from app.snoo_mapper import (
    FrameRef,
    MapParams,
    Station,
    build_map,
    load_station,
    session_grid,
    to_frame,
)

CONSISTENT_M = 0.05
# 近くに戻ってきたとみなす ORB の対応の数（snoo_home.MIN_ORB_MATCHES より厳しく）と、組の数の上限
LOOP_MIN_ORB = 40
LOOP_MAX_PAIRS = 40
LOOP_PER_STATION = 2
LOOP_MIN_GAP = 4  # これより近い番号のステーションは「戻ってきた」に数えない
BRIDGE_FRAMES = 4


# --- 読む ---


def station_list(session: Path) -> list[int]:
    return sorted(int(d.name) for d in session.iterdir() if d.is_dir() and d.name.isdigit())


def meta(session: Path, i: int) -> dict:
    return json.loads((session / f"{i:03d}" / "meta.json").read_text())


def odom_edges(session: Path, ids: list[int]) -> list[pg.Edge]:
    """続く 2 か所の間を走った辺（prev_step。区間を Pi の見積もりの長さで切る）。
    地図のサーバーがステーションごとに足すもの（odom_edge_for）と同じ。"""
    have = set(ids)
    return [e for b in ids if b - 1 in have and (e := odom_edge_for(session, b)) is not None]


def live_windows(session: Path, cache: Path) -> list[pg.Window]:
    if cache.is_file():
        return pickle.loads(cache.read_bytes())
    wins = []
    for i in station_list(session):
        p = session / f"{i:03d}" / "recon.npz"
        if not p.is_file():
            continue
        t = time.perf_counter()
        keys, ext, pts, conf = pg.load_recon(p)
        w = pg.solve_window(session, keys, ext, pts, conf, f"live:{i:03d}", keep=True)
        w.seconds = time.perf_counter() - t
        wins.append(w)
    cache.write_bytes(pickle.dumps(wins))
    return wins


# --- 近くに戻ってきた組（ORB） ---


def loop_candidates(session: Path, ids: list[int], cache: Path) -> list[tuple[int, int, int]]:
    """(a, b, ORB の対応の数)。番号が LOOP_MIN_GAP 以上離れた組のうち、対応の多いもの。"""
    if cache.is_file():
        return pick_loops([tuple(x) for x in json.loads(cache.read_text())])
    from app.snoo_home import station_images

    orb = cv2.ORB_create(500)
    feats = {}
    for i in ids:
        imgs = station_images(session, f"{i:03d}", limit=5)
        feats[i] = [orb.detectAndCompute(x, None) for x in imgs]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    out = []
    for x, a in enumerate(ids):
        for b in ids[x + 1 :]:
            if b - a < LOOP_MIN_GAP:
                continue
            best = 0
            for ka, da in feats[a]:
                for kb, db in feats[b]:
                    if da is None or db is None or len(da) < 8 or len(db) < 8:
                        continue
                    pairs = bf.knnMatch(da, db, k=2)
                    good = [m for m, *n in pairs if n and m.distance < 0.75 * n[0].distance]
                    if len(good) < 15:
                        continue
                    pa = np.float32([ka[m.queryIdx].pt for m in good])
                    pb = np.float32([kb[m.trainIdx].pt for m in good])
                    _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.99)
                    if mask is not None and mask.shape[0] == len(good):
                        best = max(best, int(np.count_nonzero(mask)))
            if best >= LOOP_MIN_ORB:
                out.append((a, b, best))
    cache.write_text(json.dumps(out))
    return pick_loops(out)


def pick_loops(pairs: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """対応の多い順に、1 つのステーションが LOOP_PER_STATION 回までになるよう選ぶ
    （最後に同じ所で切り返し続けた 66〜75 番の組ばかりにならないように）。"""
    count: dict[int, int] = {}
    out = []
    for a, b, n in sorted(pairs, key=lambda t: -t[2]):
        if count.get(a, 0) >= LOOP_PER_STATION or count.get(b, 0) >= LOOP_PER_STATION:
            continue
        out.append((a, b, n))
        count[a], count[b] = count.get(a, 0) + 1, count.get(b, 0) + 1
        if len(out) >= LOOP_MAX_PAIRS:
            break
    return out


# --- GPU で足す窓 ---


def pair_window_inputs(session: Path, a: int, b: int, bridge: bool):
    """2 か所の止まって撮ったコマ（と b へ走りながらのコマ）の Station・FrameRef・高さ。
    走りながらのコマは別の「ステーション」（高さ None。辺には使わない）にする。
    """
    sa, sb = (load_station(session / f"{i:03d}", "meta.json") for i in (a, b))
    still_a = Station(sa.dir, [f for f in sa.frames if not f.get("moving")])
    still_b = Station(sb.dir, [f for f in sb.frames if not f.get("moving")])
    groups = [(a, still_a)]
    if bridge:
        mov = [f for f in sb.frames if f.get("moving")]
        if len(mov) > BRIDGE_FRAMES:
            k = np.linspace(0, len(mov) - 1, BRIDGE_FRAMES).round().astype(int)
            mov = [mov[j] for j in k]
        if mov:
            groups.append((-1, Station(sb.dir, mov)))
    groups.append((b, still_b))
    stations, refs, ids, heights = [], [], [], []
    for si, (i, st) in enumerate(groups):
        stations.append(st)
        ids.append(i)
        heights.append(None if i < 0 else pg.camera_height(st.dir))
        for f in st.frames:
            refs.append(FrameRef(st.dir / f["file"], si, f["pan_deg"], f["tilt_deg"]))
    return stations, refs, ids, heights


def run_extra(session: Path, jobs: list[tuple[str, int, int, bool]], out: Path) -> dict:
    """jobs（名前, a, b, 走りながらのコマを入れるか）を VGGT に通して out/<名前>.npz に残す。"""
    from app.snoo_mapper import load_vggt_model, run_vggt, unproject

    todo = [j for j in jobs if not (out / f"{j[0]}.npz").is_file()]
    times = {}
    if not todo:
        return times
    model = load_vggt_model()
    params = MapParams()
    for name, a, b, bridge in todo:
        _, refs, _, _ = pair_window_inputs(session, a, b, bridge)
        t = time.perf_counter()
        v = run_vggt([r.path for r in refs], model)
        times[name] = time.perf_counter() - t
        pts, conf = unproject(v, params.stride, params.border_px)
        np.savez_compressed(
            out / f"{name}.npz", extrinsic=v.extrinsic.astype(np.float32),
            points=pts.astype(np.float32), conf=conf.astype(np.float32),
            seconds=np.array(times[name]), frames=np.array(len(refs)),
        )  # fmt: skip
        print(f"  {name}: {len(refs)} コマ {times[name]:.2f} 秒", flush=True)
    import torch

    del model  # 後の比べる所（CPU）の間、GPU を握らない
    torch.cuda.empty_cache()
    return times


def extra_window(session: Path, name: str, a: int, b: int, bridge: bool, out: Path) -> pg.Window:
    d = np.load(out / f"{name}.npz")
    stations, refs, ids, heights = pair_window_inputs(session, a, b, bridge)
    params = MapParams(camera_height_m=heights)
    r = build_map(stations, refs, d["extrinsic"], d["points"], d["conf"], params, plan=False)
    w = pg.Window(name, ids, r.stations, heights, float(d["seconds"]))
    w.map_result, w.points, w.conf, w.refs, w.extrinsic = (
        r, d["points"], d["conf"], refs, d["extrinsic"]
    )  # fmt: skip
    return w


# --- 道筋（SLAM）からステーションの位置へ ---


def _up_cam(tilt_deg: float) -> np.ndarray:
    t = math.radians(tilt_deg)
    return np.array([0.0, -math.cos(t), math.sin(t)])


def traj_stations(path: Path, session: Path) -> tuple[dict[int, Pose], dict[int, float], dict]:
    """道筋のファイルから、ステーションごとの (位置, 首振りのばらつき) と、ほかの情報。
    c2w の行（外の SLAM）は大きさが決まっていないので、1 歩の長さを走った長さに合わせて決め、
    ステーション 0 を原点にする。app/snoo_slam.py の形の行はそのまま（m、セッションの座標）。
    """
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    info: dict = {"rows": len(rows)}
    if rows and "c2w" in rows[0]:
        return _c2w_stations(rows, session, info)
    by: dict[int, list[dict]] = {}
    for r in rows:
        if r.get("source") == "still" and r.get("station") is not None:
            by.setdefault(int(r["station"]), []).append(r)
    poses, spread = {}, {}
    for s, rs in by.items():
        xy = np.array([[r["x"], r["y"]] for r in rs])
        c = xy.mean(0)
        head = min(rs, key=lambda r: (round(abs(r["tilt_deg"])), abs(r["pan_deg"])))
        poses[s] = Pose(float(c[0]), float(c[1]), float(head["yaw"]))
        z = np.array([r.get("z", 0.0) for r in rs])
        p3 = np.column_stack([xy, z])
        spread[s] = float(np.linalg.norm(p3 - p3.mean(0), axis=1).max())
    return poses, spread, info


def _c2w_stations(rows, session: Path, info: dict):
    seq = {}
    for i in station_list(session):
        for f in meta(session, i)["frames"]:
            seq[str(session / f"{i:03d}" / f["file"])] = (i, f)
    stills: dict[int, list] = {}
    ups = []
    for r in rows:
        key = str(Path(r["path"]))
        if key not in seq:
            continue
        i, f = seq[key]
        T = np.array(r["c2w"], float)
        ups.append(T[:3, :3] @ _up_cam(float(f["tilt_deg"])))
        if not f.get("moving"):
            stills.setdefault(i, []).append((T, f))
    if not ups:
        return {}, {}, info
    up = np.mean(ups, 0)
    up /= np.linalg.norm(up)
    # 床の向きの軸：ステーション 0（無ければ最初）の向きを x に
    first = min(stills)
    T0, f0 = min(stills[first], key=lambda t: (round(abs(t[1]["tilt_deg"])), abs(t[1]["pan_deg"])))
    ex = _heading(T0, up, f0["pan_deg"])
    ey = np.cross(up, ex)
    raw, spread_raw = {}, {}
    for i, ts in stills.items():
        C = np.array([T[:3, 3] for T, _ in ts])
        c = C.mean(0)
        T, f = min(ts, key=lambda t: (round(abs(t[1]["tilt_deg"])), abs(t[1]["pan_deg"])))
        h = _heading(T, up, f["pan_deg"])
        raw[i] = (np.array([c @ ex, c @ ey]), math.atan2(h @ ey, h @ ex))
        spread_raw[i] = float(np.linalg.norm(C - c, axis=1).max())
    # 大きさ：続く 2 か所の距離と走った長さの比の中央値
    od = {(e.a, e.b): math.hypot(e.z.x, e.z.y) for e in odom_edges(session, station_list(session))}
    ratios = []
    for (a, b), d in od.items():
        if a in raw and b in raw and d > 0.05:
            m = float(np.linalg.norm(raw[b][0] - raw[a][0]))
            if m > 0:
                ratios.append(d / m)
    scale = float(np.median(ratios)) if ratios else 1.0
    info["scale_from_odom"] = scale
    o = raw[first][0]
    th = raw[first][1]
    c, s = math.cos(-th), math.sin(-th)
    poses, spread = {}, {}
    for i, (p, yaw) in raw.items():
        q = scale * (p - o)
        poses[i] = Pose(c * q[0] - s * q[1], s * q[0] + c * q[1], float(pg.wrap(yaw - th)))
        spread[i] = scale * spread_raw[i]
    return poses, spread, info


def _heading(T: np.ndarray, up: np.ndarray, pan_deg: float) -> np.ndarray:
    fwd = T[:3, 2] - (T[:3, 2] @ up) * up
    from app.snoo_mapper import _rotate_about

    fwd = _rotate_about(fwd, up, math.radians(pan_deg))
    return fwd / np.linalg.norm(fwd)


def anchor_to(poses: dict[int, Pose], ref: Pose, at: int) -> dict[int, Pose]:
    """poses のステーション at が ref に来るよう、全体を回して動かす。"""
    if at not in poses:
        return poses
    T = to_frame(ref, pg.inverse(poses[at]))
    return {i: to_frame(T, p) for i, p in poses.items()}


# --- 物差し ---


def _q(v, q):
    return round(float(np.percentile(v, q)), 3) if len(v) else None


def metrics(
    poses: dict[int, Pose],
    spread: dict[int, float],
    ids: list[int],
    odom: list[pg.Edge],
    loops: list[pg.Edge],
) -> dict:
    n = len(ids)
    ok = [i for i in ids if i in poses and spread.get(i, 1.0) < CONSISTENT_M]
    dt, dyaw, ratio = [], [], []
    for e in odom:
        if e.a in poses and e.b in poses:
            r = pg.relative(poses[e.a], poses[e.b])
            dt.append(math.hypot(r.x - e.z.x, r.y - e.z.y))
            dyaw.append(abs(math.degrees(pg.wrap(r.yaw - e.z.yaw))))
            d = math.hypot(e.z.x, e.z.y)
            if d > 0.1:
                ratio.append(math.hypot(r.x, r.y) / d)
    lt, lyaw = [], []
    for e in loops:
        if e.a in poses and e.b in poses:
            r = pg.relative(poses[e.a], poses[e.b])
            lt.append(math.hypot(r.x - e.z.x, r.y - e.z.y))
            lyaw.append(abs(math.degrees(pg.wrap(r.yaw - e.z.yaw))))
    iqr = (_q(ratio, 75) - _q(ratio, 25)) if ratio else None
    return {
        "stations": n,
        "with_pose": sum(i in poses for i in ids),
        "consistent": round(len(ok) / n, 3),
        "odom_n": len(dt),
        "odom_t_med_cm": _cm(_q(dt, 50)),
        "odom_t_p90_cm": _cm(_q(dt, 90)),
        "odom_yaw_med_deg": _q(dyaw, 50),
        "odom_yaw_p90_deg": _q(dyaw, 90),
        "scale_med": _q(ratio, 50),
        "scale_iqr": None if iqr is None else round(iqr, 3),
        "loop_n": len(lt),
        "loop_t_med_cm": _cm(_q(lt, 50)),
        "loop_t_p90_cm": _cm(_q(lt, 90)),
        "loop_yaw_med_deg": _q(lyaw, 50),
    }


def _cm(v):
    return None if v is None else round(100 * v, 1)


def _mask(grid: Grid, xy: np.ndarray) -> np.ndarray:
    m = np.zeros((grid.n, grid.n), np.uint8)
    ix, iy = grid.cell(xy[:, 0], xy[:, 1])
    ok = grid.inside(ix, iy)
    m[ix[ok], iy[ok]] = 1
    return m


def pair_agreement(grid: Grid, obs: dict, pairs: list[tuple[int, int]]) -> float | None:
    """組（続く 2 か所と戻ってきた組。どの出し方でも同じ組）ごとに、a の障害物のセルのうち、b も
    見たセル（床か障害物、1 セル広げる）にあるものが、b の障害物（1 セル広げる）に重なる割合。
    両向きの平均の中央値。位置がずれていると、同じ壁が 2 枚になって下がる。
    """
    k = np.ones((3, 3), np.uint8)
    vals = []
    for a, b in pairs:
        if a not in obs or b not in obs:
            continue
        r = []
        for x, y in ((a, b), (b, a)):
            occ = _mask(grid, obs[x].occupied).astype(bool)
            ob = cv2.dilate(_mask(grid, obs[y].occupied), k).astype(bool)
            seen = cv2.dilate(_mask(grid, obs[y].free) | _mask(grid, obs[y].occupied), k)
            base = occ & seen.astype(bool)
            if base.sum() >= 20:
                r.append((base & ob).sum() / base.sum())
        if r:
            vals.append(float(np.mean(r)))
    return round(float(np.median(vals)), 3) if vals else None


def map_metrics(obs: list, poses: dict[int, Pose], pairs=None) -> tuple[dict, Grid]:
    grid, _ = session_grid([o for _, o in obs])
    state = grid.state()
    occ = state == OCCUPIED
    k = np.ones((3, 3), np.uint8)
    nb = cv2.filter2D(occ.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT) - occ
    speckle = int((occ & (nb == 0)).sum())
    # ほかの 2 か所以上が床と見たセルに、あるステーションが障害物を置いた割合
    free_votes = np.zeros(state.shape, np.int32)
    cell_sets = []
    for _, o in obs:
        fx, fy = grid.cell(o.free[:, 0], o.free[:, 1])
        ox, oy = grid.cell(o.occupied[:, 0], o.occupied[:, 1])
        f = np.zeros(state.shape, bool)
        okf = grid.inside(fx, fy)
        f[fx[okf], fy[okf]] = True
        free_votes += f
        cell_sets.append((f, ox, oy))
    occ_total = conflict = 0
    for f, ox, oy in cell_sets:
        ok = grid.inside(ox, oy)
        ox, oy = ox[ok], oy[ok]
        occ_total += len(ox)
        conflict += int(((free_votes[ox, oy] - f[ox, oy]) >= 2).sum())
    a = grid.res**2
    agree = pair_agreement(grid, dict(obs), pairs or [])
    return {
        "agree": agree,
        "free_m2": round(float((state == FREE).sum()) * a, 2),
        "occupied_cells": int(occ.sum()),
        "speckle": round(speckle / max(1, int(occ.sum())), 3),
        "conflict": round(conflict / max(1, occ_total), 3),
    }, grid


def render(grid: Grid, poses: dict[int, Pose], title: str, px: int = 4, half_m: float = 3.5):
    state = grid.state()
    n = int(2 * half_m / grid.res)
    c = grid.cell(np.array([-half_m]), np.array([-half_m]))
    i0, j0 = int(c[0][0]), int(c[1][0])
    sub = state[i0 : i0 + n, j0 : j0 + n]
    img = np.full((n, n, 3), 128, np.uint8)
    img[sub == FREE] = (235, 235, 235)
    img[sub == OCCUPIED] = (30, 30, 30)
    img = img.transpose(1, 0, 2)[::-1].copy()  # 行 = y（上が +）、列 = x
    img = cv2.resize(img, (n * px, n * px), interpolation=cv2.INTER_NEAREST)

    def to_px(x, y):
        return int((x + half_m) / grid.res * px), int((half_m - y) / grid.res * px)

    pts = [to_px(poses[i].x, poses[i].y) for i in sorted(poses)]
    for a, b in zip(pts, pts[1:], strict=False):
        cv2.line(img, a, b, (200, 140, 60), 1, cv2.LINE_AA)
    for i in sorted(poses):
        p = poses[i]
        u, v = to_px(p.x, p.y)
        cv2.circle(img, (u, v), 4, (0, 0, 220), -1, cv2.LINE_AA)
        e = to_px(p.x + 0.12 * math.cos(p.yaw), p.y + 0.12 * math.sin(p.yaw))
        cv2.line(img, (u, v), e, (0, 0, 220), 1, cv2.LINE_AA)
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# --- 本体 ---


def best_clouds(windows: list[pg.Window]) -> dict[int, pg.LocalCloud]:
    """ステーションごとに、いちばん首振りのばらつきが小さい窓の点。"""
    out: dict[int, pg.LocalCloud] = {}
    for w in windows:
        for k, i in enumerate(w.ids):
            if i < 0 or not w.good(k):
                continue
            if i in out and out[i].spread_m <= w.results[k].spread_m:
                continue
            c = pg.local_cloud(w, k)
            if c is not None:
                out[i] = c
    return out


def best_spread(windows: list[pg.Window], sol: pg.Solution) -> dict[int, float]:
    """外れ値でない窓の辺に出てくるステーションの、その窓でのばらつきの最小。"""
    used = {(e.source, i) for e in sol.edges if e.kind != "odom" and not e.outlier
            for i in (e.a, e.b)}  # fmt: skip
    out: dict[int, float] = {}
    for w in windows:
        for k, i in enumerate(w.ids):
            if (w.source, i) in used:
                out[i] = min(out.get(i, 1.0), w.results[k].spread_m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", action="store_true", help="足す窓を VGGT に通す（GPU）")
    ap.add_argument("--traj", action="append", default=[], help="名前=道筋のファイル")
    ap.add_argument(
        "--no-bridge", action="store_true", help="続く 2 か所に走りながらのコマを入れない"
    )
    args = ap.parse_args()
    session = Path(args.session).resolve()
    out = Path(args.out)
    (out / "extra").mkdir(parents=True, exist_ok=True)
    ids = station_list(session)
    odom = odom_edges(session, ids)
    print(f"ステーション {len(ids)} か所、走った辺 {len(odom)} 本")

    t = time.perf_counter()
    live = live_windows(session, out / "live_windows.pkl")
    live_s = [w.seconds for w in live]
    print(f"地図のサーバーの窓 {len(live)} 個（読むのに {time.perf_counter() - t:.1f} 秒）")

    cand = loop_candidates(session, ids, out / "loops.json")
    print(f"戻ってきた組の候補 {len(cand)}（ORB {LOOP_MIN_ORB} 以上）")
    bridge = not args.no_bridge
    jobs = [(f"pair_{a:03d}_{b:03d}{'_b' if bridge else ''}", a, b, bridge)
            for a, b in zip(ids, ids[1:], strict=False) if b == a + 1]  # fmt: skip
    jobs += [(f"loop_{a:03d}_{b:03d}", a, b, False) for a, b, _ in cand]
    gpu_times = {}
    if args.gpu:
        gpu_times = run_extra(session, jobs, out / "extra")
    extra = [extra_window(session, *j, out / "extra") for j in jobs
             if (out / "extra" / f"{j[0]}.npz").is_file()]  # fmt: skip
    pairs = [w for w in extra if w.source.startswith("pair_")]
    loop_wins = [w for w in extra if w.source.startswith("loop_")]
    print(f"足した窓：続く 2 か所 {len(pairs)}・戻ってきた組 {len(loop_wins)}")

    # 戻ってきた組：奇数番目を pg_full に使い、偶数番目で確かめる
    loop_use, loop_eval = [], []
    for k, w in enumerate(loop_wins):
        es = pg.window_edges(w)
        for e in es:
            e.kind = "loop"
        (loop_use if k % 2 else loop_eval).extend(es)
    print(f"戻ってきた辺：使う {len(loop_use)}・確かめる {len(loop_eval)}")

    live_edges = [e for w in live for e in pg.window_edges(w)]
    pair_edges = [e for w in pairs for e in pg.window_edges(w)]
    fixed = ids[0]
    methods: dict[str, tuple[dict[int, Pose], dict[int, float], dict]] = {}

    cur = read_poses(session)
    cur_spread = {}
    for w in live:  # その窓の最後のステーションのばらつき（地図のサーバーが見たもの）
        cur_spread[w.ids[-1]] = w.results[-1].spread_m
    methods["current"] = (cur, cur_spread, {"seconds_per_station": float(np.mean(live_s))})

    od_sol = pg.optimize([replace(e) for e in odom], fixed)
    methods["odom"] = (od_sol.poses, {}, {})

    def run_pg(name, edges, info=None):
        t = time.perf_counter()
        sol = pg.optimize([replace(e) for e in edges], fixed)
        sec = time.perf_counter() - t
        wins = live + (pairs + loop_wins if name != "pg_live" else [])
        sp = best_spread(wins, sol)
        n_out = sum(e.outlier for e in sol.edges)
        n_vis = sum(e.kind != "odom" for e in sol.edges)
        meta_info = {"solve_s": round(sec, 2), "edges": len(sol.edges), "vision_edges": n_vis,
                     "outliers": n_out, "seen": len(sol.seen), **(info or {})}  # fmt: skip
        methods[name] = (sol.poses, sp, meta_info)
        return sol

    run_pg("pg_live", live_edges + odom)
    sol_full = run_pg("pg_full", live_edges + pair_edges + loop_use + odom)
    loose = [replace(e, sigma_m=5 * e.sigma_m, sigma_rad=5 * e.sigma_rad) for e in odom]
    run_pg("pg_vision", live_edges + pair_edges + loop_use + loose)

    for spec in args.traj:
        name, path = spec.split("=", 1)
        p, sp, info = traj_stations(Path(path), session)
        methods[name] = (anchor_to(p, cur.get(fixed, Pose(0, 0, 0)), fixed), sp, info)

    clouds = best_clouds(live + pairs + loop_wins)
    agree_pairs = [(a, b) for a, b in zip(ids, ids[1:], strict=False) if b == a + 1]
    agree_pairs += [(a, b) for a, b, _ in cand]
    print(f"点のあるステーション {len(clouds)} か所")
    table, images = {}, []
    for name, (poses, spread, info) in methods.items():
        m = metrics(poses, spread, ids, odom, loop_eval)
        obs = [(i, pg.observation(clouds[i], poses[i])) for i in ids if i in poses and i in clouds]
        mm, grid = map_metrics(obs, poses, agree_pairs)
        table[name] = {**m, **mm, "info": info}
        img = render(grid, poses, name)
        cv2.imwrite(str(out / f"map_{name}.png"), img)
        images.append(img)
    rows = [images[k : k + 4] for k in range(0, len(images), 4)]
    w = max(len(r) for r in rows)
    blank = np.full_like(images[0], 255)
    grid_img = np.vstack([np.hstack(r + [blank] * (w - len(r))) for r in rows])
    cv2.imwrite(str(out / "maps_side_by_side.png"), grid_img)
    table["_gpu"] = {
        "pair_s_med": _q(list(gpu_times.values()), 50) if gpu_times else None,
        "live_window_build_s_med": _q(live_s, 50),
    }
    (out / "table.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))
    (out / "poses.json").write_text(json.dumps(
        {n: {str(i): [round(p.x, 4), round(p.y, 4), round(math.degrees(p.yaw), 2)]
             for i, p in v[0].items()} for n, v in methods.items()}, indent=1))  # fmt: skip
    outl = [{"a": e.a, "b": e.b, "kind": e.kind, "source": e.source, "chi": round(e.chi, 1)}
            for e in sol_full.edges if e.outlier]  # fmt: skip
    (out / "outliers_pg_full.json").write_text(json.dumps(outl, indent=1))
    print_table(table)


COLS = [
    ("with_pose", "pose"), ("consistent", "cons<5cm"), ("odom_t_med_cm", "odo_t50"),
    ("odom_t_p90_cm", "odo_t90"), ("odom_yaw_med_deg", "odo_y50"), ("odom_yaw_p90_deg", "odo_y90"),
    ("scale_med", "scale"), ("scale_iqr", "s_iqr"), ("loop_t_med_cm", "loop_t50"),
    ("loop_t_p90_cm", "loop_t90"), ("loop_yaw_med_deg", "loop_y50"), ("free_m2", "free_m2"),
    ("speckle", "speckle"), ("conflict", "conflict"), ("agree", "agree"),
]  # fmt: skip


def print_table(table: dict) -> None:
    print("| method | " + " | ".join(h for _, h in COLS) + " |")
    print("|---" * (len(COLS) + 1) + "|")
    for name, m in table.items():
        if name.startswith("_"):
            continue
        print(f"| {name} | " + " | ".join(str(m.get(k)) for k, _ in COLS) + " |")


if __name__ == "__main__":
    main()
