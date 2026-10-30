"""取り出した snoo の動き（scripts/snoo_behavior_extract.py）から、猫のモデルの数を作る。

  .venv/bin/python -m scripts.snoo_behavior_fit [--validate <絵の置き場>]
data/snoo_behavior/tracks/*.jsonl を読み、app/snoo_behavior.PRIOR_MODEL の数のうち録画で
分かるものを上書きして、data/snoo_behavior/model.json と app/snoo_behavior_model.json
（リポジトリに置く写し。50 KB 未満）に書く。数ごとに source（data か prior）と n を付ける。

- 姿勢：フレームごとに Clef の印（cat.teacher。0.6 秒以内の印を使う）、無ければ状態の分類器。
  5 秒の多数決でならし、同じ姿勢が続く間（bout）を作る。2 秒より短い bout は前の bout に
  まとめる（暗い部屋では Clef も lie と sleep・sit と lie の間で揺れるため）。両側が別の姿勢の
  bout だけ（場面の端や見失いで切れていない）を続く時間に使う。5 個以上あれば対数正規を当てる
- 移る確率：bout の境目を数え、PRIOR_MODEL の確率を 5 回分の重みで混ぜる（数が少ないので）
- 車への反応：車が動いた間（flags の MOVING）を、3 秒より短い止まりをはさんでまとめて 1 回
  とする（Pi は clear_cm が無いと 0.3 秒で止まるので、追う走りは細切れになる）。走り出す前に
  snoo が見えていた回だけを数える。その間と後 5 秒で、猫が歩いた（walk）か距離が 0.3 m 以上
  広がったなら retreat、そのまま居たなら freeze（その場で見ている）とする。見失っただけの回は
  車が向きを変えたせいのこともあるので、数えない（unknown）。Clef の向き（facing）は暗い
  低いカメラでは当たらない（正面の猫を away と言う）ので、freeze と ignore は分けない。
  猫が家具の上（下の端が床から 20 cm 以上）にいた回は別に数え、確率には入れない
- 距離は、車のカメラから猫の体の手前まで。モデルはカメラから猫の真ん中までなので 0.1 m 足す
--validate を渡すと、BehaviorSnoo を回して同じ数を出し、録画と比べる絵と数を書く。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from app.snoo_behavior import MODEL_PATH, POSTURES, PRIOR_MODEL

TRACKS = Path("data/snoo_behavior/tracks")
OUT = Path("data/snoo_behavior/model.json")
# 状態の分類器・Clef の答え → モデルの姿勢。食べる（eating）は家の中で短く、器も無いので stand
ACTIVITY = {
    "sleeping": "sleep",
    "lying": "lie",
    "sitting": "sit",
    "standing": "stand",
    "walking": "walk",
    "eating": "stand",
    "grooming": "groom",
}
SMOOTH_S = 5.0
GAP_S = 1.0  # これより長く見えなければ bout を切る
MIN_BOUT_S = 2.0
REST = {"lie": "rest", "sleep": "rest"}
EPISODE_GAP_S = 3.0
MIN_BOUTS = 5
PRIOR_WEIGHT = 5.0
HALF_BODY_M = 0.10
ELEVATED_M = 0.20


def load_tracks(root: Path) -> dict[str, list[dict]]:
    return {
        p.stem: [json.loads(line) for line in p.read_text().splitlines()]
        for p in sorted(root.glob("*.jsonl"))
    }


def frame_postures(rows: list[dict]) -> list[tuple[float, str | None, str]]:
    """フレームごとの (時刻, 姿勢（snoo が見えなければ None）, 印の出どころ)。"""
    taught = [(r["t"], r["cat"]["teacher"]) for r in rows if r["cat"] and r["cat"].get("teacher")]
    tt = np.array([t for t, _ in taught]) if taught else np.zeros(0)
    out = []
    for r in rows:
        c = r["cat"]
        if not c or not c["is_snoo"]:
            out.append((r["t"], None, ""))
            continue
        src, act = "head", c["state"]["activity"]
        if len(tt):
            k = int(np.abs(tt - r["t"]).argmin())
            if abs(tt[k] - r["t"]) <= 0.6:
                src, act = "teacher", taught[k][1]["activity"]
        out.append((r["t"], ACTIVITY[max(act, key=act.get)], src))
    return out


def smooth(seq: list[tuple[float, str | None, str]]) -> list[tuple[float, str | None]]:
    """SMOOTH_S 秒の多数決（見えていないフレームは数えない）。"""
    out = []
    for i, (t, p, _) in enumerate(seq):
        if p is None:
            out.append((t, None))
            continue
        votes = Counter(
            q
            for s, q, _ in seq[max(0, i - 80) : i + 81]
            if q is not None and abs(s - t) <= SMOOTH_S / 2
        )
        out.append((t, votes.most_common(1)[0][0]))
    return out


def bouts(seq: list[tuple[float, str | None]]) -> list[dict]:
    """同じ姿勢が続く間。start_cut・end_cut は、端が見失い・場面の端で切れたか。"""
    out: list[dict] = []
    cur = None
    last_t = None
    for t, p in seq:
        if p is None:
            continue
        gap = last_t is not None and t - last_t > GAP_S
        if cur is None or gap or p != cur["posture"]:
            if cur is not None:
                cur["end_cut"] = bool(gap)
                out.append(cur)
            cur = {"posture": p, "t0": t, "t1": t, "start_cut": cur is None or bool(gap)}
        cur["t1"] = t
        last_t = t
    if cur is not None:
        cur["end_cut"] = True
        out.append(cur)
    merged: list[dict] = []
    for b in out:
        prev = merged[-1] if merged else None
        short = b["t1"] - b["t0"] + 0.1 < MIN_BOUT_S
        if prev is not None and not b["start_cut"] and (short or b["posture"] == prev["posture"]):
            prev["t1"], prev["end_cut"] = b["t1"], b["end_cut"]
        else:
            merged.append(b)
    for b in merged:
        b["dur"] = round(b["t1"] - b["t0"] + 0.1, 2)
    return merged


def fit_lognorm(d: list[float]) -> tuple[float, float]:
    x = np.log(np.maximum(d, 0.05))
    return float(np.exp(np.median(x))), float(max(0.2, x.std(ddof=1) if len(x) > 1 else 0.5))


def reaction_events(name: str, rows: list[dict], post: dict[float, str | None]) -> list[dict]:
    """車が動いた回（短い止まりはまとめる）ごとの、そのときの猫と、その後 5 秒の様子。"""
    runs = []
    for k, r in enumerate(rows):
        if not (r["robot"] and r["robot"]["moving"]):
            continue
        if runs and r["t"] - rows[runs[-1][1]]["t"] <= EPISODE_GAP_S:
            runs[-1][1] = k
        else:
            runs.append([k, k])
    events = []
    for i, j in runs:
        t0, t1 = rows[i]["t"], rows[j]["t"]
        before = [x for x in rows[:i] if t0 - x["t"] <= 1.0 and x["cat"] and x["cat"]["is_snoo"]]
        d0 = [x["cat"]["dist_m"] for x in before if x["cat"].get("dist_m")]
        if not d0:
            continue
        c0 = before[-1]["cat"]
        dist = float(np.median(d0)) + HALF_BODY_M
        during = [x for x in rows[i : j + 1]]
        after = [x for x in rows[j + 1 :] if x["t"] - t1 <= 5.0]
        seen = [x for x in during + after if x["cat"] and x["cat"]["is_snoo"]]
        speeds = [x["robot"]["speed_m_s"] for x in during if x["robot"] and x["robot"]["moving"]]
        v = max(speeds, key=abs) if speeds else 0.0
        # 猫はたいてい画面の真ん中（首が追う）。前へ走れば近づく、後ろへなら離れる
        closing = max(
            (
                x["robot"]["speed_m_s"] * math.cos(math.radians(x["cat"]["bearing_deg"] or 0.0))
                for x in during
                if x["robot"] and x["robot"]["moving"] and x["cat"] and x["cat"]["is_snoo"]
            ),
            default=v * math.cos(math.radians(c0.get("bearing_deg") or 0.0)),
        )
        dists = [x["cat"]["dist_m"] + HALF_BODY_M for x in seen if x["cat"].get("dist_m")]
        p0 = post.get(before[-1]["t"])
        moved = any(post.get(x["t"]) == "walk" for x in seen) or (
            len(dists) >= 3 and float(np.median(dists[-5:])) - dist > 0.3
        )
        if moved:
            kind = "retreat"
        elif len(seen) >= 0.3 * max(len(during + after), 1):
            kind = "freeze"
        else:
            kind = "unknown"
        z = c0.get("floor_z_m")
        events.append(
            {
                "session": name,
                "t": round(t0, 1),
                "dur_s": round(t1 - t0 + 0.1, 1),
                "dist_m": round(dist, 2),
                "min_dist_m": round(min(dists + [dist]), 2),
                "robot_m_s": round(v, 2),
                "closing_m_s": round(closing, 2),
                "posture": p0,
                "kind": kind,
                "elevated": bool(z is not None and z > ELEVATED_M),
            }
        )
    return events


def speed_class(closing: float, robot: float, fast: float) -> str:
    return "fast" if closing > fast else ("slow" if abs(robot) > 0.02 else "still")


def fit(tracks: dict[str, list[dict]]) -> tuple[dict, dict]:
    model = copy.deepcopy(PRIOR_MODEL)
    summary: dict = {"sessions": {}, "label_src": Counter()}
    all_bouts, events = [], []
    seen_s: Counter = Counter()
    for name, rows in tracks.items():
        seq = frame_postures(rows)
        summary["label_src"].update(s for _, p, s in seq if p)
        sm = smooth(seq)
        post = dict(sm)
        # lie と sleep は 1 つの休み（rest）として bout を作る。暗い部屋の Clef は、同じ寝そべった
        # 猫を目の開け閉めで lie・sleep と行き来させ、数秒ごとの切り替えになってしまうため
        bs = bouts([(t, REST.get(p, p) if p else None) for t, p in sm])
        for b in bs:
            b["session"] = name
        all_bouts += bs
        # 見えていた時間（フレームの間隔で重みを付ける。録画は 10 fps、動画は 30 fps）
        ts = [t for t, p in sm if p]
        dt = float(np.median(np.diff(ts))) if len(ts) > 2 else 0.1
        for _, p in sm:
            if p:
                seen_s[p] += dt
        if any(r["robot"] for r in rows):
            events += reaction_events(name, rows, post)
        summary["sessions"][name] = {
            "frames": len(rows),
            "snoo_frames": len(ts),
            "postures": dict(Counter(p for _, p in sm if p)),
        }
    summary["label_src"] = dict(summary["label_src"])
    total = sum(seen_s.values())
    summary["seen_s"] = {k: round(v, 1) for k, v in seen_s.items()}

    # 姿勢の割合（見えていた時間の中で）。夜の 10 分ほどなので、1 日の割合ではない
    if total >= 120:
        share = {p: round(seen_s.get(p, 0.0) / total, 3) for p in POSTURES}
        model["share"] = {
            "value": share,
            "source": "data",
            "n": round(total),
            "why": "録画で snoo が見えていた秒の割合（2026-10-29 夜と動画）",
        }

    # 続く時間
    durs = defaultdict(list)
    cut = defaultdict(list)
    for b in all_bouts:
        (cut if b["start_cut"] or b["end_cut"] else durs)[b["posture"]].append(b["dur"])
    summary["bouts"] = {
        p: {
            "complete": len(durs[p]),
            "cut": len(cut[p]),
            "complete_median_s": round(float(np.median(durs[p])), 1) if durs[p] else None,
            "longest_cut_s": round(max(cut[p]), 1) if cut[p] else None,
        }
        for p in [*POSTURES, "rest"]
    }
    for p in POSTURES:
        d = durs[REST.get(p, p)]
        prior = model["postures"][p]["duration_s"]
        longest = max(cut[REST.get(p, p)] + d, default=0.0)
        if len(d) >= MIN_BOUTS:
            med, sig = fit_lognorm(d)
            if longest > 3 * med and p in ("lie", "sleep", "sit"):
                # 見えている間に終わる bout だけを使うと短いほうに偏る（長い休みは場面より長い）。
                # 切れた bout のほうが長いなら、事前の値と幾何平均で混ぜる
                med = math.sqrt(med * prior["median"])
                why = "終わりまで見えた bout（短いほうに偏る）と prior の幾何平均"
            else:
                why = "終わりまで見えた bout の対数正規"
            model["postures"][p]["duration_s"] = {
                "median": round(med, 2),
                "sigma": round(sig, 2),
                "source": "data",
                "n": len(d),
                "why": why,
            }
        else:
            prior["n"] = len(d)
            prior["longest_seen_s"] = round(longest, 1)

    # 移る確率（境目が見えたものだけ）
    trans = defaultdict(Counter)
    for a, b in zip(all_bouts, all_bouts[1:], strict=False):
        if a["session"] == b["session"] and not a["end_cut"] and not b["start_cut"]:
            trans[a["posture"]][b["posture"]] += 1
    summary["transitions"] = {k: dict(v) for k, v in trans.items()}
    for p in POSTURES:
        if p in REST:
            # 休みの中の lie と sleep の行き来は数えられない（上の理由）。prior のまま
            model["postures"][p]["next"]["n"] = 0
            continue
        cnt = Counter()
        for q, c in trans.get(p, Counter()).items():
            if q == "rest":
                # 休みへ移ったら、lie と sleep に prior の比で分ける
                pr = model["postures"][p]["next"]["value"]
                tot = pr.get("lie", 0.0) + pr.get("sleep", 0.0) or 1.0
                cnt["lie"] += c * pr.get("lie", 0.0) / tot
                cnt["sleep"] += c * pr.get("sleep", 0.0) / tot
            else:
                cnt[q] += c
        n = sum(cnt.values())
        prior = model["postures"][p]["next"]["value"]
        if n >= MIN_BOUTS:
            post = {
                q: (cnt.get(q, 0) + PRIOR_WEIGHT * prior.get(q, 0.0)) / (n + PRIOR_WEIGHT)
                for q in POSTURES
                if q != p
            }
            s = sum(post.values())
            model["postures"][p]["next"] = {
                "value": {q: round(v / s, 3) for q, v in post.items()},
                "source": "data",
                "n": round(n),
                "why": f"境目 {n} 個の数え（prior を {PRIOR_WEIGHT:g} 回分混ぜた）",
            }
        else:
            model["postures"][p]["next"]["n"] = n

    # 車への反応
    r = model["reaction"]
    summary["events"] = events
    floor = [e for e in events if not e["elevated"]]
    fast = r["fast_m_s"]["value"]
    tolerated = [e for e in floor if e["kind"] == "freeze" and e["closing_m_s"] > 0.05]
    summary["tolerated"] = [(e["min_dist_m"], e["closing_m_s"]) for e in tolerated]
    flight = r["flight_m"]["value"]
    inside = [e for e in tolerated if e["min_dist_m"] < flight]
    if inside:
        # 飛び退く距離より内で、車が近づいても逃げなかった。その速さでは「速い」に入らない
        speed = max(e["closing_m_s"] for e in inside)
        if speed >= fast:
            r["fast_m_s"] = {
                "value": round(speed + 0.05, 2),
                "source": "data",
                "n": len(inside),
                "why": f"{flight:g} m より内で {speed:.2f} m/s で近づかれても逃げなかった。"
                "速いのはこれより上",
            }
            fast = r["fast_m_s"]["value"]
    if tolerated:
        near = min(e["min_dist_m"] for e in tolerated)
        r["flight_m"]["why"] += (
            f"。録画では {fast:g} m/s 以下の行き来なら {near:.2f} m まで逃げなかった（矛盾しない）"
        )
    # その場で見ている長さ：freeze の回から、同じ場面で snoo が最後に見えたまで（場面の端で
    # 切れるので下限）。逃げた回がある場面は使わない
    spans = []
    for name, rows in tracks.items():
        mine = [e for e in floor if e["session"] == name]
        if mine and all(e["kind"] == "freeze" for e in mine):
            last = max(x["t"] for x in rows if x["cat"] and x["cat"]["is_snoo"])
            spans.append(last - mine[0]["t"])
    summary["freeze_spans_s"] = [round(x, 1) for x in spans]
    if len(spans) >= 3 and float(np.median(spans)) > r["freeze_s"]["median"]:
        r["freeze_s"] = {
            "median": round(float(np.median(spans)), 1),
            "sigma": 0.8,
            "source": "data",
            "n": len(spans),
            "why": "車が行き来する間、その場で座って見ていた長さ（場面の端で切れるので下限）",
        }
    by_class = defaultdict(Counter)
    for e in floor:
        if e["kind"] != "unknown":
            by_class[speed_class(e["closing_m_s"], e["robot_m_s"], fast)][e["kind"]] += 1
    summary["reaction_counts"] = {k: dict(v) for k, v in by_class.items()}
    summary["elevated_counts"] = dict(Counter(e["kind"] for e in events if e["elevated"]))
    choice = copy.deepcopy(r["choice"]["value"])
    srcs = {}
    for k, cnt in by_class.items():
        n = sum(cnt.values())
        if n < MIN_BOUTS:
            continue
        prior = choice[k]
        post = {
            q: (cnt.get(q, 0) + PRIOR_WEIGHT * prior.get(q, 0.0)) / (n + PRIOR_WEIGHT)
            for q in prior
        }
        s = sum(post.values())
        choice[k] = {q: round(v / s, 3) for q, v in post.items()}
        srcs[k] = n
    if srcs:
        r["choice"] = {
            "value": choice,
            "source": "data",
            "n": srcs,
            "why": "車が動いた回の、その間と後 5 秒の様子の数え（prior を "
            f"{PRIOR_WEIGHT:g} 回分混ぜた）。n の無い組は prior のまま。その場に居た回は freeze",
        }
    model["meta"] = {
        "note": "scripts/snoo_behavior_fit.py が data/snoo_behavior/tracks から作った",
        "sessions": len(tracks),
        "snoo_seen_s": round(total, 1),
        "events": len(events),
    }
    return model, summary


# --- 比べる ---


def simulate(model: dict, minutes: float, seed: int) -> dict:
    """広い部屋（4 x 4 m、真ん中に 1 x 1 m の箱）で、車が録画のように猫の近くで行き来する。

    車は 30 秒ごとに猫へ 0.2 m/s で 1 秒近づいて 1 秒戻るか（録画と同じ）、10 回に 1 回は
    0.5 m/s で猫へまっすぐ走る（逃げる距離を見るため）。
    """
    from app.roommap import Pose
    from app.snoo_behavior import BehaviorPose, BehaviorSnoo

    def free(x, y):
        if not (0.15 <= x <= 3.85 and 0.15 <= y <= 3.85):
            return False
        return not (1.35 <= x <= 2.65 and 1.35 <= y <= 2.65)

    rng = np.random.default_rng(seed)
    snoo = BehaviorSnoo(BehaviorPose(0.6, 0.6, 0.0), free, rng, model=model)
    dt = 0.1
    steps = int(minutes * 60 / dt)
    shares: Counter = Counter()
    speeds = []
    robot = Pose(3.4, 3.4, math.radians(-135))
    plan: list[float] = []
    for k in range(steps):
        if not plan and k % 300 == 0:
            if rng.random() < 0.1:
                plan = [0.5] * 40
            else:
                plan = [0.2] * 10 + [-0.2] * 10
        v = plan.pop(0) if plan else 0.0
        yaw = math.atan2(snoo.y - robot.y, snoo.x - robot.x) if v > 0 else robot.yaw
        nx, ny = robot.x + v * dt * math.cos(yaw), robot.y + v * dt * math.sin(yaw)
        if math.hypot(nx - snoo.x, ny - snoo.y) > 0.35 and free(nx, ny):
            robot = Pose(nx, ny, yaw)
        else:
            plan = []
        x0, y0 = snoo.x, snoo.y
        p = snoo.step(dt, robot)
        shares[p.posture] += dt
        if p.posture == "walk" and snoo.mode == "calm":
            speeds.append(math.hypot(snoo.x - x0, snoo.y - y0) / dt)
    total = sum(shares.values())
    return {
        "share": {k: v / total for k, v in shares.items()},
        "walk_speeds": speeds,
        "reactions": snoo.reactions,
    }


BLUE, ORANGE, RED, GREEN, GRAY = (
    (167, 121, 78),
    (43, 142, 242),
    (89, 87, 225),
    (79, 161, 89),
    (
        172,
        176,
        186,
    ),
)


def _text(img, text, xy, scale=0.45, color=(0, 0, 0)):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw(data_share, sim_share, speeds, summary, reacts) -> np.ndarray:
    """3 つの絵を横に並べる：姿勢の割合・歩く速さ・反応の距離と車の速さ。"""
    W, H, m = 420, 320, 45
    img = np.full((H, 3 * W, 3), 255, np.uint8)
    # 1. 姿勢の割合（青 録画、橙 BehaviorSnoo）
    names = [*POSTURES, "run", "sniff"]
    bw = (W - 2 * m) / len(names)
    for i, k in enumerate(names):
        x = int(m + i * bw)
        for j, (v, c) in enumerate(((data_share.get(k, 0), BLUE), (sim_share.get(k, 0), ORANGE))):
            h = int(v * (H - 2 * m))
            x0 = x + j * int(bw / 2) + 2
            cv2.rectangle(img, (x0, H - m - h), (x0 + int(bw / 2) - 3, H - m), c, -1)
        _text(img, k, (x, H - m + 15), 0.38)
    _text(img, "posture share: blue=recordings orange=sim", (10, 20))
    cv2.line(img, (m, H - m), (W - m, H - m), (0, 0, 0), 1)
    for v in (0.25, 0.5):
        y = int(H - m - v * (H - 2 * m))
        _text(img, f"{v:g}", (5, y + 4), 0.38)
    # 2. 歩く速さ（BehaviorSnoo。録画の歩きは少なすぎて出せない）
    ox = W
    _text(img, "walk speed m/s (sim; recordings: too few walks)", (ox + 10, 20))
    if len(speeds):
        hist, edges = np.histogram(speeds, bins=24, range=(0, 0.8))
        hmax = max(hist.max(), 1)
        bw = (W - 2 * m) / len(hist)
        for i, h in enumerate(hist):
            x0 = int(ox + m + i * bw)
            cv2.rectangle(img, (x0, int(H - m - h / hmax * (H - 2 * m))), (int(x0 + bw) - 1, H - m),
                          ORANGE, -1)  # fmt: skip
        for v in (0.0, 0.2, 0.4, 0.6, 0.8):
            _text(img, f"{v:.1f}", (int(ox + m + v / 0.8 * (W - 2 * m)) - 8, H - m + 15), 0.38)
    cv2.line(img, (ox + m, H - m), (ox + W - m, H - m), (0, 0, 0), 1)
    # 3. 反応：距離（横）と車の近づく速さ（縦）。丸 録画、x BehaviorSnoo
    ox = 2 * W
    _text(img, "reactions: o=recordings x=sim", (ox + 10, 20))
    _text(img, "freeze", (ox + 260, 40), 0.4, BLUE)
    _text(img, "retreat", (ox + 260, 55), 0.4, ORANGE)
    _text(img, "flee", (ox + 260, 70), 0.4, RED)
    _text(img, "approach", (ox + 260, 85), 0.4, GREEN)
    _text(img, "ignore", (ox + 260, 100), 0.4, GRAY)
    colors = {"freeze": BLUE, "retreat": ORANGE, "flee": RED, "approach": GREEN, "ignore": GRAY}

    def P(d, v):
        return (int(ox + m + min(d, 2.5) / 2.5 * (W - 2 * m)),
                int(H - m - min(max(v, 0), 0.8) / 0.8 * (H - 2 * m)))  # fmt: skip

    for e in reacts:
        x, y = P(e["dist_m"], e["closing_m_s"])
        c = colors.get(e["kind"], GRAY)
        cv2.line(img, (x - 3, y - 3), (x + 3, y + 3), c, 1)
        cv2.line(img, (x - 3, y + 3), (x + 3, y - 3), c, 1)
    for e in summary["events"]:
        if not e["elevated"] and e["kind"] != "unknown":
            cv2.circle(img, P(e["min_dist_m"], e["closing_m_s"]), 5, colors[e["kind"]], 2)
    cv2.line(img, (ox + m, H - m), (ox + W - m, H - m), (0, 0, 0), 1)
    cv2.line(img, (ox + m, H - m), (ox + m, m), (0, 0, 0), 1)
    for d in (0.5, 1.0, 1.5, 2.0, 2.5):
        _text(img, f"{d:g}", (P(d, 0)[0] - 8, H - m + 15), 0.38)
    for v in (0.2, 0.4, 0.6, 0.8):
        _text(img, f"{v:g}", (ox + 10, P(0, v)[1] + 4), 0.38)
    _text(img, "distance camera->cat centre (m)", (ox + m + 40, H - 8), 0.4)
    _text(img, "closing m/s", (ox + 2, m - 8), 0.4)
    return img


def validate(model: dict, summary: dict, out: Path, minutes: float = 120, seeds: int = 4) -> dict:
    runs = [simulate(model, minutes, s) for s in range(seeds)]
    share = Counter()
    for r in runs:
        for k, v in r["share"].items():
            share[k] += v / len(runs)
    speeds = np.concatenate([r["walk_speeds"] for r in runs]) if runs else np.zeros(0)
    reacts = [e for r in runs for e in r["reactions"]]
    seen = summary["seen_s"]
    tot = sum(seen.values()) or 1.0
    data_share = {k: seen.get(k, 0.0) / tot for k in POSTURES}
    kinds = Counter(e["kind"] for e in reacts)
    flee_d = [e["dist_m"] for e in reacts if e["kind"] == "flee"]
    res = {
        "sim_share": {k: round(share.get(k, 0.0), 3) for k in sorted(share)},
        "data_share": {k: round(v, 3) for k, v in data_share.items()},
        "sim_walk_speed_pct": [round(float(v), 2) for v in np.percentile(speeds, [10, 50, 90])]
        if len(speeds)
        else None,
        "sim_reactions": dict(kinds),
        "sim_flee_dist_pct": [round(float(v), 2) for v in np.percentile(flee_d, [10, 50, 90])]
        if flee_d
        else None,
        "data_reactions": summary["reaction_counts"],
        "data_event_dist_m": [e["dist_m"] for e in summary["events"] if not e["elevated"]],
    }
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(out / "behavior_validate.png"), draw(data_share, share, speeds, summary, reacts)
    )
    (out / "behavior_validate.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracks", type=Path, default=TRACKS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--repo-copy", type=Path, default=MODEL_PATH)
    parser.add_argument("--validate", type=Path, help="比べた絵と数の置き場")
    args = parser.parse_args()

    model, summary = fit(load_tracks(args.tracks))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, ensure_ascii=False, indent=1))
    (args.out.parent / "fit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1)
    )
    text = json.dumps(model, ensure_ascii=False, indent=1)
    if len(text.encode()) < 50_000:
        args.repo_copy.write_text(text + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("events", "sessions")},
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"{args.out}（{len(text.encode())} バイト）")
    if args.validate:
        print(json.dumps(validate(model, summary, args.validate), ensure_ascii=False))


if __name__ == "__main__":
    main()
