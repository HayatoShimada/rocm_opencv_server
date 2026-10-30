"""本物の snoo に似せたシミュレーターの猫（BehaviorSnoo）。録画から取った動きの数で動かす。

RandomWalkSnoo（app/snoo_sim.py）は止まる・歩くを繰り返し、車には 0.5 m で離れるだけだった。
それでは探り方を学んでも、猫にぶつかる数が減らない。
ここでは scripts/snoo_behavior_fit.py が録画（scripts/snoo_behavior_extract.py）から作った
数（MODEL。app/snoo_behavior_model.json）で、次のように動かす。

- 姿勢：walk・stand・sit・lie・sleep・groom を、半マルコフ（姿勢ごとの続く時間の分布と、
  次の姿勢への移る確率）で変える。寝ている猫は長く寝て、気づくのも遅い
- 歩く：速さは 1 回歩くごとに分布から引き、向きは少しずつ揺らす。前がふさがっていれば、
  空いているほうへ曲がる（壁に沿って歩く形になる）。free(x, y) が False の所には入らない
- 車への反応：車（カメラ）までの距離と、車がこちらへ近づく速さで決める。
  気づく（notice）→ 固まる（freeze）・近づいて嗅ぐ（approach → sniff）・離れる（retreat）・
  逃げる（flee）・気にしない（ignore）。飛び退く距離（flight_m）より内に速く近づかれたら
  必ず逃げる。何度か何も起きなければ慣れて、逃げにくくなる（habituation）

数ごとに source（"data" か "prior"）を持たせる。録画で足りない数は、猫の行動の本や
常識からの見積もり（prior）で、PRIOR_MODEL に理由を書いておく。

mujoco も cv2 も読み込まない（app/snoo_gridsim.py の速い 2 次元のシミュレーターでも使うため）。
姿勢（SnooPose.posture）は MuJoCo の猫の体（app/snoo_sim.py）が見た目を変えるのに使う。
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.roommap import Pose

MODEL_PATH = Path(__file__).with_name("snoo_behavior_model.json")
# 落ち着いているときの姿勢。反応の間は run（逃げる）・sniff（嗅ぐ）・stand（固まる）も使う
POSTURES = ("walk", "stand", "sit", "lie", "sleep", "groom")
REACTIONS = ("flee", "retreat", "freeze", "approach", "ignore")
# 車の大きさの円（app/roommap.CAR_RADIUS_M と同じ）と猫の体の円。中心がこれより近づかない
CAR_R_M, SNOO_R_M = 0.17, 0.15
# 車の真ん中は、カメラ（渡される姿勢）から少し後ろ（app/snoo_sim.RandomWalkSnoo と同じ）
CAR_CENTER_BACK_M = 0.08
# 内側の刻み（秒）。長い dt はこれに分ける（反応が遅れないように）
SUB_DT_S = 0.1


def _p(value, source: str = "prior", why: str = "", **extra) -> dict:
    return {"value": value, "source": source, **({"why": why} if why else {}), **extra}


def _lognorm(median: float, sigma: float, source: str = "prior", why: str = "") -> dict:
    """続く時間などの対数正規分布（median と、log の標準偏差 sigma）。"""
    return {"median": median, "sigma": sigma, "source": source, **({"why": why} if why else {})}


# 録画が足りない数の見積もり。猫の行動の本（Bradshaw「Cat Sense」、Turner & Bateson
# 「The Domestic Cat」）や、家の猫の一般的な様子から。臆病な長毛の猫として決めた
PRIOR_MODEL: dict[str, Any] = {
    "version": 1,
    "postures": {
        "walk": {
            "duration_s": _lognorm(4.0, 0.7, why="家の中の 1 回の歩きは数秒〜十数秒"),
            "next": _p({"stand": 0.35, "sit": 0.35, "lie": 0.15, "groom": 0.10, "sleep": 0.05}),
        },
        "stand": {
            "duration_s": _lognorm(3.0, 0.8, why="立ち止まりは数秒"),
            "next": _p({"walk": 0.55, "sit": 0.30, "lie": 0.10, "groom": 0.05}),
        },
        "sit": {
            "duration_s": _lognorm(30.0, 1.0, why="座って見ているのは数十秒〜数分"),
            "next": _p({"walk": 0.35, "lie": 0.30, "groom": 0.20, "stand": 0.15}),
        },
        "lie": {
            "duration_s": _lognorm(120.0, 1.0, why="寝そべりは数分"),
            "next": _p({"sleep": 0.35, "sit": 0.30, "groom": 0.20, "walk": 0.10, "stand": 0.05}),
        },
        "sleep": {
            "duration_s": _lognorm(900.0, 0.8, why="猫は 1 日 12〜16 時間寝て、1 回は数十分"),
            "next": _p({"lie": 0.50, "groom": 0.25, "sit": 0.15, "walk": 0.10}),
        },
        "groom": {
            "duration_s": _lognorm(40.0, 0.9, why="毛づくろいは 1 回 数十秒〜数分"),
            "next": _p({"lie": 0.35, "sit": 0.30, "walk": 0.25, "sleep": 0.10}),
        },
    },
    "share": _p(
        {"walk": 0.04, "stand": 0.03, "sit": 0.13, "lie": 0.25, "sleep": 0.45, "groom": 0.10},
        why="起きている時間の大半は休み。歩くのは数 %",
    ),
    "walk": {
        "speed_m_s": _lognorm(0.25, 0.35, why="猫の歩きは 0.2〜0.5 m/s（家の中ではゆっくり）"),
        "turn_sd_deg_s": _p(25.0, why="歩く向きの揺れ。家具の間をゆっくり曲がる"),
        "run_speed_m_s": _p(1.2, why="驚いて逃げる猫は 1〜2 m/s。家の中は狭いので低め"),
        "max_speed_m_s": _p(2.0),
    },
    "reaction": {
        "notice_m": _p(
            {"awake": 2.0, "sleep": 0.6},
            why="起きていれば部屋の向こうの動く物に気づく。寝ていると音か近さで",
        ),
        "flight_m": _p(0.7, why="臆病な猫が速く近づく物から飛び退く距離 0.5〜1 m"),
        "fast_m_s": _p(0.10, why="これより速く近づけば速い。車の speed 12（0.2 m/s）は速い"),
        # 気づいたときの反応の確率（車の近づき方ごと）
        "choice": _p(
            {
                "fast": {
                    "flee": 0.55,
                    "retreat": 0.25,
                    "freeze": 0.15,
                    "approach": 0.0,
                    "ignore": 0.05,
                },
                "slow": {
                    "flee": 0.15,
                    "retreat": 0.30,
                    "freeze": 0.30,
                    "approach": 0.10,
                    "ignore": 0.15,
                },
                "still": {
                    "flee": 0.0,
                    "retreat": 0.10,
                    "freeze": 0.25,
                    "approach": 0.25,
                    "ignore": 0.40,
                },
            },
            why="臆病な猫：動く物から離れ、止まっている物には少しずつ近づいて嗅ぐ",
        ),
        "freeze_s": _lognorm(3.0, 0.6, why="固まって見つめるのは数秒"),
        "sniff_m": _p(0.30, why="嗅ぐときは鼻が届くまで（体の中心で 30 cm ほど）"),
        "sniff_s": _lognorm(4.0, 0.6, why="嗅ぐのは数秒"),
        "approach_speed_m_s": _lognorm(0.15, 0.3, why="知らない物にはそろそろと近づく"),
        "retreat_m": _p(1.2, why="ゆっくり離れて、この距離まで空ける"),
        "safe_m": _p(1.8, why="逃げたら、この距離まで離れる（部屋の外へ出ることもある）"),
        "cooldown_s": _p(8.0, why="反応の後しばらくは同じ車に反応し直さない"),
        "habituation": _p(0.85, why="何も起きなかった反応 1 回ごとに逃げる確率を掛ける"),
        "habituation_floor": _p(0.35, why="慣れても逃げる確率はこれより下げない"),
    },
    "meta": {"note": "PRIOR_MODEL のまま（録画からの数で上書きしていない）"},
}


def load_model(path: str | Path | None = None) -> dict:
    """数を読み込む。無い項目は PRIOR_MODEL で埋める（古い model.json でも動くように）。"""
    model = copy.deepcopy(PRIOR_MODEL)
    p = Path(path) if path is not None else MODEL_PATH
    if p.exists():
        _merge(model, json.loads(p.read_text()))
    return model


def _merge(base: dict, new: dict) -> None:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and "source" not in v:
            _merge(base[k], v)
        else:
            base[k] = v


def _val(d):
    return d["value"] if isinstance(d, dict) and "value" in d else d


def _sample_lognorm(rng: np.random.Generator, d: dict, lo: float = 0.0, hi: float = math.inf):
    return float(min(hi, max(lo, d["median"] * math.exp(d["sigma"] * rng.standard_normal()))))


def _choice(rng: np.random.Generator, probs: dict[str, float]) -> str:
    keys = list(probs)
    p = np.array([max(probs[k], 0.0) for k in keys], float)
    if p.sum() <= 0:
        return keys[0]
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class BehaviorPose:
    """app/snoo_sim.SnooPose と同じ欄（あちらは mujoco を読み込むので、2 次元のときはこれ）。"""

    x: float
    y: float
    yaw: float
    moving: bool = False
    posture: str = "walk"


class BehaviorSnoo:
    """MODEL の数で動く猫。app/snoo_sim.SnooModel の形（step(dt, robot) -> SnooPose）。

    free(x, y)：猫の体の中心がそこに立てるか（壁・家具・部屋の外でないか）。車の円は
    ここで別に避ける。pose_cls：返す姿勢の型（app/snoo_sim.SnooPose。x・y・yaw・moving・
    posture を受ける）。読み込むと mujoco が要るので、呼ぶ側から渡す。
    同じ seed（rng）と同じ車の動きなら、いつも同じ動きになる。
    """

    def __init__(
        self,
        start,
        free: Callable[[float, float], bool],
        rng: np.random.Generator | None = None,
        model: dict | str | Path | None = None,
        pose_cls=None,
        posture: str | None = None,
    ):
        self.m = model if isinstance(model, dict) else load_model(model)
        self.free = free
        self.rng = rng or np.random.default_rng(1)
        self.pose_cls = pose_cls or type(start)
        self.x, self.y, self.yaw = float(start.x), float(start.y), float(start.yaw)
        self.t = 0.0
        share = {k: v for k, v in _val(self.m["share"]).items() if k in POSTURES}
        self.posture = posture or _choice(self.rng, share)
        # 始めの姿勢は、途中から見ている（続く時間の残りは一様に引く）
        self.timer = self._duration(self.posture) * float(self.rng.uniform(0.1, 1.0))
        self.speed = self._walk_speed()
        self.heading = self.yaw
        self.mode = "calm"  # calm・freeze・approach・sniff・retreat・flee
        self.mode_timer = 0.0
        self.cooldown = 0.0
        self.flee_scale = 1.0
        self.reactions: list[dict] = []  # 反応の記録（時刻・種類・距離・車の近づく速さ）
        self._robot: tuple[float, float, float] | None = None  # 前の (t, x, y)
        self._robot_v = (0.0, 0.0)
        self._away_dir: float | None = None
        self._reacted_at: tuple[float, float] | None = None  # 前に反応したときの車の真ん中
        self.pose = self._make_pose(False)

    # --- 数 ---

    def _duration(self, posture: str) -> float:
        return _sample_lognorm(self.rng, self.m["postures"][posture]["duration_s"], 0.5, 7200)

    def _walk_speed(self) -> float:
        hi = _val(self.m["walk"]["max_speed_m_s"])
        return _sample_lognorm(self.rng, self.m["walk"]["speed_m_s"], 0.05, hi)

    def _next_posture(self) -> str:
        nxt = dict(_val(self.m["postures"][self.posture]["next"]))
        nxt.pop(self.posture, None)
        return _choice(self.rng, {k: v for k, v in nxt.items() if k in POSTURES})

    def _make_pose(self, moving: bool):
        posture = {
            "flee": "run",
            "approach": "walk",
            "retreat": "walk",
            "sniff": "sniff",
            "freeze": "stand",
        }.get(self.mode, self.posture)
        return self.pose_cls(self.x, self.y, self.yaw, moving, posture)

    # --- 車 ---

    def _see_robot(self, dt: float, robot: Pose | None):
        """車の真ん中・カメラの位置と、車の速さ（前の刻みとの差。少しならす）。"""
        if robot is None:
            self._robot = None
            return None
        cx = robot.x - CAR_CENTER_BACK_M * math.cos(robot.yaw)
        cy = robot.y - CAR_CENTER_BACK_M * math.sin(robot.yaw)
        if self._robot is not None and dt > 0:
            _, px, py = self._robot
            vx, vy = (cx - px) / dt, (cy - py) / dt
            a = min(1.0, dt / 0.3)
            self._robot_v = (
                self._robot_v[0] + a * (vx - self._robot_v[0]),
                self._robot_v[1] + a * (vy - self._robot_v[1]),
            )
        self._robot = (self.t, cx, cy)
        return cx, cy, robot.x, robot.y

    def _clear_of_robot(self, x: float, y: float) -> bool:
        if self._robot is None:
            return True
        return math.hypot(x - self._robot[1], y - self._robot[2]) >= CAR_R_M + SNOO_R_M

    def _react(self, dist: float, closing: float, robot_speed: float) -> None:
        """車に気づくか・気づいたらどうするかを決める。"""
        r = self.m["reaction"]
        flight, fast = _val(r["flight_m"]), _val(r["fast_m_s"])
        if self.mode == "flee":
            return
        # 飛び退く距離より内に速く近づかれたら、何をしていても逃げる
        if dist < flight and closing > fast:
            self._start("flee", dist, closing)
            return
        if self.mode != "calm" or self.cooldown > 0:
            return
        notice = _val(r["notice_m"])["sleep" if self.posture == "sleep" else "awake"]
        if dist > notice:
            return
        # 寝ている猫は、車が止まっていれば気づかない（音がしない）
        if self.posture == "sleep" and robot_speed < 0.02 and dist > flight:
            return
        kind = "fast" if closing > fast else ("slow" if robot_speed > 0.02 else "still")
        if kind == "still" and self._reacted_at is not None and self._robot is not None:
            # 止まっている車には 1 度だけ反応する。前に反応した所から車が動いていなければ、
            # もう知っている物として気にしない（何度も近づいて嗅ぎに行かない）
            if math.dist(self._reacted_at, self._robot[1:]) < 0.3:
                return
        probs = dict(_val(r["choice"])[kind])
        probs["flee"] = probs.get("flee", 0.0) * self.flee_scale
        self._start(_choice(self.rng, probs), dist, closing)

    def _start(self, mode: str, dist: float, closing: float) -> None:
        r = self.m["reaction"]
        if self._robot is not None:
            self._reacted_at = self._robot[1:]
        self.reactions.append(
            {
                "t": round(self.t, 2),
                "kind": mode,
                "dist_m": round(dist, 2),
                "closing_m_s": round(closing, 3),
                "posture": self.posture,
            }
        )
        if self.posture in ("sleep", "lie"):
            # 起きて体を起こす（次に落ち着いたときは座っているところから）
            self.posture = "sit"
            self.timer = self._duration("sit")
        if mode == "ignore":
            self.cooldown = _val(r["cooldown_s"])
            return
        self.mode = mode
        self._away_dir = None
        if mode == "freeze":
            self.mode_timer = _sample_lognorm(self.rng, r["freeze_s"], 0.5, 300)
        elif mode == "flee":
            self.mode_timer = 6.0
            self.speed = _val(self.m["walk"]["run_speed_m_s"]) * float(self.rng.uniform(0.8, 1.2))
        elif mode == "approach":
            self.mode_timer = 20.0
            self.speed = _sample_lognorm(self.rng, r["approach_speed_m_s"], 0.05, 0.5)
        elif mode == "retreat":
            self.mode_timer = 10.0
            self.speed = self._walk_speed()

    def _end_reaction(self, contact: bool = False) -> None:
        r = self.m["reaction"]
        if not contact:
            self.flee_scale = max(
                _val(r["habituation_floor"]), self.flee_scale * _val(r["habituation"])
            )
        self.mode = "calm"
        self.cooldown = _val(r["cooldown_s"])
        self.posture = "sit" if self.posture not in ("walk", "stand", "sit") else self.posture
        self.timer = self._duration(self.posture)
        self.speed = self._walk_speed()

    # --- 歩く ---

    def _walk(self, dt: float, heading: float, speed: float) -> bool:
        """heading へ向きを変えながら speed で進む。進めたら True。ふさがっていれば曲がる。"""
        turn = _wrap(heading - self.yaw)
        rate = 6.0 if self.mode == "flee" else 3.0
        self.yaw = _wrap(self.yaw + max(-rate * dt, min(rate * dt, turn)))
        ds = speed * dt * (0.3 if abs(turn) > 0.8 else 1.0)
        nx, ny = self.x + ds * math.cos(self.yaw), self.y + ds * math.sin(self.yaw)
        ahead = 0.15 + ds
        ax, ay = self.x + ahead * math.cos(self.yaw), self.y + ahead * math.sin(self.yaw)
        if self.free(nx, ny) and self.free(ax, ay) and self._clear_of_robot(nx, ny):
            self.x, self.y = nx, ny
            return True
        # 前がふさがっている：空いている向きのうち、いまの向きに近いほうへ（壁に沿う）
        self.heading = self._open_heading(heading)
        return False

    def _open_heading(self, want: float) -> float:
        best, best_cost = self.yaw + math.pi, math.inf
        side = 1.0 if self.rng.random() < 0.5 else -1.0
        for k in range(1, 13):
            for s in (side, -side):
                a = _wrap(self.yaw + s * k * math.pi / 12)
                px, py = self.x + 0.25 * math.cos(a), self.y + 0.25 * math.sin(a)
                if self.free(px, py) and self._clear_of_robot(px, py):
                    cost = abs(_wrap(a - want))
                    if cost < best_cost:
                        best, best_cost = a, cost
            if best_cost < math.inf and k >= 3:
                break
        return best

    # --- 1 刻み ---

    def step(self, dt: float, robot: Pose | None):
        n = max(1, int(math.ceil(dt / SUB_DT_S - 1e-9)))
        moved = False
        for _ in range(n):
            moved = self._substep(dt / n, robot) or moved
        self.pose = self._make_pose(moved)
        return self.pose

    def _substep(self, dt: float, robot: Pose | None) -> bool:
        self.t += dt
        self.cooldown = max(0.0, self.cooldown - dt)
        seen = self._see_robot(dt, robot)
        r = self.m["reaction"]
        dist = math.inf
        if seen is not None:
            cx, cy, camx, camy = seen
            dx, dy = self.x - camx, self.y - camy
            dist = math.hypot(dx, dy)
            vx, vy = self._robot_v
            closing = (vx * dx + vy * dy) / max(dist, 1e-6)
            self._react(dist, closing, math.hypot(vx, vy))
        if self.mode == "calm":
            return self._calm(dt)
        self.mode_timer -= dt
        away = math.atan2(self.y - seen[1], self.x - seen[0]) if seen else self.yaw
        if self.mode == "freeze":
            if seen:
                self.yaw = _wrap(
                    self.yaw + max(-3 * dt, min(3 * dt, _wrap(away + math.pi - self.yaw)))
                )
            if self.mode_timer <= 0:
                self._end_reaction()
            return False
        if self.mode == "sniff":
            if self.mode_timer <= 0:
                self._end_reaction()
            return False
        if self.mode == "approach":
            if seen is None or self.mode_timer <= 0:
                self._end_reaction()
                return False
            if dist <= _val(r["sniff_m"]) + CAR_R_M:
                self.mode = "sniff"
                self.mode_timer = _sample_lognorm(self.rng, r["sniff_s"], 0.5, 30)
                return False
            return self._walk(dt, away + math.pi, self.speed)
        # retreat・flee：車から離れる向きへ
        goal = _val(r["safe_m"] if self.mode == "flee" else r["retreat_m"])
        if seen is None or dist >= goal or self.mode_timer <= 0:
            self._end_reaction()
            return False
        if self._away_dir is None:
            # 逃げ始めの向きは車の反対から少しずらす（いつも同じ線の上を逃げないように）
            self._away_dir = away + float(self.rng.normal(0, 0.3))
        ok = self._walk(dt, self._away_dir, self.speed)
        if not ok:
            self._away_dir = self.heading
        return ok

    def _calm(self, dt: float) -> bool:
        self.timer -= dt
        if self.timer <= 0:
            self.posture = self._next_posture()
            self.timer = self._duration(self.posture)
            if self.posture == "walk":
                self.speed = self._walk_speed()
                self.heading = self.yaw + float(self.rng.normal(0, 1.2))
        if self.posture != "walk":
            return False
        sd = math.radians(_val(self.m["walk"]["turn_sd_deg_s"]))
        self.heading = _wrap(self.heading + float(self.rng.normal(0, sd * math.sqrt(dt))))
        return self._walk(dt, self.heading, self.speed)


class BehaviorAgent:
    """app/snoo_gridsim.GridSim の動く物（app/snoo_homes.Agent と同じ at(t)）で、車に反応する猫。

    GridSim は車が動くたびに follow(t, 車のカメラの姿勢) を呼ぶ。at(t) はそこまで進めた位置。
    前の時刻を聞かれたら、いまの位置を返す（GridSim の時計は戻らない）。
    """

    def __init__(self, snoo: BehaviorSnoo, radius: float, height: float, name: str = "snoo"):
        self.snoo, self.radius, self.height, self.name = snoo, radius, height, name
        self.t = 0.0
        self.robot: Pose | None = None
        self.traj: list[tuple[float, float, float, str]] = []  # (t, x, y, posture)

    def follow(self, t: float, robot: Pose | None) -> None:
        if t > self.t:
            self.snoo.step(t - self.t, self.robot)
            self.t = t
            p = self.snoo.pose
            self.traj.append((round(t, 2), p.x, p.y, p.posture))
        self.robot = robot

    def at(self, t: float) -> tuple[float, float]:
        self.follow(t, self.robot)
        return self.snoo.x, self.snoo.y


def grid_free(ok: np.ndarray, origin: float, res: float) -> Callable[[float, float], bool]:
    """格子（ok[ix, iy] が猫の中心が立てるセル）から free(x, y) を作る。"""

    def free(x: float, y: float) -> bool:
        ix, iy = int(math.floor((x - origin) / res)), int(math.floor((y - origin) / res))
        return 0 <= ix < ok.shape[0] and 0 <= iy < ok.shape[1] and bool(ok[ix, iy])

    return free
