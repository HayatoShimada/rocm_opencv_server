"""MuJoCo を使わない、速い 2 次元のシミュレーター（app/snoo_homes.py の家で、探り方を学ぶため）。

MuJoCo の Sim（app/snoo_sim.py）は 1 回の探索に数十秒かかり、何千もの家で何万回も回せない。
ここでは家の格子（5 cm、層と高さ）だけを使い、numpy で同じ周りを回す。

- 撮る：止まって首を振る（REAL_PACE と同じパン ±30° ＋ 画角の半分 27°）。カメラの真下から
  光線を格子の上に引き、見通せる床・障害物のセルを、地図のサーバーと同じ足し方で
  app/roommap.Grid に足す（床 FREE_HIT・障害物 OCCUPIED_HIT、1 回の撮影で 1 セル 1 回）。
  高さで見通しを決める：カメラは床から 8 cm なので、それより低い物（コード・座布団）は
  少し先の床だけを隠し、高い物（壁・家具）はその先を全部隠す。下をくぐれる物（HIGH：机の天板・
  椅子の座面）は床を隠さない（MuJoCo の Sim でも、低いカメラは机の下の床を見通す）
- 超音波：前の端から ±15° で、床から 4.5 cm より高い物までの距離（コードは見えない）。
  app/roommap.mark_sonar で弱い証拠として足す
- 走る：自転車モデル（app/roommap.arc）で 1 cm ずつ。外形（26 x 19 cm）が LOW・CLOSED の
  セルか人・snoo の円に重なればぶつかった（bump）としてその手前で止める。前へ走るときは、
  Pi が clear_cm を見て止まるのをまねて、まっすぐに近い（CLEAR_STEER_DEG 以下）区切りでは
  前の端から CLEAR_STOP_M（25 cm。区切りの min_clear_cm があればそれ）の通り道に物があれば
  止める（stop）。ハンドルを大きく切った区切り（Pi の clear_skip）は 5 cm 先がふさがれば止める。
  走った所は app/roommap.mark_driven で床にする
- 決め方：地図のサーバーと同じ（app/snoo_mapper.decide。Sim.think と同じ）。1 歩は前へ・後ろへの
  混じった区切りの列。chooser を渡せば差し替えられる
- 確信度と手ざわり（app/snoo_belief.py）の層も、地図のサーバーと同じに足す。家に手ざわり
  （app/snoo_homes.home_texture）を渡すと、撮った所から見えた目印（手ざわりの高い壁・家具）が
  LOST_MIN_LANDMARKS より少ないステーションは位置を見失ったとし、見たものを地図に足さない
  （地図のサーバーが bad_registration のステーションの見たものを足さないのと同じ）
- 時間：REAL_PACE（10 コマ × 1.44 秒、地図のサーバー 11 秒、speed 12 で 20 cm/s）
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from app.roommap import (
    CAM_X_M,
    CAR_CENTER_M,
    CAR_FRONT_M,
    CAR_REAR_M,
    FREE,
    FREE_HIT,
    OCCUPIED_HIT,
    SONAR_AHEAD_M,
    SONAR_HALF_DEG,
    SONAR_RANGE_M,
    Grid,
    Pose,
    arc,
    mark_blocked_ahead,
    mark_driven,
    mark_sonar,
    mark_station,
    rear_axle,
)
from app.snoo_belief import LANDMARK_TEX, Confidence, Texture
from app.snoo_homes import CLOSED, LOW, Home
from app.snoo_maneuver import CLEAR_STEER_DEG, CLEAR_STOP_M
from app.snoo_mapper import (
    RECENT_STATIONS,
    REVERSE_SPEED,
    SPEED,
    MapParams,
    StationResult,
    action_moves,
    decide,
)

# app/snoo_sim.py と同じ値（あちらは mujoco を読み込むので、ここでは読まない。試験で同じか確かめる）
CAM_H_M = 0.08
HFOV_DEG = 54.0
VFOV_DEG = 41.6  # 640 x 480 の縦の画角
SPEED_M_S = 0.017  # Pi の speed 1 あたり
STOP_CLEAR_M = 0.05
CORRIDOR_MARGIN_M = 0.02
SONAR_H_M = 0.045
CAR_HALF_W_M = 0.095
# 1 回の撮影で見えた目印（手ざわりの高い障害物のセル）がこれより少なければ、位置を見失う
LOST_MIN_LANDMARKS = 6


@dataclass(frozen=True)
class GridPace:
    """1 周の時間（app/snoo_sim.REAL_PACE と同じ）と撮り方。"""

    pans: tuple[float, ...] = (-30.0, -15.0, 0.0, 15.0, 30.0)
    tilts: tuple[float, ...] = (-15.0, 0.0)
    hold_s: float = 1.44
    server_s: float = 11.0

    @property
    def scan_s(self) -> float:
        return len(self.pans) * len(self.tilts) * self.hold_s

    @property
    def half_view_deg(self) -> float:
        return max(abs(p) for p in self.pans) + HFOV_DEG / 2

    @property
    def near_m(self) -> float:
        """床が写るいちばん近い距離（いちばん下向きのチルトの画面の下の端）。"""
        down = -min(self.tilts) + VFOV_DEG / 2
        return CAM_H_M / math.tan(math.radians(down))


REAL_GRID_PACE = GridPace()


@dataclass
class Episode:
    """1 回の探索の結果。timeline は撮り終えるごとの (秒, 調べた床 m², ぶつかった数)。"""

    home_seed: int
    t_s: float
    cycles: int
    covered_m2: float
    reachable_m2: float
    coverage: float
    bumps: int  # 家具・壁
    agent_bumps: int  # 人・snoo
    stops: int
    distance_m: float
    reason: str
    kinds: dict[str, int]
    timeline: list[tuple[float, float, int]] = field(default_factory=list)
    lost: int = 0  # 位置を見失った（目印が見えなかった）ステーションの数

    def m2_at(self, t_s: float) -> float:
        return max([c for t, c, _ in self.timeline if t <= t_s] or [0.0])


class GridSim:
    """家（Home）・車・作っている地図をまとめて持つ。"""

    def __init__(
        self,
        home: Home,
        params: MapParams | None = None,
        pace: GridPace = REAL_GRID_PACE,
        view_m: float = 2.0,
        ray_deg: float = 0.5,
        agents: bool = True,
        snoo: str = "random",
        texture: np.ndarray | None = None,
    ):
        self.home = home
        # 家のセルごとの手ざわり（app/snoo_homes.home_texture）。None なら見失わない
        self.texture = texture
        self.pace = pace
        self.view_m = view_m
        self.params = params or MapParams(
            camera_height_m=CAM_H_M, segments=True, reverse=True, gear_segments=True,
            max_stations=400,
        )  # fmt: skip
        self.agents = list(home.agents) if agents else []
        if agents and snoo == "behavior":
            # 前もって作った道のりの snoo を、車に反応する猫に替える（同じ所から始める）
            from app.snoo_homes import behavior_snoo

            self.agents = [
                behavior_snoo(home, a.at(0.0), home.seed + 7) if a.name == "snoo" else a
                for a in self.agents
            ]
        elif snoo not in ("random", "behavior"):
            raise ValueError(f"snoo の種類が分からない: {snoo}")
        self.grid: Grid = home.grid()
        self.grid.confidence = Confidence(self.grid.n)
        self.grid.texture = Texture(self.grid.n)
        self.pose = home.start  # カメラの真下
        # 動く物（BehaviorAgent）に見せる、いまの車のカメラの姿勢（走っている間も 1 cm ごとに）
        self.pose_now = home.start
        self.t = 0.0
        self.stations: list[StationResult] = []
        self.bumps = self.agent_bumps = self.stops = 0
        self.lost = 0
        self.last_solid = np.zeros((0, 2), int)
        self.distance_m = 0.0
        self.kinds: dict[str, int] = {}
        self.timeline: list[tuple[float, float, int]] = []
        self.action: dict = {}
        self.trail: list[Pose] = [self.pose]
        self.reach = home.reachable_floor()
        self.reach_m2 = float(self.reach.sum() * home.res**2)
        # 車がぶつかるセル（下をくぐれる物は通れる）
        self.car_block = (home.kind == LOW) | (home.kind == CLOSED)
        # 光線の向きと距離（撮るたびに同じ）
        half = math.radians(pace.half_view_deg)
        self._da = np.linspace(-half, half, int(round(2 * pace.half_view_deg / ray_deg)) + 1)
        self._d = np.arange(1, int(view_m / (home.res / 2)) + 1) * home.res / 2
        # 外形の中の点（後ろの車軸の座標、2.5 cm おき）。ぶつかるかはここで見る
        xs = np.arange(CAR_REAR_M, CAR_FRONT_M + 1e-9, home.res / 2)
        ys = np.arange(-CAR_HALF_W_M, CAR_HALF_W_M + 1e-9, home.res / 2)
        self._fx, self._fy = (a.ravel() for a in np.meshgrid(xs, ys))
        ys2 = np.arange(
            -CAR_HALF_W_M - CORRIDOR_MARGIN_M, CAR_HALF_W_M + CORRIDOR_MARGIN_M + 1e-9, home.res / 2
        )
        self._gx, self._gy = (a.ravel() for a in np.meshgrid(xs, ys2))

    # --- 家の様子 ---

    def agents_at(self, t: float) -> list[tuple[float, float, float, float]]:
        """時刻 t の人・snoo の (x, y, 半径, 高さ)。車に反応する物には、いまの車を見せてから。"""
        for a in self.agents:
            if hasattr(a, "follow"):
                a.follow(t, self.pose_now)
        return [(*a.at(t), a.radius, a.height) for a in self.agents]

    def _layers(self, t: float, ix: np.ndarray, iy: np.ndarray):
        """セル（範囲の外も可）の種類と高さ。人・snoo は LOW として重ねる。"""
        h = self.home
        inside = (ix >= 0) & (ix < h.n) & (iy >= 0) & (iy < h.n)
        cx, cy = np.clip(ix, 0, h.n - 1), np.clip(iy, 0, h.n - 1)
        kind = np.where(inside, h.kind[cx, cy], CLOSED)
        height = np.where(inside, h.height[cx, cy], 0.0)
        if self.agents:
            px = h.origin + (ix + 0.5) * h.res
            py = h.origin + (iy + 0.5) * h.res
            for ax, ay, r, ah in self.agents_at(t):
                hit = (px - ax) ** 2 + (py - ay) ** 2 <= r * r
                if hit.any():
                    kind = np.where(hit, LOW, kind)
                    height = np.where(hit, np.maximum(height, ah), height)
        return kind, height

    # --- 撮る ---

    def observe(self, pose: Pose, t: float) -> tuple[np.ndarray, np.ndarray]:
        """pose で首を振って撮ったとき、床と見えるセル・障害物と見えるセル（(K, 2) ずつ）。"""
        h = self.home
        a = pose.yaw + self._da[:, None]
        d = self._d[None, :]
        ix, iy = h.cell(pose.x + d * np.cos(a), pose.y + d * np.sin(a))
        kind, height = self._layers(t, ix, iy)
        low = kind == LOW
        full = (kind == CLOSED) | (low & (height >= CAM_H_M))
        # 光線ごとに、最初の「その先を全部隠す」物まで（その物は見える）
        before = np.cumsum(full, axis=1) - full  # 手前に full がいくつあるか
        open_ = before == 0
        # 低い物の影：高さ hh・距離 x の物は、床を x / (1 - hh / CAM_H) まで隠す
        dd = np.broadcast_to(d, kind.shape)
        shadow = np.where(low & ~full, dd / np.maximum(1 - height / CAM_H_M, 1e-3), 0.0)
        shadow = np.maximum.accumulate(shadow, axis=1)
        prev = np.concatenate([np.zeros((len(a), 1)), shadow[:, :-1]], axis=1)
        hmax = np.maximum.accumulate(np.where(low, height, 0.0), axis=1)
        hprev = np.concatenate([np.zeros((len(a), 1)), hmax[:, :-1]], axis=1)
        floorish = (kind != LOW) & (kind != CLOSED)
        floor_vis = floorish & open_ & (dd > prev) & (dd >= self.pace.near_m)
        obst_vis = low & open_ & ((dd > prev) | (height > hprev))
        # 見えた固い物（壁・家具。光線ごとの最初の壁も）：目印が見えたかを数える（見失うか）
        solid_vis = (full & open_) | obst_vis
        out = []
        for m in (floor_vis, obst_vis, solid_vis):
            cells = np.stack([ix[m], iy[m]], 1)
            out.append(np.unique(cells, axis=0) if len(cells) else cells.reshape(0, 2))
        self.last_solid = out[2]
        return out[0], out[1]

    def sonar(self, pose: Pose, t: float) -> float | None:
        """前の超音波の読み（m）。SONAR_H_M より高い物（と家の外）までのいちばん近い距離。"""
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        ox, oy = pose.x + SONAR_AHEAD_M * c, pose.y + SONAR_AHEAD_M * s
        half = math.radians(SONAR_HALF_DEG)
        a = pose.yaw + np.linspace(-half, half, 13)[:, None]
        d = np.arange(1, int(SONAR_RANGE_M / 0.025) + 1)[None, :] * 0.025
        ix, iy = self.home.cell(ox + d * np.cos(a), oy + d * np.sin(a))
        kind, height = self._layers(t, ix, iy)
        hit = (kind == CLOSED) | ((kind == LOW) & (height >= SONAR_H_M))
        if not hit.any():
            return None
        first = np.where(hit.any(axis=1), np.argmax(hit, axis=1), hit.shape[1])
        best = float(d[0, first[first < hit.shape[1]]].min())
        return best if best <= SONAR_RANGE_M else None

    def scan(self) -> StationResult:
        """止まって首を振って撮り、地図に足す（地図のサーバーと同じ足し方）。"""
        t_mid = self.t + self.pace.scan_s / 2
        free, occ = self.observe(self.pose, t_mid)
        g = self.grid
        lost = False
        solid = self.last_solid
        solid = solid[(solid >= 0).all(1) & (solid < self.home.n).all(1)]
        if self.texture is not None and self.stations:  # 最初の所は原点なので見失わない
            seen = self.texture[solid[:, 0], solid[:, 1]] if len(solid) else np.zeros(0)
            lost = int((seen >= LANDMARK_TEX).sum()) < LOST_MIN_LANDMARKS
        if lost:
            self.lost += 1  # 位置が分からないので、見たものは足さない
        else:
            mark_station(g, self.pose)
            for cells, v in ((free, FREE_HIT), (occ, OCCUPIED_HIT)):
                if len(cells):
                    g.logodds[cells[:, 0], cells[:, 1]] += v
                    np.clip(g.logodds, -4, 4, out=g.logodds)
            g.confidence.add_cells(g, free, occ, self.pose)
            if self.texture is not None and len(solid):
                # 手ざわりの層は、見えた壁・家具の手ざわり（地図の点の色の勾配の代わり）
                x, y = g.center(solid[:, 0], solid[:, 1])
                g.texture.add(g, np.stack([x, y, self.texture[solid[:, 0], solid[:, 1]]], 1))
            # 超音波は弱い証拠として足す（地図のサーバーと同じ）
            sonar = self.sonar(self.pose, t_mid)
            mark_sonar(g, self.pose, sonar)
            g.confidence.add_sonar(g, self.pose, sonar)
        self.t += self.pace.scan_s
        frames = len(self.pace.pans) * len(self.pace.tilts)
        st = StationResult(f"grid{len(self.stations):03d}", self.pose, CAM_H_M, 0.0, frames)
        self.stations.append(st)
        self.timeline.append((round(self.t, 1), self.covered_m2(), self.bumps + self.agent_bumps))
        return st

    def covered_m2(self) -> float:
        """行ける床のうち、床と分かった広さ。"""
        return float(((self.grid.state() == FREE) & self.reach).sum() * self.home.res**2)

    # --- 走る ---

    def _hits(self, x: float, y: float, yaw: float, t: float, wide: bool = False) -> str | None:
        """後ろの車軸が (x, y, yaw) の車の外形が重なる物。'wall'・'agent'・None。"""
        fx, fy = (self._gx, self._gy) if wide else (self._fx, self._fy)
        c, s = math.cos(yaw), math.sin(yaw)
        px, py = x + c * fx - s * fy, y + s * fx + c * fy
        ix, iy = self.home.cell(px, py)
        h = self.home
        inside = (ix >= 0) & (ix < h.n) & (iy >= 0) & (iy < h.n)
        if not inside.all() or self.car_block[ix, iy].any():
            return "wall"
        for ax, ay, r, _ in self.agents_at(t):
            # 外形の真ん中から近ければ、点で確かめる
            mx, my = x + CAR_CENTER_M * c, y + CAR_CENTER_M * s
            if math.hypot(mx - ax, my - ay) > r + 0.2:
                continue
            if (((px - ax) ** 2 + (py - ay) ** 2) <= r * r).any():
                return "agent"
        return None

    def _clear_blocked(
        self, x: float, y: float, yaw: float, steer: float, t: float, clear_m: float
    ) -> bool:
        """後ろの車軸が (x, y, yaw) の車の前の端から、同じハンドルで clear_m 先までの通り道
        （左右の端と真ん中）に物（LOW・CLOSED・人・snoo）があるか。Pi の clear_cm の代わり。"""
        h = self.home
        lat = np.array([-CAR_HALF_W_M, 0.0, CAR_HALF_W_M])
        agents = self.agents_at(t)
        for s in np.arange(0.0, clear_m + 1e-9, h.res / 2):
            ax, ay, ayaw = arc(x, y, yaw, steer, float(s))
            c, sn = math.cos(ayaw), math.sin(ayaw)
            px = ax + CAR_FRONT_M * c - lat * sn
            py = ay + CAR_FRONT_M * sn + lat * c
            ix, iy = h.cell(px, py)
            inside = (ix >= 0) & (ix < h.n) & (iy >= 0) & (iy < h.n)
            if not inside.all() or self.car_block[ix, iy].any():
                return True
            for bx, by, r, _ in agents:
                if (((px - bx) ** 2 + (py - by) ** 2) <= r * r).any():
                    return True
        return False

    def drive(
        self,
        steer_deg: float,
        speed: float,
        distance_m: float,
        reverse: bool,
        min_clear_m: float | None = None,
    ) -> tuple[str, float]:
        """Pi と同じ指令で走る。('ok'・'stop'・'bump', 走った m)。
        前へ走るとき、まっすぐに近ければ前の端から max(CLEAR_STOP_M, min_clear_m) の通り道に物が
        入れば止める（Pi の clear_cm）。ハンドルを大きく切っていれば 5 cm 先で止める。"""
        r = rear_axle(self.pose)
        x, y, yaw = r.x, r.y, r.yaw
        steer = max(-30.0, min(30.0, steer_deg))
        v = max(speed, 1.0) * SPEED_M_S
        sign = -1.0 if reverse else 1.0
        done, result = 0.0, "ok"
        clear_m = None
        if not reverse and abs(steer) <= CLEAR_STEER_DEG:
            clear_m = max(CLEAR_STOP_M, min_clear_m or 0.0)
        while done < distance_m - 1e-9:
            ds = min(0.01, distance_m - done)
            self.pose_now = Pose(x + CAM_X_M * math.cos(yaw), y + CAM_X_M * math.sin(yaw), yaw)
            nx, ny, nyaw = arc(x, y, yaw, steer, sign * ds)
            hit = self._hits(nx, ny, nyaw, self.t)
            if hit:
                result = "bump"
                if hit == "agent":
                    self.agent_bumps += 1
                else:
                    self.bumps += 1
                break
            x, y, yaw = nx, ny, nyaw
            done += ds
            self.distance_m += ds
            self.t += ds / v
            if not reverse and done < distance_m - 1e-9:
                ax, ay, ayaw = arc(x, y, yaw, steer, STOP_CLEAR_M)
                stop = self._hits(ax, ay, ayaw, self.t, wide=True)
                if not stop and clear_m is not None:
                    stop = self._clear_blocked(x, y, yaw, steer, self.t, clear_m)
                if stop:
                    result = "stop"
                    self.stops += 1
                    break
        c, s = math.cos(yaw), math.sin(yaw)
        cyaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        self.pose = self.pose_now = Pose(x + CAM_X_M * c, y + CAM_X_M * s, cyaw)
        self.trail.append(self.pose)
        return result, done

    # --- 決める ---

    def think(self) -> dict:
        """地図のサーバーと同じ決め方（app/snoo_mapper.decide。Sim.think と同じ）。"""
        heights = [CAM_H_M] * len(self.stations)
        params = dataclasses.replace(self.params, seed=len(self.stations) - 1)
        recent = [s.pose for s in self.stations[-RECENT_STATIONS:]]
        action, _ = decide(self.grid, self.stations, params, heights, recent=recent)
        return action

    def run(
        self,
        time_budget_s: float = 1200.0,
        max_cycles: int = 400,
        chooser: Callable[[GridSim], dict] | None = None,
        on_cycle: Callable[[GridSim], None] | None = None,
    ) -> Episode:
        """撮る → 決める → 走る を、time_budget_s 秒（実機の時計）か、やめる理由が出るまで。"""
        reason = "cycles"
        for _ in range(max_cycles):
            if self.t >= time_budget_s:
                reason = "time"
                break
            self.scan()
            action = (chooser or GridSim.think)(self)
            self.action = action
            kind = _kind(action)
            self.kinds[kind] = self.kinds.get(kind, 0) + 1
            self.t += self.pace.server_s
            if on_cycle:
                on_cycle(self)
            if action.get("action") != "move":
                reason = str(action.get("reason", "done"))
                break
            self.execute(action)
        return self.episode(reason)

    def execute(self, action: dict) -> None:
        """返事の区切りを順に走る。前へ止まった・ぶつかったら、そこで 1 歩を終える。"""
        fwd = float(action.get("speed", SPEED))
        back_speed = float(action.get("reverse_speed", REVERSE_SPEED))
        segs = action.get("segments") or [action]
        for i, (steer, m, back) in enumerate(action_moves(action)):
            start = self.pose
            clear = segs[i].get("min_clear_cm") if i < len(segs) else None
            clear_m = float(clear) / 100 if clear is not None else None
            result, moved = self.drive(steer, back_speed if back else fwd, m, back, clear_m)
            if moved > 1e-6:
                mark_driven(self.grid, start, [(steer, moved)], back)
            if result != "ok":
                # 止まった・ぶつかった所の先は、たぶん障害物（地図のサーバーも prev_step から足す）
                mark_blocked_ahead(self.grid, self.pose, steer, back)
                break

    def episode(self, reason: str) -> Episode:
        cov = self.covered_m2()
        return Episode(
            self.home.seed, round(self.t, 1), len(self.stations), round(cov, 3),
            round(self.reach_m2, 3), round(cov / max(self.reach_m2, 1e-9), 4), self.bumps,
            self.agent_bumps, self.stops, round(self.distance_m, 2), reason, dict(self.kinds),
            list(self.timeline), self.lost,
        )  # fmt: skip


def _kind(action: dict) -> str:
    if action.get("action") != "move":
        return "done:" + str(action.get("reason", ""))
    moves = action_moves(action)
    backs = {b for _, _, b in moves}
    if backs == {True}:
        return "reverse"
    if backs == {True, False}:
        return "gear"
    return "forward"
