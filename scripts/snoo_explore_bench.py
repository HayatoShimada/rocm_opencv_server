"""部屋を探る速さ（1 分あたり・走った 1 m あたりに分かった床の広さ）を、シミュレーターで測る。

    MUJOCO_GL=egl .venv/bin/python -m scripts.snoo_explore_bench \\
        --real data/snoo_map/sessions/explore_20261029_223944 --out out/bench/base.json
次の 1 歩の決め方を変えたら、前と後で同じ部屋・同じくじ（--seeds）で回して比べる。
1 回の実走は電池と時間と猫の機嫌を使い、部屋も毎回少し違うので、比べるのはここでする。

- 部屋：作り物の居間（synthetic）と、--real のセッションの間取り（room.json）から作った部屋
  （見た障害物はそのまま、見ていない所は床が続き、まわりを壁で囲む。room_from_map の unseen_floor）
- 時間：3 回目の実走と同じ（app/snoo_sim.REAL_PACE。10 コマ × 1.44 秒の首振り、
  地図のサーバー 11 秒、走る時間は速さから）。--server-s・--hold-s で変えられる
- 決め方：地図のサーバーと同じ（Sim.think）。Pi は segments・reverse を受け、超音波は格子に足す
- 止めるのは --minutes（実機の時計）か --cycles のどちらか早いほう
出すのは部屋とくじごとの数と、部屋ごとの平均。covered_m2 は本当の床のうち床と分かった広さ、
m2_at_5min は 5 分の時点の広さ、m2_per_min は covered_m2 を --minutes で割ったもの
（早く止まっても --minutes で割る。止まらずに探り続けるほうが良い）、m2_per_m は走った 1 m あたり。
--param name=value で MapParams を変えて試せる（例 --param max_step_m=0.6）。
--window N は地図のサーバーの窓（N か所だけで間取りを作る）をまねる。
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

from app.snoo_mapper import MapParams
from app.snoo_sim import CAM_H_M, REAL_PACE, Sim, room_from_map, synthetic_room

SEEDS = tuple(range(16))
FIELDS = (
    "stations", "t_s", "distance_m", "m2_at_5min", "covered_m2", "m2_per_min", "m2_per_m", "bumps",
)  # fmt: skip


def _value(text: str):
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def run_one(room, seed: int, args, params: MapParams) -> dict:
    pace = dataclasses.replace(
        REAL_PACE,
        server_s=args.server_s if args.server_s is not None else REAL_PACE.server_s,
        hold_s=args.hold_s if args.hold_s is not None else REAL_PACE.hold_s,
    )
    if args.pans:
        pace = dataclasses.replace(pace, pans=tuple(args.pans))
    if args.tilts:
        pace = dataclasses.replace(pace, tilts=tuple(args.tilts))
    sim = Sim(
        room, rays=args.rays, params=params, seed=seed, sonar_map=True, pace=pace,
        window=args.window,
    )  # fmt: skip
    t0 = time.perf_counter()
    s = sim.explore(args.cycles, backtrack=1, time_budget_s=args.minutes * 60)
    out = dataclasses.asdict(s) | {"room": room.name, "seed": seed}
    out["cpu_s"] = round(time.perf_counter() - t0, 1)
    out["m2_per_min"] = round(s.covered_m2 / max(s.t_s / 60, args.minutes), 3)
    out["stops"], out["reverses"], out["kinds"] = s.stops, s.reverses, sim.kinds
    # 5 分の時点（撮り終えた所で数える）。部屋が狭いと 10 分では全部分かってしまい、差が出ない
    out["m2_at_5min"] = max([c for t, c in sim.timeline if t <= 300] or [0.0])
    out["timeline"] = sim.timeline
    out["bump_log"] = sim.bump_log
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", type=Path, help="間取りから部屋を作るセッションの置き場")
    parser.add_argument("--no-synthetic", action="store_true", help="作り物の部屋は回さない")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--minutes", type=float, default=10.0, help="実機の時計で何分まで")
    parser.add_argument("--cycles", type=int, default=60)
    parser.add_argument("--server-s", type=float, help="地図のサーバーの返事までの秒")
    parser.add_argument("--hold-s", type=float, help="首振りの 1 コマの秒")
    parser.add_argument(
        "--pans", type=float, nargs="+", help="首振りのパン（既定 -30〜30 を 15° おき）"
    )
    parser.add_argument("--tilts", type=float, nargs="+", help="首振りのチルト（既定 -15 0）")
    parser.add_argument("--window", type=int, help="地図のサーバーの窓（ステーションの数）")
    parser.add_argument("--rays", type=lambda t: tuple(map(int, t.split("x"))), default=(160, 120))
    parser.add_argument("--param", action="append", default=[], help="MapParams の name=value")
    parser.add_argument("--out", type=Path, help="結果の JSON")
    args = parser.parse_args()

    extra = dict(p.split("=", 1) for p in args.param)
    params = MapParams(camera_height_m=CAM_H_M, segments=True, reverse=True)
    params = dataclasses.replace(params, **{k: _value(v) for k, v in extra.items()})
    rooms = [] if args.no_synthetic else [synthetic_room()]
    if args.real:
        rooms.append(room_from_map(args.real, unseen_floor=True))
    rows = []
    for room in rooms:
        for seed in args.seeds:
            row = run_one(room, seed, args, params)
            rows.append(row)
            print(
                json.dumps(
                    {k: row[k] for k in ("room", "seed", *FIELDS, "reason", "kinds", "bump_log")}
                ),
                flush=True,
            )
    means = {}
    for room in rooms:
        mine = [r for r in rows if r["room"] == room.name]
        means[room.name] = {k: round(sum(r[k] for r in mine) / len(mine), 3) for k in FIELDS}
        print(room.name, json.dumps(means[room.name]))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        info = {"args": {k: str(v) for k, v in vars(args).items()}, "means": means, "runs": rows}
        args.out.write_text(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
