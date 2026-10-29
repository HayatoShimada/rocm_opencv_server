"""PiCar-X の間取り作りを MuJoCo の部屋で回し、絵と動画にする（app/snoo_sim.py）。

  MUJOCO_GL=egl python -m scripts.snoo_sim_view --explore 20 --out out/sim
  MUJOCO_GL=egl python -m scripts.snoo_sim_view --room data/snoo_map/sessions/<id> --out out/sim
--room が無ければ作り物の部屋（4 x 3 m の居間）を使う。--room には地図のサーバーのセッションの
置き場か room.json を渡す（その間取りを部屋にする）。
--explore N で「撮る → 地図を直す → 1 歩走る」を N 回まわし、explore.mp4・last.png・
summary.json・grid.png を書く。--explore が無ければ、--room の間取りをそのまま地図として
1 枚だけ描く（frame.png）。
--viewer は MuJoCo の窓を開いて回す（画面のあるときだけ）。
"""

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import cv2

from app.snoo_mapper import MapParams, grid_png
from app.snoo_sim import CAM_H_M, Sim, room_from_map, synthetic_room


def _rays(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def run_viewer(sim: Sim, cycles: int, backtrack: int = 0) -> None:
    """MuJoCo の窓で回す。地図などは user_scn に足して描く。"""
    import mujoco.viewer

    from app.snoo_sim_render import add_scene_overlays

    with mujoco.viewer.launch_passive(sim.model, sim.data) as v:

        def show(s: Sim) -> None:
            with v.lock():
                v.user_scn.ngeom = 0
                add_scene_overlays(v.user_scn, s)
            v.sync()
            time.sleep(0.03)

        sim.explore(cycles, show, backtrack=backtrack)
        while v.is_running():
            show(sim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--room", type=Path, help="セッションの置き場か room.json")
    parser.add_argument("--explore", type=int, default=0, help="まわす回数")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rays", type=_rays, default=(320, 240), help="1 コマの光線（横x縦）")
    parser.add_argument("--high-board-m", type=float, default=0.40, help="間取りの上の層の高さ")
    parser.add_argument(
        "--backtrack",
        type=int,
        default=0,
        help="行き場が無いとき来た道を戻って撮り直す回数（Pi の reverse。サーバーは 2）",
    )
    parser.add_argument("--sonar", action="store_true", help="前の超音波を格子に足す")
    parser.add_argument("--segments", action="store_true", help="1 歩を曲がる → まっすぐで走る")
    parser.add_argument("--viewer", action="store_true", help="MuJoCo の窓で見る")
    args = parser.parse_args()

    room = room_from_map(args.room, args.high_board_m) if args.room else synthetic_room()
    params = MapParams(camera_height_m=CAM_H_M, segments=args.segments)
    sim = Sim(room, rays=args.rays, params=params, seed=args.seed, sonar_map=args.sonar)
    if args.viewer:
        run_viewer(sim, args.explore or 20, args.backtrack)
        return

    from app.snoo_sim_render import SimRenderer, VideoWriter

    args.out.mkdir(parents=True, exist_ok=True)
    renderer = SimRenderer(sim)
    try:
        if not args.explore:
            if args.room:
                sim.load_knowledge(args.room)
            img = renderer.frame(f"snoo_sim  map {room.name}")
            cv2.imwrite(str(args.out / "frame.png"), img)
            print(f"{args.out / 'frame.png'}  next: {json.dumps(sim.action, ensure_ascii=False)}")
            return
        video = VideoWriter(args.out / "explore.mp4", args.fps)
        t0 = time.time()
        last = None

        def on_frame(s: Sim) -> None:
            nonlocal last
            last = renderer.frame()
            video.write(last)

        on_frame(sim)
        summary = sim.explore(args.explore, on_frame, backtrack=args.backtrack)
        last = renderer.frame()
        for _ in range(args.fps * 2):  # 最後の絵を 2 秒見せる
            video.write(last)
        video.close()
        cv2.imwrite(str(args.out / "last.png"), last)
        cv2.imwrite(str(args.out / "grid.png"), grid_png(sim.grid.state()))
        out = dataclasses.asdict(summary) | {
            "room": room.name,
            "frames": video.frames,
            "seconds": round(time.time() - t0, 1),
            "station_poses": [
                {
                    "x": round(s.pose.x, 3),
                    "y": round(s.pose.y, 3),
                    "yaw_deg": round(math.degrees(s.pose.yaw), 1),
                }
                for s in sim.stations
            ],
            "last_action": sim.action,
        }
        (args.out / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        brief = {k: v for k, v in out.items() if k != "station_poses"}
        print(json.dumps(brief, ensure_ascii=False))
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
