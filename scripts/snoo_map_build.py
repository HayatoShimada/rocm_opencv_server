"""PiCar-X の撮影（ステーション）から間取りを作り、次の 1 歩を決める（app/snoo_mapper.py）。

  python -m scripts.snoo_map_build <ステーションの置き場...> --camera-height-cm 8 --out <出力先>
置き場は撮った順に並べる（最初のステーションが原点と x の向きになり、最後のステーション
から次の 1 歩を出す）。VGGT は GPU で 1 回だけ回し、出力を <出力先>/vggt.npz に残す。
--vggt でそれを（または前に作ったものを）渡せば GPU を使わずに作り直せる。
書き出すのは room.json・grid.png・high.png・topdown.png・plan.json。
カメラの高さはステーションごとに書いてもよい。0 のステーションは高さが分からないものとして、
ほかのステーションで決めた大きさで測る（机の上から撮った置き場を混ぜるときなど）。
"""

import argparse
import json
from pathlib import Path

from app.snoo_mapper import (
    MAX_FRAMES,
    MapParams,
    build_map,
    frame_colors,
    load_station,
    load_vggt,
    match_frames,
    run_vggt,
    save_vggt,
    select_frames,
    unproject,
    write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stations", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--camera-height-cm",
        type=float,
        nargs="+",
        default=[10.0],
        help="1 つか置き場ごと。0 は不明",
    )
    parser.add_argument("--car-height-cm", type=float, default=12.0)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--conf-percentile", type=float, default=50.0)
    parser.add_argument("--max-range-m", type=float, default=3.0)
    parser.add_argument("--vggt", type=Path, help="前に作った vggt.npz を使う（GPU を使わない）")
    args = parser.parse_args()

    stations = [load_station(d) for d in args.stations]
    heights = [h / 100 if h > 0 else None for h in args.camera_height_cm]
    if len(heights) == 1:
        heights = heights * len(stations)
    if args.vggt:
        v = load_vggt(args.vggt)
        refs = match_frames(v, stations)
    else:
        refs = select_frames(stations, args.max_frames)
        print(f"VGGT に {len(refs)} コマを入れる")
        v = run_vggt([r.path for r in refs])
        args.out.mkdir(parents=True, exist_ok=True)
        save_vggt(v, args.out / "vggt.npz")

    params = MapParams(
        camera_height_m=heights,
        car_height_m=args.car_height_cm / 100,
        conf_percentile=args.conf_percentile,
        max_range_m=args.max_range_m,
    )
    points, conf = unproject(v, params.stride, params.border_px)
    colors = frame_colors([r.path for r in refs], v.depth.shape[1:], params.stride)
    result = build_map(stations, refs, v.extrinsic, points, conf, params, colors)
    write_outputs(result, args.out)

    print(f"大きさ: {result.world.scale:.4f} m / VGGT の長さ")
    print(f"床の点: {result.floor.inliers}/{result.floor.candidates}")
    state = result.grid.state()
    print(f"セル: 床 {(state == 1).sum()} 障害物 {(state == 2).sum()} 上 {(result.high > 0).sum()}")
    for i, s in enumerate(result.stations):
        print(
            f"{i + 1} {s.name}: x {s.pose.x:+.3f} y {s.pose.y:+.3f} "
            f"yaw {s.pose.yaw * 57.2958:+.1f}° 高さ {s.height_m * 100:.1f} cm "
            f"ばらつき {s.spread_m * 100:.1f} cm コマ {s.frames} "
            f"点（床/障害物/上） {s.floor_points}/{s.obstacle_points}/{s.high_points}"
        )
    print(json.dumps(result.action, ensure_ascii=False))


if __name__ == "__main__":
    main()
