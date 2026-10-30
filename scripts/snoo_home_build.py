"""残っている探索のセッションから、家の地図（app/snoo_home.py）をまとめて作る。GPU は使わない。

    python -m scripts.snoo_home_build --out data/snoo_map/home [セッション名...]
    make home-build OUT=data/snoo_map/home ARGS="explore_20261030_175740 ..."

セッションを名前（= 時刻）の順に家に合わせ、合ったものを足す。最初のセッション（--anchor で
選べる）の座標が家の座標になる。合わなかったものは島にし（今ある島に合えばその島に足す）、
後のセッションを家に足すたびに、島ごと家に合わせ直す。セッションごとの変換・
点数・重なり・断った理由を出し、<out>/home.png・home.json・build.json を書く。
--out は空の置き場か、前にこのスクリプトで作った置き場にする（中身は作り直す）。
走っている探索のセッションは入れない（ステーションが増えている途中なので）。
--upto N を付けると、各セッションの N 番目のステーションまでの間取りで合わせてみる（地図の
サーバーが 1 か所目・2 か所目で合わせるときに、どれだけ合うかを見る）。
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from app.snoo_home import HomeMap, load_session_grid, register_session

SESSIONS = Path("data/snoo_map/sessions")


def build(root: Path, names: list[str], out: Path, upto: int | None, visual: bool) -> list[dict]:
    if out.exists():
        if any(out.iterdir()) and not (out / "build.json").is_file():
            raise SystemExit(f"{out} は空でなく、このスクリプトで作った置き場でもない")
        shutil.rmtree(out)
    home = HomeMap(out)
    rows = []
    for name in names:
        d = root / name
        try:
            grid, high, source = load_session_grid(d, upto)
        except (OSError, ValueError) as e:
            rows.append({"session": name, "status": "skipped", "reason": str(e)})
            print(f"{name}: 読めない（{e}）")
            continue
        match = register_session(home, grid, d, visual=visual and not home.empty())
        row = {"session": name, "source": source, **match.to_dict()}
        if match.ok:
            home.merge_closed(name, grid, high, match.pose, match, d)
            row["status"] = "anchor" if match.reason == "anchor" else "registered"
        else:
            iid = home.add_island(name, grid, high, match, d, join=True, visual=visual)
            row["status"] = "island"
            row["island"] = iid
        joined = home.connect_islands(skip=name, visual=visual) if match.ok else []
        row["joined_islands"] = joined
        rows.append(row)
        p = match.pose
        print(
            f"{name}: {row['status']:<10} ({source}) x={p.x:+.2f} y={p.y:+.2f} "
            f"yaw={p.yaw * 57.2958:+.1f}° score={match.score:.2f} second={match.second:.2f} "
            f"margin={match.margin:.2f} overlap={match.overlap:.2f} {match.reason} "
            f"{match.seconds:.2f}s"
            + (f" visual={match.visual}" if match.visual else "")
            + (f" 島={row['island']}" if row.get("island") else "")
            + (f" → 島をつないだ: {', '.join(joined)}" if joined else "")
        )
    # 後でつながった島は、つないだときの答えで書き直す
    for row in rows:
        s = home.sessions.get(row["session"])
        if row.get("status") == "island" and s is not None:
            row["status"] = "registered_later"
            row["pose"] = s["pose"]
            row["via"] = s.get("match", {})
    home.save()
    (out / "build.json").write_text(
        json.dumps(
            {"built": time.strftime("%Y-%m-%dT%H:%M:%S"), "upto": upto, "sessions": rows},
            ensure_ascii=False,
            indent=1,
        )
    )
    info = home.info()
    print(
        f"家の床 {info['free_m2']} m²、足したセッション {len(info['sessions'])}、"
        f"島 {len(info['islands'])}。{out / 'home.png'}"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessions", nargs="*", help="セッションの名前（無ければ explore_* 全部）")
    parser.add_argument("--root", type=Path, default=SESSIONS)
    parser.add_argument("--out", type=Path, required=True, help="家の地図の置き場")
    parser.add_argument("--upto", type=int, help="このステーションまでの間取りで合わせる")
    parser.add_argument("--exclude", nargs="*", default=[], help="入れないセッション")
    parser.add_argument("--anchor", help="家の座標にするセッション（先に足す。ほかは時刻の順）")
    parser.add_argument("--no-visual", action="store_true", help="画像（ORB）で確かめない")
    args = parser.parse_args()
    names = args.sessions or sorted(p.name for p in args.root.glob("explore_*") if p.is_dir())
    names = sorted(n for n in names if n not in args.exclude)
    if args.anchor:
        names = [args.anchor] + [n for n in names if n != args.anchor]
    build(args.root, names, args.out, args.upto, not args.no_visual)


if __name__ == "__main__":
    main()
