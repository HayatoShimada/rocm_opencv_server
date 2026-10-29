"""古いセッションの重いファイルを消し、置き場が際限なく大きくならないようにする。

  python -m scripts.snoo_prune [--keep 5] [--dry-run]
  make prune

新しい順に --keep 個のセッションは触らない。それより古いセッションからは、決め直し用の
recon.npz（1 か所 30 MB ほど）・vggt.npz・SLAM が写したフレーム（slam/frames）だけを消す。
ステーションの画像・meta.json・reply.json・obs.npz・間取り（room.json・grid.png）は残すので、
地図を見たり、run-report を出したりはできる（make replay は recon.npz が要るのでできなくなる）。
地図のサーバーは起動したときにも同じ後始末をする（app/snoo_map_server.py）。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from app.snoo_map_server import DATA, RECON_FILE

KEEP = 5
HEAVY_IN_SESSION = ("vggt.npz",)


def sessions(root: Path) -> list[Path]:
    """新しい順（いちばん最近に受けたステーションの meta.json の時刻で）。

    置き場の時刻やファイル全部の時刻では比べない。消すと置き場の時刻が新しくなり、
    次に片付けるとき、消したほうのセッションを新しいとみなしてしまう。
    """
    out = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]

    def received(p: Path) -> float:
        return max((m.stat().st_mtime for m in p.glob("[0-9][0-9][0-9]/meta.json")), default=0.0)

    return sorted(out, key=received, reverse=True)


def heavy_files(session: Path) -> list[Path]:
    files = [session / n for n in HEAVY_IN_SESSION if (session / n).exists()]
    files += sorted(session.glob(f"[0-9][0-9][0-9]/{RECON_FILE}"))
    frames = session / "slam" / "frames"
    if frames.is_dir():
        files.append(frames)
    return files


def prune(root: Path, keep: int = KEEP, dry_run: bool = False) -> tuple[int, int]:
    """(消したファイル・置き場の数, 減ったバイト数)。"""
    if not root.is_dir():
        return 0, 0
    n = freed = 0
    for session in sessions(root)[keep:]:
        for f in heavy_files(session):
            size = sum(p.stat().st_size for p in f.rglob("*")) if f.is_dir() else f.stat().st_size
            if not dry_run:
                shutil.rmtree(f) if f.is_dir() else f.unlink()
            n += 1
            freed += size
    return n, freed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", type=int, default=KEEP)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(DATA))
    args = parser.parse_args()
    n, freed = prune(args.root, args.keep, args.dry_run)
    verb = "消せる" if args.dry_run else "消した"
    print(f"{n} 個を{verb}（{freed / 1e6:.0f} MB）。新しい {args.keep} セッションは残した")


if __name__ == "__main__":
    main()
