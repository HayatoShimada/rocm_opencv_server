"""探索を 1 回走らせた後に、うまく行ったかを 1 枚にまとめる（GPU を使わない）。

  python -m scripts.snoo_run_report <セッション> [--clear <clear.jsonl>] [--base <セッション>]
  make run-report SESSION=explore_... [CLEAR=path/to/clear.jsonl]

出すもの:
- ステーションの数・返事の内訳（move / random / reverse・turn / done の理由）・止まった理由
  （prev_step.stop_reason）・低い物を越えた回数・電池
- 地図の広さ（床・障害物・上の層、m²）と、走った長さ・ステーションが散らばった広さ
- サーバーの時間（plan.json の timing_s は最後のステーションだけなので、ログが無ければ最後だけ）
- 止まって撮ったコマのぶれ（ラプラシアンの分散。大きいほどくっきり）。--base のセッション
  （たとえば 3 回目の実走）と比べ、中央値が 0.7 倍を切れば「撮る前の待ちが短すぎる」とする（#22）
- 止まって撮ったコマの向きのずれ：recon.npz の VGGT のカメラの向きと、指令のパンの差（#22）
- --clear に Pi の clear.jsonl を渡すと、clear_cm と超音波の差、settling・out_of_view の割合（#21）
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.snoo_map_server import DATA, RECON_FILE, load_recon, station_dir, station_ids

CELL_M2 = 0.05 * 0.05
# ぶれの判定：ベースの中央値に対してこれより下なら、撮る前の待ちが短すぎる
BLUR_RATIO_MIN = 0.7
# 向きのずれ：止まって撮ったコマの VGGT の向きと、指令のパンの差がこれを超えるコマを数える
PAN_ERR_DEG = 4.0


def resolve(name: str) -> Path:
    p = Path(name)
    if p.is_dir():
        return p
    p = Path(DATA) / name
    if p.is_dir():
        return p
    raise SystemExit(f"{name} というセッションは無い")


def metas(session: Path) -> list[tuple[int, dict]]:
    return [
        (i, json.loads((station_dir(session, i) / "meta.json").read_text()))
        for i in station_ids(session)
    ]


def sharpness(path: Path) -> float | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def still_sharpness(session: Path) -> list[float]:
    out = []
    for i, m in metas(session):
        for f in m.get("frames", []):
            if not f.get("moving"):
                s = sharpness(station_dir(session, i) / f["file"])
                if s is not None:
                    out.append(s)
    return out


def rotation_deg(r: np.ndarray) -> float:
    """回転行列の回転の大きさ（度）。"""
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2))))


def pan_errors(session: Path) -> list[float]:
    """ステーションごとに、止まって撮ったコマの VGGT の向きと指令のパンの差（度）。

    ステーションの中で向きの差だけを見る（車は動いていないので、差はパンだけのはず）。
    チルトの段ごとに、パン 0 に近いコマを基準にし、2 コマの間の回転の大きさとパンの差を比べる。
    """
    errs = []
    for i, m in metas(session):
        path = station_dir(session, i) / RECON_FILE
        if not path.exists():
            continue
        keys, recon = load_recon(path)
        cmd = {
            f["file"]: (f.get("pan_deg"), round(f.get("tilt_deg") or 0.0))
            for f in m.get("frames", [])
            if not f.get("moving") and f.get("pan_deg") is not None
        }
        own = [(cmd[k.split("/")[1]], j) for j, k in enumerate(keys)
               if k.split("/")[0] == f"{i:03d}" and k.split("/")[1] in cmd]  # fmt: skip
        rot = recon.extrinsic[:, :, :3]
        # 同じチルトの段の中で、2 コマの間の回転の大きさ（どの軸でも）を、パンの差と比べる。
        # 傾いたカメラの座標で水平の向きを測ると、チルトの分だけずれる
        for tilt in {t for (_, t), _ in own}:
            row = [(pan, j) for (pan, t), j in own if t == tilt]
            if len(row) < 2:
                continue
            ref_pan, ref_j = min(row, key=lambda r: abs(r[0]))
            for pan, j in row:
                if j != ref_j:
                    errs.append(abs(rotation_deg(rot[j] @ rot[ref_j].T) - abs(pan - ref_pan)))
    return errs


def grid_area(session: Path) -> dict:
    g = cv2.imread(str(session / "grid.png"), cv2.IMREAD_GRAYSCALE)
    h = cv2.imread(str(session / "high.png"), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return {}
    out = {
        "床 m²": round((g == 255).sum() * CELL_M2, 2),
        "障害物 m²": round((g == 0).sum() * CELL_M2, 2),
    }
    if h is not None:
        out["上の層 m²"] = round((h > 0).sum() * CELL_M2, 2)
    return out


def poses(session: Path) -> list[tuple[float, float]]:
    p = session / "poses.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    return [(v["x"], v["y"]) for _, v in sorted(d.items(), key=lambda kv: int(kv[0]))]


def replies(session: Path) -> list[dict]:
    out = []
    for i in station_ids(session):
        p = station_dir(session, i) / "reply.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


def reply_kind(r: dict) -> str:
    if r.get("action") != "move":
        return f"done:{r.get('reason')}"
    if r.get("reverse"):
        return "turn" if r.get("turn") else "reverse"
    return "random" if r.get("random") else "move"


def clear_stats(path: Path) -> dict:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    skip = Counter(r.get("clear_skip") for r in rows if r.get("clear_skip"))
    both = [r for r in rows if r.get("clear_cm") is not None and r.get("ultrasonic_cm") is not None]
    diff = [r["ultrasonic_cm"] - r["clear_cm"] for r in both]
    nulls = sum(1 for r in rows if r.get("clear_cm") is None and not r.get("clear_skip"))
    out = {
        "行": len(rows),
        "値あり": len(both),
        "null": nulls,
        **{f"skip:{k}": v for k, v in skip.items()},
    }
    if diff:
        out["超音波 − clear_cm の中央値 cm"] = round(statistics.median(diff), 1)
        out["clear_cm が 25 cm 以上短い（低い物か見誤り）"] = sum(d >= 25 for d in diff)
        out["clear_cm が 10 cm 以上長い（見落としの疑い）"] = sum(d <= -10 for d in diff)
    return out


def report(session: Path, clear: Path | None = None, base: Path | None = None) -> list[str]:
    ms = metas(session)
    lines = [f"# {session.name}（{len(ms)} か所）"]
    stops = Counter(
        (m.get("prev_step") or {}).get("stop_reason") for _, m in ms if m.get("prev_step")
    )
    passed = sum((m.get("prev_step") or {}).get("low_obstacle_passed", 0) or 0 for _, m in ms)
    bats = [m.get("battery_v") for _, m in ms if (m.get("battery_v") or 0) > 0]
    lines.append("止まった理由: " + ", ".join(f"{k} {v}" for k, v in stops.most_common()))
    lines.append(f"低い物を越えた: {passed} 回")
    if bats:
        lines.append(
            f"電池: {bats[0]:.2f} → {bats[-1]:.2f} V（0 V と読めたのは "
            f"{sum(1 for _, m in ms if m.get('battery_v') == 0)} 回）"
        )
    kinds = Counter(reply_kind(r) for r in replies(session))
    if kinds:
        lines.append("返事: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    lines.append("地図: " + ", ".join(f"{k} {v}" for k, v in grid_area(session).items()))
    ps = poses(session)
    if len(ps) >= 2:
        driven = sum(math.dist(a, b) for a, b in zip(ps, ps[1:], strict=False))
        xs, ys = zip(*ps, strict=True)
        lines.append(
            f"ステーションの間の長さ {driven:.2f} m、散らばり "
            f"{max(xs) - min(xs):.2f} × {max(ys) - min(ys):.2f} m"
        )
    sharp = still_sharpness(session)
    if sharp:
        med = statistics.median(sharp)
        line = f"止まったコマのくっきりさ（中央値）: {med:.0f}"
        if base is not None:
            base_sharp = still_sharpness(base)
            if base_sharp:
                ratio = med / statistics.median(base_sharp)
                verdict = "撮る前の待ちが短すぎる" if ratio < BLUR_RATIO_MIN else "問題なし"
                line += f"（{base.name} の {ratio:.2f} 倍 → {verdict}）"
        lines.append(line)
    errs = pan_errors(session)
    if errs:
        lines.append(
            f"止まったコマの向きのずれ: 中央値 {statistics.median(errs):.1f}°・"
            f"最大 {max(errs):.1f}°、"
            f"{PAN_ERR_DEG:.0f}° を超えたのは {sum(e > PAN_ERR_DEG for e in errs)}/{len(errs)} コマ"
        )
    else:
        lines.append("止まったコマの向きのずれ: recon.npz が無い（2026-10-30 より前の版で受けた）")
    if clear is not None:
        lines.append("clear_cm: " + ", ".join(f"{k} {v}" for k, v in clear_stats(clear).items()))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session")
    parser.add_argument("--clear", type=Path, help="Pi の clear.jsonl")
    parser.add_argument("--base", help="くっきりさを比べるセッション（たとえば 3 回目の実走）")
    args = parser.parse_args()
    base = resolve(args.base) if args.base else None
    print("\n".join(report(resolve(args.session), args.clear, base)))


if __name__ == "__main__":
    main()
