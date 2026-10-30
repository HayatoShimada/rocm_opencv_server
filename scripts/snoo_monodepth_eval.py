"""単眼の metric の深度のモデルを、PiCar-X の画像で比べる（オフライン）。

  python -m scripts.snoo_monodepth_eval infer --model da2-small   # 1 つずつ GPU で深度を作る
  python -m scripts.snoo_monodepth_eval detector                 # 今の検出の clear_cm を作り直す
  python -m scripts.snoo_monodepth_eval report                   # GPU なしで比べて表と絵を出す

答え合わせ（外のセンサーがまだ無い）
- 床の幾何（主）: 床の画素の本当の奥行きは、カメラの高さ 8 cm と tilt で決まる
  （app/snoo_monodepth.floor_depth）。床の画素は VGGT の点のうち床の平面から ±2 cm のもの
  （ステーションの recon.npz）と、tilt -15° の画像の下 20 %（ほぼ必ず床）。
- VGGT との一致: recon.npz の点を、床からのカメラの高さが 8 cm になるよう cm にしたもの。
  床でない画素（壁・家具）で比べる。VGGT も見積もりなので、目安とする。
- clear_cm: 走っている間の画像（d000xx）で、metric の深度から出した clear_cm を、今の検出の
  やり方（app/clearance.Estimator を作り直して回したもの）と比べる。止まって撮った tilt -15° の
  画像では VGGT の clear_cm とも比べる。Pi の clear.jsonl は手元に無い（Pi につながらなかった）。

電気: 家のブレーカーが落ちたので、GPU の仕事は 1 つずつ、1 コマごとに休み、GPU の電力の
平均（power1_average）が POWER_LIMIT_W を越えたら下がるまで待つ。CPU は 4 つまで。

結果（2026-10-31、10/30 の探索 2 回の 1411 コマ。くわしくは report.json）
- どの metric のモデルも、床から 8 cm の視点では奥行きを 2〜8 倍に見積もる。近くの床での
  本当 / 見積もり の中央値は UniDepth V2 L（測った K）0.52・DepthPro（測った画角）0.53・
  DA V2 Base 0.36・Small 0.31・UniDepth V2 S 0.22・ZoeDepth 0.13。
  暗いコマ（灰色の平均 < 50）ではさらに遠くに見る。
  コマごとにも 0.2〜0.7 とばらつくので、決まった数をかけて直すこともできない。
- そのままの深度で clear_cm を出すと、今の検出より 20 cm 以上長い（障害物を見落とす）コマが
  6 割を超える。走るときには使えない。
- 近くの床で尺度（floor_scale_check）か逆数の一次式（floor_affine）を合わせ直すと、今の検出と
  中央値 5〜15 cm で揃う。今の検出（相対の DA V2 Small を床に合わせる）と同じことになり、
  metric のモデルにしても得るものは少ない。VGGT と比べて今の検出より良かったのは
  UniDepth V2 L + 床の尺度（clear_cm の差の中央値 6 cm、10 cm 以内 64 %。48 ms・2.1 GB）と
  DepthPro + 床の尺度（9 cm・54 %。386 ms・4 GB）。今の検出は 14〜17 cm・38〜41 %。
  UniDepth は transformers に無いので、別の venv で scratchpad の小さなスクリプトから回した
  （pip install --no-deps で入る。MonoDepth は包んでいない）。
- 床で尺度を合わせても、壁や家具は VGGT より 13〜29 % 近く出る（遠いほど縮める）。単眼の深度は
  地図の尺度の確かめには向かない。VGGT 自身を床の幾何と比べる方が早くて確か（中央値の比 0.93）。

検出につなぐなら（まだつながない）
- 検出は 15 fps で返事を返し、clear_cm は 45 ms ごと。単眼の深度は 5 Hz（200 ms ごと）に落とし、
  ClearanceWorker と同じく検出の返事を送った後に、別の CUDA stream で最新のフレームにかける。
  UniDepth V2 L で 48 ms（GPU の 2 割強）、transformers だけで済ますなら DA V2 Base で 18 ms。
  GPU を使うプロセスを増やすと待っている間も GPU が
  120 W に張りつくので、別のサービスにせず検出のプロセスの中で回す。
- floor_scale_check が ok でない（床の当てはまりが半分未満・比が外れる）ときは使わない。
  深度は 200 ms 古くなるので、走った分（最大 3 cm）を clear_cm から引き、今の見積もりと
  近い方（combine）を取る。
"""

import argparse
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from app import snoo_mapper
from app.clearance import Camera
from app.clearance import Path as CarPath
from app.snoo_monodepth import (
    MODELS,
    clearance_from_metric,
    depth_stats,
    floor_affine,
    floor_depth,
    floor_scale_check,
)

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = [
    "data/snoo_map/sessions/explore_20261030_190244",
    "data/snoo_map/sessions/explore_20261030_185508",
]
DEFAULT_OUT = Path(os.getenv("SNOO_MONODEPTH_OUT", "/tmp/snoo_monodepth"))
OUT = DEFAULT_OUT  # --out で変える
# 比べる大きさ（深度は半分にして残す。床・VGGT も同じ大きさで比べる）
W, H = 320, 240
POWER_FILE = Path("/sys/class/drm/card1/device/hwmon/hwmon5/power1_average")
POWER_LIMIT_W = 150.0
# 明るさ（画像全体の灰色の平均）の区切り。止まった画像の 5/50/95 % が 40/63/86
BRIGHT_BINS = [(0, 50, "暗い <50"), (50, 70, "中 50-70"), (70, 256, "明るい >=70")]
DIST_BINS = [(0, 50), (50, 100), (100, 300)]
# 床の幾何の答え合わせは 3 m まで（それより遠い床は tilt の誤差が大きく効く）
FLOOR_MAX_CM = 300.0
# ソファの下のすき間（目で決めた枠: x0, y0, x1, y1。640x480）。front は座面の前の板
SOFA = {
    "004/47_pan+000_tilt+00.jpg": {"front": (60, 140, 400, 185), "gap": (260, 235, 520, 275)},
    "027/277_pan+000_tilt+00.jpg": {"front": (160, 160, 560, 200), "gap": (260, 235, 620, 285)},
    "034/347_pan+000_tilt+00.jpg": {"front": (120, 170, 560, 205), "gap": (130, 225, 600, 280)},
}
PANELS = [
    "014/147_pan+000_tilt+00.jpg",  # 白い壁
    "011/117_pan+000_tilt+00.jpg",  # 白い壁と角
    "004/47_pan+000_tilt+00.jpg",  # ソファの下のすき間
    "027/277_pan+000_tilt+00.jpg",  # ソファの下のすき間
    "065/657_pan+000_tilt+00.jpg",  # 暗い
    "051/517_pan+000_tilt+00.jpg",  # 物が多い
    "021/212_pan+000_tilt-15.jpg",  # 椅子の脚
]


# --- コマの一覧 ---


def frames(sessions: list[str]) -> list[dict]:
    """ステーションの meta.json から、止まった画像と走っている間の画像を集める。"""
    out = []
    for s in sessions:
        sdir = ROOT / s
        for st in sorted(p for p in sdir.iterdir() if re.fullmatch(r"\d{3}", p.name)):
            meta = st / "meta.json"
            if not meta.exists():
                continue
            for f in json.loads(meta.read_text()).get("frames", []):
                p = st / f["file"]
                if not p.exists():
                    continue
                out.append(
                    {
                        "key": f"{sdir.name}/{st.name}/{f['file']}",
                        "path": str(p),
                        "session": sdir.name,
                        "station": int(st.name),
                        "file": f["file"],
                        "tilt": float(f["tilt_deg"]),
                        "pan": float(f["pan_deg"]),
                        "steer": float(f.get("steer_deg", 0.0)),
                        "moving": bool(f.get("moving", f["file"].startswith("d"))),
                    }
                )
    return out


# --- 電気 ---


def power_w() -> float | None:
    try:
        return int(POWER_FILE.read_text()) / 1e6
    except (OSError, ValueError):
        return None


class Governor:
    """GPU の電力の平均を POWER_LIMIT_W より下に保つ。1 コマごとに、かかった時間の rest 倍休む。"""

    def __init__(self, rest: float = 1.0):
        self.rest, self.samples = rest, []

    def after(self, busy_s: float) -> None:
        time.sleep(busy_s * self.rest)
        p = power_w()
        while p is not None and p > POWER_LIMIT_W * 0.85:
            time.sleep(1.0)
            p = power_w()
        if p is not None:
            self.samples.append(p)

    def summary(self) -> dict:
        s = self.samples
        return (
            {"power_mean_w": round(statistics.mean(s), 1), "power_max_w": round(max(s), 1)}
            if s
            else {}
        )


# --- GPU の仕事 ---


def cmd_infer(args) -> None:
    import torch

    from app.snoo_monodepth import MonoDepth

    torch.set_num_threads(4)
    label = args.label or args.model
    size = tuple(int(x) for x in args.size.split("x")) if args.size else None
    t = time.perf_counter()
    m = MonoDepth(args.model, size=size, known_fov=args.known_fov)
    load_s = time.perf_counter() - t
    items = frames(args.sessions)
    if args.limit:
        items = items[: args.limit]
    img0 = cv2.imread(items[0]["path"])
    for _ in range(3):
        m.depth_cm(img0)
    torch.cuda.reset_peak_memory_stats()
    gov = Governor(args.rest)
    depth = np.zeros((len(items), H, W), np.float16)
    times, fovs = defaultdict(list), []
    for i, it in enumerate(items):
        img = cv2.imread(it["path"])
        t = time.perf_counter()
        d = m.depth_cm(img)
        busy = time.perf_counter() - t
        depth[i] = cv2.resize(d, (W, H), interpolation=cv2.INTER_AREA)
        for k, v in m.timing.items():
            times[k].append(v)
        if m.last_fov_deg is not None:
            fovs.append(m.last_fov_deg)
        gov.after(busy)
        if i % 200 == 0:
            print(
                f"{label}: {i}/{len(items)} {m.timing['total_ms']:.1f} ms {power_w()} W", flush=True
            )
    OUT.mkdir(parents=True, exist_ok=True)
    keys = [it["key"] for it in items]
    np.savez_compressed(OUT / f"depth_{label}.npz", keys=keys, depth=depth)
    if fovs and not args.known_fov:
        # DepthPro の深度は焦点距離に比例する。推定した画角の代わりに測った画角（54°）を使った版も
        # 同じ推論から作れる（GPU をもう一度回さない）
        f_est = np.tan(np.radians(np.array(fovs)) / 2)
        k = (f_est / math.tan(math.radians(Camera().hfov_deg) / 2)).astype(np.float32)
        known = (depth.astype(np.float32) * k[:, None, None]).astype(np.float16)
        np.savez_compressed(OUT / f"depth_{label}-f.npz", keys=keys, depth=known)
    info = {
        "model": MODELS.get(args.model, args.model),
        "label": label,
        "size": list(m.size) if m.kind == "depth_anything" else None,
        "dtype": str(m.dtype),
        "load_s": round(load_s, 1),
        "frames": len(items),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "params_m": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 1),
        **{f"{k}_p50": round(float(np.median(v)), 2) for k, v in times.items()},
        **{f"{k}_p95": round(float(np.percentile(v, 95)), 2) for k, v in times.items()},
        **gov.summary(),
    }
    if fovs:
        info["fov_est_deg_median"] = round(float(np.median(fovs)), 1)
    (OUT / f"info_{label}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(json.dumps(info, ensure_ascii=False))


def clear_targets(items: list[dict]) -> list[dict]:
    """clear_cm を比べるコマ。走っている間の全部と、止まった tilt -15°・|pan| <= 15° のもの。"""
    return [
        it for it in items if it["moving"] or (round(it["tilt"]) == -15 and abs(it["pan"]) <= 15)
    ]


def cmd_detector(args) -> None:
    """今の検出の clear_cm（相対の DA V2 Small を床に合わせたもの + 床の見た目）を作り直す。"""
    import torch

    from app.clearance import Estimator

    torch.set_num_threads(4)
    est = Estimator(compile=False)
    items = clear_targets(frames(args.sessions))
    gov = Governor(args.rest)
    rows = []
    for i, it in enumerate(items):
        img = cv2.imread(it["path"])
        t = time.perf_counter()
        h, w = img.shape[:2]
        cam = Camera.from_pi(it["tilt"], it["pan"], width=w, height=h)
        dep = est.depth.estimate(img, cam, it["steer"])
        flo = est.floor.estimate(img, cam, it["steer"])
        busy = time.perf_counter() - t
        rows.append(
            {
                "key": it["key"],
                "depth_cm": dep.clear_cm,
                "depth_conf": dep.confidence,
                "floor_cm": flo.clear_cm,
                "floor_conf": flo.confidence,
            }
        )
        gov.after(busy)
        if i % 200 == 0:
            print(f"detector: {i}/{len(items)} {power_w()} W", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "detector.json").write_text(json.dumps(rows))
    print("detector:", len(rows), gov.summary())


# --- VGGT を cm にする ---


def station_tilts(sdir: Path) -> dict[str, float]:
    out = {}
    for st in sdir.iterdir():
        meta = st / "meta.json"
        if re.fullmatch(r"\d{3}", st.name) and meta.exists():
            for f in json.loads(meta.read_text()).get("frames", []):
                out[f"{st.name}/{f['file']}"] = float(f["tilt_deg"])
    return out


def _grid_index(n_out: int, n_img: int, n_vggt: int, n_grid: int) -> np.ndarray:
    """比べる大きさの画素 → VGGT の点の格子（2 画素おき）の番号。

    VGGT は 640x480 を、切らずに 518x392 へ伸ばす。
    """
    x = (np.arange(n_out) + 0.5) * n_img / n_out - 0.5
    xv = (x + 0.5) * n_vggt / n_img - 0.5
    return np.clip(np.round(xv / 2).astype(int), 0, n_grid - 1)


def vggt_station(sdir: Path, station: int, tilts: dict[str, float]) -> dict:
    """recon.npz から、そのステーションのコマの VGGT の深度と床からの高さ（cm）。

    床は snoo_mapper.fit_floor（地図と同じ）。大きさは、このステーションのカメラの中心の
    床からの高さ（中央値）が 8 cm になるように決める。
    """
    p = sdir / f"{station:03d}" / "recon.npz"
    if not p.exists():
        return {}
    d = np.load(p)
    keys = [str(k) for k in d["keys"]]
    ext, pts, conf = d["extrinsic"], d["points"], d["conf"]
    F = len(keys)
    flat, cf = pts.reshape(F, -1, 3), conf.reshape(F, -1)
    finite = np.isfinite(flat).all(-1) & (cf > 0)
    thr = np.percentile(cf[finite], 50)
    good = finite & (cf >= thr)
    tl = np.array([tilts.get(k, 0.0) for k in keys])
    up = snoo_mapper.up_in_world(ext, tl).mean(0)
    cams = snoo_mapper.camera_centers(ext)
    floor = snoo_mapper.fit_floor(flat[good], up, cams)
    if floor is None:
        return {}
    mine = [i for i, k in enumerate(keys) if k.startswith(f"{station:03d}/")]
    if not mine:
        return {}
    h_cam = float(np.median(floor.height(cams[mine])))
    if h_cam <= 0:
        return {}
    scale = 8.0 / h_cam  # cm / VGGT の長さ
    gh, gw = pts.shape[1:3]
    rows = _grid_index(H, 480, 392, gh)
    cols = _grid_index(W, 640, 518, gw)
    out = {}
    for i in mine:
        E = ext[i]
        P = pts[i][rows][:, cols]
        z = (P @ E[:, :3].T + E[:, 3])[..., 2] * scale
        hcm = floor.height(P) * scale
        ok = good.reshape(pts.shape[:3])[i][rows][:, cols]
        axis = E[2, :3] / np.linalg.norm(E[2, :3])
        tilt = math.degrees(math.asin(float(np.clip(axis @ floor.normal, -1, 1))))
        out[keys[i].split("/", 1)[1]] = {
            "z": np.where(ok, z, np.nan).astype(np.float32),
            "z_all": z.astype(np.float32),  # 確からしさで落とさない（ソファの下のすき間を見るため）
            "h": np.where(ok, hcm, np.nan).astype(np.float32),
            "tilt": tilt,
            "cam_h_cm": float(floor.height(cams[i]) * scale),
        }
    return out


# --- 比べる ---


def brightness(path: str) -> float:
    return float(cv2.imread(path, cv2.IMREAD_GRAYSCALE).mean())


def bin_of(v: float, bins) -> str:
    for lo, hi, name in bins:
        if lo <= v < hi:
            return name
    return bins[-1][2]


def white_wall_mask(img: np.ndarray, vg: dict) -> np.ndarray:
    """白い壁らしい画素: 床から 15 cm より上で、明るく、色が薄く、模様が無い所。"""
    small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (7, 7))
    std = np.sqrt(np.maximum(cv2.blur(gray * gray, (7, 7)) - mean * mean, 0))
    bright = gray > np.percentile(gray, 60)
    return (vg["h"] > 15) & bright & (hsv[..., 1] < 70) & (std < 4)


def pooled(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict | None:
    if not pairs:
        return None
    p = np.concatenate([a for a, _ in pairs])
    g = np.concatenate([b for _, b in pairs])
    s = depth_stats(p, g)
    return s.to_json() if s else None


def pairs_of(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    ok = mask & np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    return pred[ok].astype(np.float32), gt[ok].astype(np.float32)


def agreement(a: list, b: list) -> dict:
    """2 つの clear_cm の一致（両方が出ているコマだけ）。"""
    both = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not both:
        return {"n": 0}
    d = np.array([x - y for x, y in both])
    return {
        "n": len(both),
        "median_abs_cm": round(float(np.median(np.abs(d))), 1),
        "within_10cm": round(float(np.mean(np.abs(d) <= 10)), 2),
        "shorter_20cm": round(float(np.mean(d < -20)), 2),  # a が 20 cm 以上短い（止まりすぎ）
        "longer_20cm": round(float(np.mean(d > 20)), 2),  # a が 20 cm 以上長い（見落とし）
        "null_a": sum(x is None for x in a),
    }


def colorize(d: np.ndarray, lo: float = 0, hi: float = 300) -> np.ndarray:
    x = np.clip((np.nan_to_num(d, nan=hi) - lo) / (hi - lo), 0, 1)
    c = cv2.applyColorMap((255 - x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    c[~np.isfinite(d)] = 40
    return c


def err_color(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """log(pred / gt) を色に。青 = 近すぎ、赤 = 遠すぎ、白 = 合う、灰 = 答えが無い。

    ±log 3 で振り切る。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.log(pred / gt) / math.log(3)
    e = np.clip(e, -1, 1)
    out = np.full((*pred.shape, 3), 60, np.uint8)
    ok = np.isfinite(e)
    pos, neg = np.clip(e, 0, 1), np.clip(-e, 0, 1)
    out[..., 2] = np.where(ok, 255 - neg * 255, 60)
    out[..., 1] = np.where(ok, 255 - (pos + neg) * 255, 60)
    out[..., 0] = np.where(ok, 255 - pos * 255, 60)
    return out


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (4, 16), 0, 0.45, (0, 0, 0), 3)
    cv2.putText(img, text, (4, 16), 0, 0.45, (255, 255, 255), 1)
    return img


def cmd_report(args) -> None:
    cv2.setNumThreads(4)
    items = frames(args.sessions)
    by_key = {it["key"]: it for it in items}
    labels = [p.stem.removeprefix("depth_") for p in sorted(OUT.glob("depth_*.npz"))]
    depths = {}
    for lb in labels:
        d = np.load(OUT / f"depth_{lb}.npz")
        depths[lb] = dict(zip([str(k) for k in d["keys"]], d["depth"], strict=True))
    print("models:", labels)

    # VGGT（ステーションごと）
    vggt = {}
    for s in args.sessions:
        sdir = ROOT / s
        tl = station_tilts(sdir)
        for st in sorted({it["station"] for it in items if it["session"] == sdir.name}):
            for f, v in vggt_station(sdir, st, tl).items():
                vggt[f"{sdir.name}/{st:03d}/{f}"] = v
    print("VGGT frames:", len(vggt))

    cams = {}
    floors = {}
    for it in items:
        cam = Camera.from_pi(it["tilt"], it["pan"]).scaled(W, H)
        cams[it["key"]] = cam
        tk = round(it["tilt"] * 2) / 2
        if tk not in floors:
            floors[tk] = floor_depth(Camera(tilt_deg=tk).scaled(W, H))
        it["bright"] = brightness(it["path"])
        it["bbin"] = bin_of(it["bright"], BRIGHT_BINS)

    vv, _ = np.mgrid[0:H, 0:W]
    bottom = vv >= int(H * 0.8)
    res = {"frames": len(items), "vggt_frames": len(vggt)}

    # VGGT 自身を床の幾何と比べる（VGGT を目安に使ってよいか）
    vp, tdiff = [], []
    for k, v in vggt.items():
        it = by_key.get(k)
        if it is None or it["moving"]:
            continue
        fl = floors[round(it["tilt"] * 2) / 2]
        m = (np.abs(v["h"]) <= 2) & (fl <= FLOOR_MAX_CM)
        vp.append(pairs_of(v["z"], fl, m))
        tdiff.append(v["tilt"] - it["tilt"])
    res["vggt_vs_floor"] = pooled(vp)
    res["vggt_tilt_minus_cmd_deg"] = {
        "median": round(float(np.median(tdiff)), 2),
        "p10": round(float(np.percentile(tdiff, 10)), 2),
        "p90": round(float(np.percentile(tdiff, 90)), 2),
    }

    per_model = {}
    for lb in labels:
        D = depths[lb]
        acc = defaultdict(list)
        frame_ratios = []
        for k, it in by_key.items():
            if k not in D:
                continue
            pred = D[k].astype(np.float32)
            fl = floors[round(it["tilt"] * 2) / 2]
            v = vggt.get(k)
            if v is not None and not it["moving"]:
                m = (np.abs(v["h"]) <= 2) & np.isfinite(fl) & (fl <= FLOOR_MAX_CM)
                pr = pairs_of(pred, fl, m)
                acc["floor_vggt"].append(pr)
                acc[f"floor_vggt|{it['bbin']}"].append(pr)
                acc[f"floor_vggt|tilt{round(it['tilt']):+d}"].append(pr)
                for lo, hi in DIST_BINS:
                    acc[f"floor_vggt|{lo}-{hi}cm"].append(
                        pairs_of(pred, fl, m & (fl >= lo) & (fl < hi))
                    )
                nf = (v["h"] > 3) & (v["z"] < 400)
                pr = pairs_of(pred, v["z"], nf)
                acc["vggt_nonfloor"].append(pr)
                acc[f"vggt_nonfloor|{it['bbin']}"].append(pr)
                sc = floor_scale_check(pred, cams[k])
                if sc.ratio is not None:
                    acc["vggt_nonfloor_scaled"].append(pairs_of(pred * sc.ratio, v["z"], nf))
                    frame_ratios.append(sc.ratio)
                img = cv2.imread(it["path"])
                wm = white_wall_mask(img, v) & nf
                acc["white_wall"].append(pairs_of(pred, v["z"], wm))
                if sc.ratio is not None:
                    acc["white_wall_scaled"].append(pairs_of(pred * sc.ratio, v["z"], wm))
                aff = floor_affine(pred, cams[k])
                if aff is not None:
                    acc["vggt_nonfloor_affine"].append(pairs_of(aff[0], v["z"], nf))
                    acc["white_wall_affine"].append(pairs_of(aff[0], v["z"], wm))
            if round(it["tilt"]) == -15:
                name = "floor_bottom_moving" if it["moving"] else "floor_bottom_still"
                pr = pairs_of(pred, fl, bottom & np.isfinite(fl))
                acc[name].append(pr)
                acc[f"{name}|{it['bbin']}"].append(pr)
        per_model[lb] = {k: pooled(v) for k, v in sorted(acc.items())}
        if frame_ratios:
            per_model[lb]["floor_ratio_per_frame"] = {
                "median": round(float(np.median(frame_ratios)), 3),
                "p10": round(float(np.percentile(frame_ratios, 10)), 3),
                "p90": round(float(np.percentile(frame_ratios, 90)), 3),
            }
        print(lb, json.dumps(per_model[lb].get("floor_vggt"), ensure_ascii=False))
    res["models"] = per_model
    res["brightness_counts"] = {
        b[2]: sum(1 for it in items if it["bbin"] == b[2] and not it["moving"]) for b in BRIGHT_BINS
    }

    # ソファの下のすき間
    sofa = {}
    for f, boxes in SOFA.items():
        k = f"explore_20261030_190244/{f}"
        row = {}
        v = vggt.get(k)
        srcs = {lb: depths[lb].get(k) for lb in labels}
        if v is not None:
            srcs["vggt"] = v["z"]
            srcs["vggt_all"] = v["z_all"]
        for name, d in srcs.items():
            if d is None:
                continue
            d = d.astype(np.float32)

            def med(box, d=d):
                x0, y0, x1, y1 = (round(c / 2) for c in box)
                return float(np.nanmedian(d[y0:y1, x0:x1]))

            row[name] = {
                "front_cm": round(med(boxes["front"]), 1),
                "gap_cm": round(med(boxes["gap"]), 1),
            }
            row[name]["gap_over_front"] = round(row[name]["gap_cm"] / row[name]["front_cm"], 2)
        sofa[f] = row
    res["sofa_gap"] = sofa

    # clear_cm
    det = {}
    if (OUT / "detector.json").exists():
        det = {r["key"]: r for r in json.loads((OUT / "detector.json").read_text())}
    targets = clear_targets(items)
    clears = defaultdict(dict)
    for it in targets:
        k = it["key"]
        cam = Camera.from_pi(it["tilt"], it["pan"], width=W, height=H)
        path = CarPath(steer_deg=it["steer"])
        for lb in labels:
            if k in depths[lb]:
                pred = depths[lb][k].astype(np.float32)
                clears[f"{lb}"][k] = clearance_from_metric(pred, cam, path).clear_cm
                sc = floor_scale_check(pred, cam)
                if sc.ratio is not None:
                    c = clearance_from_metric(pred, cam, path, scale=sc.ratio).clear_cm
                    clears[f"{lb}+floor"][k] = c
                aff = floor_affine(pred, cam)
                if aff is not None:
                    clears[f"{lb}+affine"][k] = clearance_from_metric(aff[0], cam, path).clear_cm
        if k in det:
            r = det[k]
            # 検出と同じく、確からしさが CLEAR_MIN_CONF（0.5）より低いものは null（app/snoo.py）
            ok_d = r["depth_cm"] is not None and r["depth_conf"] >= 0.5
            ok_f = r["floor_cm"] is not None
            clears["detector_depth"][k] = r["depth_cm"] if ok_d else None
            ok = ok_d and ok_f and min(r["depth_conf"], r["floor_conf"]) >= 0.5
            clears["detector"][k] = min(r["depth_cm"], r["floor_cm"]) if ok else None
        v = vggt.get(k)
        if v is not None:
            z = np.where(np.isfinite(v["z"]) & (v["z"] > 0), v["z"], np.nan)
            clears["vggt"][k] = clearance_from_metric(z, cam, path).clear_cm
    cl = {}
    for group, sel in (
        ("moving", [it["key"] for it in targets if it["moving"]]),
        ("still", [it["key"] for it in targets if not it["moving"]]),
    ):
        g = {}
        for ref in ("detector", "detector_depth", "vggt"):
            if not any(k in clears[ref] for k in sel):
                continue
            for name in sorted(clears):
                if name == ref:
                    continue
                a = [clears[name].get(k) for k in sel if k in clears[ref]]
                b = [clears[ref].get(k) for k in sel if k in clears[ref]]
                g[f"{name} vs {ref}"] = agreement(a, b)
        g["n_frames"] = len(sel)
        cl[group] = g
    res["clear"] = cl

    # 時間とメモリ
    res["runtime"] = {
        lb: json.loads((OUT / f"info_{lb}.json").read_text())
        for lb in labels
        if (OUT / f"info_{lb}.json").exists()
    }
    (OUT / "report.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("wrote", OUT / "report.json")

    # 絵
    for f in PANELS:
        k = f"explore_20261030_190244/{f}"
        it = by_key.get(k)
        if it is None:
            continue
        img = cv2.resize(cv2.imread(it["path"]), (W, H), interpolation=cv2.INTER_AREA)
        v = vggt.get(k)
        fl = floors[round(it["tilt"] * 2) / 2]
        ref = v["z"] if v is not None else np.full((H, W), np.nan, np.float32)
        if v is not None:  # 床の画素は床の幾何を答えにする
            ref = np.where(np.abs(v["h"]) <= 2, fl, ref)
        top = [label(img, f"{f} b={it['bright']:.0f}"), label(colorize(ref), "ref: VGGT + floor")]
        top.append(label(colorize(fl), "floor geometry"))
        rows = [np.hstack(top)]
        for lb in labels:
            if k not in depths[lb]:
                continue
            pred = depths[lb][k].astype(np.float32)
            sc = floor_scale_check(pred, cams[k])
            r = sc.ratio or 1.0
            rows.append(
                np.hstack(
                    [
                        label(colorize(pred), f"{lb} raw"),
                        label(colorize(pred * r), f"{lb} x{r:.2f} (floor)"),
                        label(err_color(pred * r, ref), "err scaled vs ref"),
                    ]
                )
            )
        name = f.replace("/", "_").replace(".jpg", ".png")
        cv2.imwrite(str(OUT / f"panel_{name}"), np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sessions", nargs="+", default=SESSIONS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="深度・表・絵の置き場")
    ap.add_argument("--rest", type=float, default=1.0, help="1 コマにかかった時間の何倍休むか")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("infer")
    p.add_argument("--model", required=True, help=f"{', '.join(MODELS)} か HF の名前")
    p.add_argument("--label", help="残すファイルの名前（既定はモデルの短い名前）")
    p.add_argument("--size", help="DA V2 の入力（幅x高さ、14 の倍数）")
    p.add_argument("--known-fov", action="store_true", help="DepthPro に測った画角を使う")
    p.add_argument("--limit", type=int, default=0)
    sub.add_parser("detector")
    sub.add_parser("report")
    args = ap.parse_args()
    global OUT
    OUT = args.out
    {"infer": cmd_infer, "detector": cmd_detector, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
