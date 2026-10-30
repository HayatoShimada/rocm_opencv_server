"""snoo の写真から毛の色と模様を測り、シミュレーターの猫の模様（app/assets/snoo/）を作る。

    .venv/bin/python -m scripts.snoo_fur_texture --analyze   # 写真を測って fur.json を書く（CPU）
    .venv/bin/python -m scripts.snoo_fur_texture --make      # fur.json から模様の PNG を作る

測る（--analyze）：data/snoo_raw の写真に猫の検出（RT-DETR）と DeepLabCut（app/cat_pose.py）を
CPU でかけ、箱から GrabCut で猫を切り抜く。部屋の明かりの色は写真ごとに違うので、猫の外の
白っぽい所（壁・床・布）を灰色とみなして色をそろえる。次に DeepLabCut の点のまわりから、
背中・脇・腹・胸・顔・耳の内側・目・鼻・脚・しっぽの色を取り、写真をまたいだ中央値にする。
縞の濃さ（明るさの細かい揺れ）も背中と脚で測る。前足の縞の間隔は、正面から歩いてくる動画
（data/snoo_video/IMG_5903）の、肘から足先までの明るさの波から測る。
作る（--make）：fur.json の色と縞から、決まったくじ（seed）で模様を作る。MuJoCo は球と楕円には
緯度・経度で、カプセルには軸に沿って 2 次元の模様を貼るので、模様の上の行が体の上（背中）、
下の行が腹になるように描く。毛の外側の殻（半分すける）には、毛の筋の形にすけた模様を使う。
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.cat_pose import NAMES  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "app" / "assets" / "snoo"
PHOTOS = Path("data/snoo_raw")
VIDEO_POSE = Path("data/snoo_video/IMG_5903_pose.json")
VIDEO = Path("data/snoo_video/IMG_5903.mov")
PX = 1024  # 写真は長い辺をこの画素に縮める
CONF = 0.6
I = {n: k for k, n in enumerate(NAMES)}  # noqa: E741

# 部位ごとに、色を取る点（2 点なら線の上を取る）と、箱の大きさに対する半径
REGIONS: dict[str, tuple[tuple[str, ...], float]] = {
    "back": (("back_middle", "back_base", "back_end"), 0.05),
    "side": (("body_middle_left", "body_middle_right"), 0.06),
    "belly": (("belly_bottom",), 0.05),
    "chest": (("throat_base", "throat_end"), 0.045),
    "face": (("upper_jaw", "mouth_end_left", "mouth_end_right"), 0.025),
    "forehead": (("left_earbase", "right_earbase"), 0.03),
    "ear_inner": (("left_earbase", "left_earend", "right_earbase", "right_earend"), 0.015),
    "eye": (("left_eye", "right_eye"), 0.008),
    "nose": (("nose",), 0.008),
    "leg": (("front_left_knee", "front_right_knee", "back_left_knee", "back_right_knee"), 0.03),
    "paw": (("front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw"), 0.02),
    "tail": (("tail_base", "tail_end"), 0.05),
    "tail_tip": (("tail_end",), 0.03),
}


def load(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    s = PX / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def detect_all(keys: list[Path], cache: Path) -> dict:
    have = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [k for k in keys if k.name not in have]
    if todo:
        from app.cat_pose import CatPose
        from app.snoo import CatDetector

        det = CatDetector(cat_min=0.5)
        if det.device != "cpu":
            raise SystemExit("GPU を使わないはずが、検出が GPU に載った")
        pose = CatPose(device="cpu")
        for n, k in enumerate(todo):
            img = load(k)
            dets = det.detect(img)
            have[k.name] = None
            if dets:
                kp = pose.estimate(img, dets[0].box)
                have[k.name] = {
                    "box": [round(float(v), 4) for v in dets[0].box],
                    "cat": round(float(dets[0].cat), 3),
                    "keypoints": kp.round(2).tolist(),
                }
            if n % 20 == 0:
                print(f"{n}/{len(todo)}")
                cache.write_text(json.dumps(have))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have))
    return have


def segment(img: np.ndarray, box) -> np.ndarray:
    """箱から GrabCut で猫を切り抜く（True が猫）。"""
    h, w = img.shape[:2]
    x0, y0 = int(box[0] * w), int(box[1] * h)
    x1, y1 = int(box[2] * w), int(box[3] * h)
    rect = (max(x0, 1), max(y0, 1), max(x1 - x0 - 2, 2), max(y1 - y0 - 2, 2))
    mask = np.zeros((h, w), np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bg, fg, 4, cv2.GC_INIT_WITH_RECT)
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


def white_gains(img: np.ndarray, cat: np.ndarray) -> np.ndarray | None:
    """猫の外の白っぽい所（彩度が低く明るい）を灰色とみなす、色ごとの掛け算（BGR）。"""
    out = ~cv2.dilate(cat.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(float)
    ok = out & (hsv[..., 1] < 40) & (v > 60) & (v < 250)
    if ok.sum() < 2000:
        return None
    sel = ok & (v >= np.percentile(v[ok], 70))
    ref = img[sel].astype(float).mean(0)
    return ref.mean() / np.maximum(ref, 1)


def sample(img, cat, kp, names, radius_px) -> np.ndarray | None:
    """点（2 点以上なら隣どうしを結ぶ線）のまわりで、猫の中の画素を集める。"""
    pts = [kp[I[n]] for n in names if kp[I[n]][2] >= CONF]
    if not pts:
        return None
    m = np.zeros(cat.shape, np.uint8)
    r = max(int(radius_px), 2)
    for x, y, _ in pts:
        cv2.circle(m, (int(x), int(y)), r, 1, -1)
    if len(names) == 2 and len(pts) == 2:
        cv2.line(m, tuple(int(v) for v in pts[0][:2]), tuple(int(v) for v in pts[1][:2]), 1, r)
    m = m.astype(bool) & cat
    if m.sum() < 12:
        return None
    return img[m]


def leg_band_period() -> dict:
    """正面から歩いてくる動画で、前足の肘から足先までの明るさの波の周期を測る（脚の長さに対する比）。"""
    d = json.loads(VIDEO_POSE.read_text())
    cap = cv2.VideoCapture(str(VIDEO))
    periods = []
    k = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        f = d["frames"][k] if k < len(d["frames"]) else None
        k += 1
        if not f or not f.get("keypoints") or not 100 <= k <= 160:
            continue
        kp = f["keypoints"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        for side in ("left", "right"):
            a, b = kp[I[f"front_{side}_knee"]], kp[I[f"front_{side}_paw"]]
            if min(a[2], b[2]) < 0.7:
                continue
            n = int(np.hypot(b[0] - a[0], b[1] - a[1]))
            if n < 80:
                continue
            xs, ys = np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)
            prof = np.array([gray[int(y) - 3 : int(y) + 4, int(x) - 3 : int(x) + 4].mean()
                             for x, y in zip(xs, ys, strict=True)])  # fmt: skip
            prof = prof - np.convolve(prof, np.ones(31) / 31, mode="same")
            prof = prof[15:-15] * np.hanning(len(prof) - 30)
            spec = np.abs(np.fft.rfft(prof, 1024))
            freqs = np.fft.rfftfreq(1024)  # 1 画素あたりの回数
            band = (freqs > 2.0 / n) & (freqs < 12.0 / n)  # 脚 1 本に 2〜12 本
            if band.any():
                f0 = freqs[band][np.argmax(spec[band])]
                periods.append(float(n * f0))  # 肘〜足先に縞が何本
    return {
        "bands_per_lower_leg": round(float(np.median(periods)), 2) if periods else None,
        "n": len(periods),
    }


def analyze(out: Path, step: int) -> dict:
    keys = sorted(PHOTOS.glob("img_*.jpg"))[::step]
    cache = out / "fur_keypoints.json"
    out.mkdir(parents=True, exist_ok=True)
    have = detect_all(keys, cache)
    per = {r: [] for r in REGIONS}
    texture = {"back": [], "leg": [], "side": []}
    used = []
    for k in keys:
        rec = have.get(k.name)
        if not rec or rec["cat"] < 0.8:
            continue
        box = rec["box"]
        area = (box[2] - box[0]) * (box[3] - box[1])
        if not 0.04 < area < 0.75:
            continue
        img = load(k)
        cat = segment(img, box)
        gains = white_gains(img, cat)
        if gains is None or cat.mean() < 0.02:
            continue
        bal = np.clip(img.astype(float) * gains, 0, 255)
        kp = np.array(rec["keypoints"])
        size = max((box[2] - box[0]) * img.shape[1], (box[3] - box[1]) * img.shape[0])
        # 明るさは写真ごとに違う。猫の中の明るさの中央値で割って、部位どうしの比で持つ
        lum_cat = float(np.median(bal[cat].mean(1)))
        if lum_cat < 15:
            continue
        used.append(k.name)
        for r, (names, rad) in REGIONS.items():
            px = sample(bal, cat, kp, names, rad * size)
            if px is not None:
                per[r].append(np.median(px, 0) / lum_cat)
        # 縞の濃さ：部位の中の明るさの、細かい揺れ（ぼかしとの差）の標準偏差 / 平均
        gray = bal.mean(2)
        hp = gray - cv2.GaussianBlur(gray, (0, 0), size * 0.02)
        for r in texture:
            names, rad = REGIONS[r]
            m = np.zeros(cat.shape, np.uint8)
            for x, y, c in (kp[I[n]] for n in names):
                if c >= CONF:
                    cv2.circle(m, (int(x), int(y)), int(rad * size * 1.5), 1, -1)
            m = m.astype(bool) & cat
            if m.sum() > 200:
                texture[r].append(float(hp[m].std() / max(gray[m].mean(), 1)))
    # 絶対の明るさ：猫の中の明るさの中央値を、白とみなした所に対する比で持つ
    colors = {}
    for r, vals in per.items():
        if len(vals) >= 3:
            v = np.median(np.array(vals), 0)  # BGR、猫の中の明るさを 1 とした比
            colors[r] = {"bgr_rel": [round(float(x), 3) for x in v], "n": len(vals)}
    result = {
        "about": "snoo の写真から測った毛の色（scripts/snoo_fur_texture.py --analyze）。"
        "bgr_rel は猫の中の明るさの中央値を 1 とした BGR の比（白っぽい所で色をそろえた後）",
        "photos": len(used),
        "colors": colors,
        "streak_contrast": {r: round(float(np.median(v)), 3) for r, v in texture.items() if v},
        "leg_bands": leg_band_period(),
    }
    print(json.dumps(result, indent=1))
    return result


# --- 模様を作る ---

# 手で決める値（測れなかったもの）。fur.json に無ければこれを使う
HAND = {
    # 猫の中の明るさの中央値（0..255）。写真ごとに露出が違い、白い所が無い写真も多いので
    # 測らず、前の一色の毛（FUR_RGBA 0.46 0.43 0.41）と同じ明るさに合わせる
    "albedo_scale": 105.0,
    # 白っぽい所でそろえた後も、部屋の電球の色が少し残るので、色の濃さを少し弱める
    "saturation": 0.75,
    "stripe": {
        "body_bands": 14,  # 胴をぐるりと一周する間のさば縞の本数（片側 約 6 本）
        "body_depth": 0.16,  # 縞の暗さ（長毛でぼやけるので弱め）
        "streak": 0.06,  # 毛の筋の明るさの揺れ（測った streak_contrast は縞も含むので小さく）
        "leg_depth": 0.1,
        "tail_depth": 0.15,
    },
    # 目：写真では瞳孔が大きく写り、点のまわりの色は暗い。色合い（赤:緑:青の比）は測った値、
    # 虹彩の明るさと濃さは決め事（写真で見える金色がかった琥珀色に寄せる）
    "iris_brightness": 170.0,
    "iris_saturation": 2.4,
    "pupil": 0.42,  # 目の前から見た瞳孔の半径（虹彩に対する比）。部屋の暗さで大きい
}
SIZE = {
    "fur_body": (256, 256), "fur_chest": (256, 256), "fur_head": (256, 256),
    "fur_leg": (128, 64), "fur_tail": (128, 64), "fur_eye": (64, 128),
    "fur_shell": (128, 256), "fur_shell_chest": (128, 256), "fur_shell_tail": (128, 256),
}  # fmt: skip


def _noise(rng, h, w, sy, sx) -> np.ndarray:
    """縦 sy・横 sx 画素の大きさの、なめらかなくじの模様（平均 0、標準偏差 1）。

    横（経度）は一周つながる。
    """
    n = rng.standard_normal((h, w))
    pad = min(int(3 * sx) + 2, w)
    n = np.pad(n, ((0, 0), (pad, pad)), mode="wrap")
    n = cv2.GaussianBlur(n, (0, 0), sigmaX=max(sx, 0.3), sigmaY=max(sy, 0.3))[:, pad:-pad]
    return (n - n.mean()) / (n.std() + 1e-9)


def palette(fur: dict) -> dict[str, np.ndarray]:
    """部位ごとの色（BGR、0..255）。測った比 × 明るさ。色の濃さを少し弱める。"""
    hand = HAND | {k: fur[k] for k in HAND if k in fur}
    out = {}
    for k, v in fur["colors"].items():
        c = np.array(v["bgr_rel"]) * hand["albedo_scale"]
        g = c.mean()
        out[k] = g + hand["saturation"] * (c - g)
    eye = np.array(fur["colors"]["eye"]["bgr_rel"])
    eye = eye / eye.mean()
    out["iris"] = (1 + hand["iris_saturation"] * (eye - 1)) * hand["iris_brightness"]
    return out


def rgba(fur: dict) -> dict[str, list[float]]:
    """色だけの材質（鼻・耳の外と内側・口のまわり）の RGBA（0..1）。"""
    p = palette(fur)
    keys = {"nose": "nose", "ear_inner": "ear_inner", "ear_outer": "forehead", "muzzle": "face"}
    return {k: [round(float(x) / 255, 3) for x in p[v][::-1]] + [1.0] for k, v in keys.items()}


def make(fur: dict, seed: int = 0) -> dict[str, np.ndarray]:
    """模様の画像（BGR か BGRA、uint8）を名前ごとに。"""
    rng = np.random.default_rng(seed)
    hand = HAND | {k: fur[k] for k in HAND if k in fur}
    stripe = hand["stripe"]
    p = palette(fur)
    streak = stripe["streak"]
    out = {}

    def grid(name):
        h, w = SIZE[name]
        v = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
        u = np.linspace(0, 1, w, endpoint=False)[None] * np.ones((h, 1))
        return h, w, v, u

    def mix(a, b, t):
        return a[None, None] * (1 - t[..., None]) + b[None, None] * t[..., None]

    def fur_noise(h, w, scale=1.0):
        # 長い毛の筋（縦に長い）と、細かい毛。球の上下の極では経線が集まって筋が放射状に
        # 見えるので、極に近いほど弱める
        pole = np.sin(np.pi * np.linspace(0, 1, h))[:, None] ** 0.7
        return 1 + pole * scale * streak * (
            0.8 * _noise(rng, h, w, h / 16, 2.5) + 0.4 * _noise(rng, h, w, 3, 1.5)
        )

    # 胴（楕円：緯度・経度）。上の行が背中。背中の真ん中に濃い筋、脇にさば縞（経線に沿って
    # 背中から腹へ下りる）、下は明るい腹
    h, w, v, u = grid("fur_body")
    g = mix(p["back"], p["side"], np.clip((v - 0.12) / 0.3, 0, 1))
    g = (
        g * (1 - np.clip((v - 0.55) / 0.25, 0, 1))[..., None]
        + p["belly"] * np.clip((v - 0.55) / 0.25, 0, 1)[..., None]
    )
    wob = 0.015 * _noise(rng, h, w, h / 8, w / 8)
    bands = np.sin(2 * np.pi * (u * stripe["body_bands"] + wob + 0.2 * v * v)) * 0.5 + 0.5
    bands = cv2.GaussianBlur(bands, (0, 0), 3)
    side_mask = np.clip((v - 0.06) / 0.15, 0, 1) * np.clip((0.72 - v) / 0.2, 0, 1)
    g = g * (1 - stripe["body_depth"] * bands * side_mask)[..., None]
    g = g * (1 - 0.25 * np.exp(-((v / 0.09) ** 2)))[..., None]
    out["fur_body"] = g * fur_noise(h, w)[..., None]

    # 胸と首（たてがみ）：上は脇の色、下へ明るい腹の色。長い筋
    h, w, v, u = grid("fur_chest")
    g = mix(p["side"], p["belly"], np.clip((v - 0.1) / 0.5, 0, 1))
    out["fur_chest"] = g * fur_noise(h, w, 1.3)[..., None]

    # 頭（球）：経度 0 が顔の前（+x）。上は濃く額に M の縞、前の下（口のまわり）は明るい
    h, w, v, u = grid("fur_head")
    du = (u + 0.5) % 1 - 0.5  # 前を 0 とした経度
    front = np.cos(2 * np.pi * du) * 0.5 + 0.5
    g = mix(p["forehead"], p["face"], np.clip((v - 0.38) / 0.3, 0, 1) * front)
    m_bands = np.clip(np.sin(2 * np.pi * du * 24), 0, 1) ** 2
    m_mask = np.exp(-((du / 0.07) ** 2)) * np.clip((0.42 - v) / 0.25, 0, 1) * (v > 0.06)
    g = g * (1 - 0.22 * m_bands * m_mask)[..., None]
    cheek = np.exp(-(((np.abs(du) - 0.14) / 0.025) ** 2)) * np.exp(-(((v - 0.52) / 0.035) ** 2))
    g = g * (1 - 0.3 * cheek)[..., None]
    out["fur_head"] = g * fur_noise(h, w, 0.7)[..., None]

    # 脚（カプセル：行が軸に沿う。上が付け根）。輪の縞（動画で数えた本数）、先は足先の色
    h, w, v, u = grid("fur_leg")
    g = mix(p["leg"], 0.5 * (p["leg"] + p["paw"]), np.clip((v - 0.7) / 0.3, 0, 1))
    nb = float(fur.get("leg_bands", {}).get("bands_per_lower_leg") or 6.0)
    ring = np.clip(np.sin(2 * np.pi * (v * nb + 0.06 * _noise(rng, h, w, 6, 6))), 0, 1) ** 1.3
    g = g * (1 - stripe["leg_depth"] * ring)[..., None]
    out["fur_leg"] = g * fur_noise(h, w)[..., None]

    # しっぽ：輪の縞、先は先の色
    h, w, v, u = grid("fur_tail")
    g = mix(p["tail"], p["tail_tip"], np.clip((v - 0.6) / 0.4, 0, 1))
    ring = np.clip(np.sin(2 * np.pi * (v * 3 + 0.05 * _noise(rng, h, w, 6, 6))), 0, 1) ** 1.5
    g = g * (1 - stripe["tail_depth"] * ring)[..., None]
    out["fur_tail"] = g * fur_noise(h, w, 1.3)[..., None]

    # 目（球）：経度 0・緯度の真ん中が前。琥珀色の虹彩に、大きな黒い瞳孔
    h, w, v, u = grid("fur_eye")
    du = (u + 0.5) % 1 - 0.5
    ang = np.hypot(du * 2 * np.pi, (v - 0.5) * np.pi)  # 前からの角度（ラジアン）
    iris = p["iris"] * (1.1 - 0.35 * np.clip(ang / 1.2, 0, 1))[..., None]
    g = iris * (1 + 0.08 * _noise(rng, h, w, 1.5, 3))[..., None]
    g[ang < hand["pupil"] * 1.2] = (12, 10, 10)
    g[ang > 1.5] = p["forehead"] * 0.6
    out["fur_eye"] = g

    # 毛の殻（すける）：色は下の芯と同じ並び、アルファは毛の筋（縦に長い）
    for name, src in (("fur_shell", "fur_body"), ("fur_shell_chest", "fur_chest"),
                      ("fur_shell_tail", "fur_tail")):  # fmt: skip
        h, w = SIZE[name]
        col = cv2.resize(out[src], (w, h), interpolation=cv2.INTER_AREA)
        sy = h / 20 if name != "fur_shell_tail" else h / 10
        a = np.clip(0.7 + 0.18 * _noise(rng, h, w, sy, 2.5), 0, 1)
        lift = 1 + 0.06 * _noise(rng, h, w, sy, 2.5)  # 毛の先は少し明るい
        out[name] = np.dstack([col * lift[..., None], a * 255])
    return {k: np.clip(v, 0, 255).astype(np.uint8) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--make", action="store_true")
    parser.add_argument("--step", type=int, default=3, help="写真を何枚おきに使うか")
    parser.add_argument("--work", type=Path, default=Path("out/fur"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    fur_path = ASSETS / "fur.json"
    if args.analyze:
        measured = analyze(args.work, args.step)
        old = json.loads(fur_path.read_text()) if fur_path.exists() else {}
        # 手で決める値（albedo_scale・stripe・head_front_u）は残し、測った値を上書きする
        ASSETS.mkdir(parents=True, exist_ok=True)
        fur_path.write_text(json.dumps(old | measured, indent=1, ensure_ascii=False) + "\n")
    if args.make:
        fur = json.loads(fur_path.read_text())
        fur["rgba"] = rgba(fur)
        fur_path.write_text(json.dumps(fur, indent=1, ensure_ascii=False) + "\n")
        total = 0
        for name, img in make(fur, args.seed).items():
            cv2.imwrite(str(ASSETS / f"{name}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            size = (ASSETS / f"{name}.png").stat().st_size
            total += size
            print(name, img.shape, size, "bytes")
        print("total", total, "bytes")


if __name__ == "__main__":
    main()
