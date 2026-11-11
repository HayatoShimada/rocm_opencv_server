"""着用イメージ（AI の合成）を、服の実寸どおりの大きさで作るための尺度の計算。

85-Store の原則は「タグではなく、測った尺度で服を見せる」。
AI に任せると、服は体に合わせて変形され、オーバーサイズの服もぴったりに見えてしまう。
そこで、服が来るべき範囲（下敷き）を実寸から決めて AI に渡し、できた画像の服の位置を測りなおして、
実寸から決めた位置とのずれが TOLERANCE_CM 以内のものだけを使う。

ここはモデルを使わない計算だけ（tests/test_garment_scale.py で確かめる）。
骨格の点と人・服の形は app/vton.py が出す。

- 人物の尺度: 人の形（マスク）の頭のてっぺんから床までのピクセル数 ÷ 身長（cm）＝ px/cm
- 首の付け根（背中側。着丈を測る起点）: 肩の点の高さより NECK_ABOVE_SHOULDER × 身長 だけ上
  （産総研 AIST 1991-92 の頚椎高 145.6cm と肩峰高 138.0cm の差 7.6cm ÷ 身長 171.4cm）
- 肩の縫い目: 体の中心から 服の肩幅 ÷ 2。体の肩より外に出る分は、腕にそって下へ落とす
  （肩が落ちる服）
- 袖口: 首の付け根から、肩 → ひじ → 手首の線にそって 裄丈（袖丈 ＋ 肩幅 ÷ 2）だけ進んだ所
- 裾: 首の付け根から 着丈 だけ下
- 身幅: 正面から見える幅 ＝ 平置きの身幅 × FRONT_WIDTH_RATIO
- 腕が曲がっている・カメラを向いている写真では、袖口の位置を決めない（notes に理由を残す）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO の 17 点（ViTPose の並び）
NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 0, 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_ANKLE, R_ANKLE = 11, 12, 15, 16

NECK_ABOVE_SHOULDER = 0.044  # 首の付け根（背中）が肩の点より上にある高さ ÷ 身長
# 正面から見える服の幅 ÷ 平置きの身幅。服は体の横にも回り込むので、身幅（周囲の半分）より狭く見える
# （体の幅 ÷ 胸囲の半分 ≒ 0.75。ゆったりした服は少し広がるので 0.78）
FRONT_WIDTH_RATIO = 0.78
# 写真の上の腕（肩先〜手首）が、身長から推定した腕の長さのこれより短ければ、腕が曲がっているか
# カメラの方を向いている（袖口の位置を決められない）。
# 腕 ≒ 身長 × ARM_RATIO（産総研の上腕長＋前腕長）
ARM_RATIO = 0.33
ARM_MIN_RATIO = 0.85
TOLERANCE_CM = 2.0  # 実寸から決めた位置と、できた画像の服の位置のずれの許容（cm）
SHORT_SLEEVE_CM = 40  # これより短い袖丈は半袖


def first(meas: dict, *keys: str) -> float | None:
    for k in keys:
        v = meas.get(k)
        if v is not None and v != "":
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


@dataclass
class PersonScale:
    px_per_cm: float
    top_y: int
    floor_y: int


def person_scale(mask: np.ndarray, height_cm: float) -> PersonScale:
    """人の形のマスク（True が人）の、頭のてっぺんから床までと身長から px/cm を出す。"""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0 or height_cm <= 0:
        raise ValueError("人の形が見つからないか、身長がない")
    top, floor = int(rows[0]), int(rows[-1])
    return PersonScale(px_per_cm=(floor - top + 1) / height_cm, top_y=top, floor_y=floor)


def _point_along(path: list[np.ndarray], distance: float) -> np.ndarray:
    """折れ線の始点から distance（px）進んだ点。線より長ければ、最後の向きにそのまま延ばす。"""
    left = distance
    for a, b in zip(path, path[1:], strict=False):
        seg = float(np.linalg.norm(b - a))
        if seg > 0 and left <= seg:
            return a + (b - a) * (left / seg)
        left -= seg
    a, b = path[-2], path[-1]
    d = b - a
    n = float(np.linalg.norm(d)) or 1.0
    return b + d / n * left


@dataclass
class Layout:
    """服が来るべき位置（画像のピクセル）。"""

    px_per_cm: float
    neck: np.ndarray  # 首の付け根（背中）の点
    hem_y: float | None  # 裾の高さ
    chest_half_px: float | None  # 正面から見た服の幅の半分
    seams: dict[str, np.ndarray] = field(default_factory=dict)  # 'left' / 'right' の肩の縫い目
    cuffs: dict[str, np.ndarray] = field(default_factory=dict)  # 'left' / 'right' の袖口
    arm_paths: dict[str, list[np.ndarray]] = field(default_factory=dict)
    expected: dict[str, float] = field(default_factory=dict)  # 実寸（cm）
    notes: list[str] = field(default_factory=list)  # 決められなかった理由など

    def shapes(self) -> list[list[tuple[float, float]]]:
        """下敷きの形。胴と、両袖（縫い目から袖口までの帯）を別の多角形で返す（重ねて塗る）。"""
        if self.hem_y is None or self.chest_half_px is None:
            raise ValueError("着丈と身幅がないと、下敷きを作れない")
        cx = float(self.neck[0])
        half = self.chest_half_px
        torso_right: list[np.ndarray] = []
        torso_left: list[np.ndarray] = []
        sleeves: list[list[tuple[float, float]]] = []
        for side, seam in self.seams.items():
            sign = 1.0 if seam[0] >= cx else -1.0
            armpit = np.array([cx + sign * half, float(seam[1]) + half * 0.45])
            edge = [seam, armpit, np.array([cx + sign * half, self.hem_y])]
            (torso_right if sign > 0 else torso_left).extend(edge)
            cuff = self.cuffs.get(side)
            if cuff is not None:
                d = cuff - seam
                u = d / (float(np.linalg.norm(d)) or 1.0)
                n = np.array([-u[1], u[0]])
                if np.dot(n, np.array([cx, float(seam[1])]) - seam) < 0:
                    n = -n  # 体の中心の側へ向ける
                w = half * 0.7  # 袖の太さ ≒ 身幅の 0.35 倍 × 2
                sleeves.append(
                    [tuple(map(float, p)) for p in (seam, cuff, cuff + n * w, seam + n * w)]
                )
        neck = (float(self.neck[0]), float(self.neck[1]))
        torso = [neck] + [tuple(map(float, p)) for p in torso_right]
        torso += [tuple(map(float, p)) for p in reversed(torso_left)]
        return [torso, *sleeves]


def layout(
    keypoints: np.ndarray,
    scale: PersonScale,
    height_cm: float,
    meas: dict,
    body_shoulder_px: float | None = None,
) -> Layout:
    """骨格の点（COCO 17 点、画像の座標）と服の実寸から、服が来るべき位置を決める。

    body_shoulder_px: 人の形の、肩の高さでの幅（体の肩幅）。無ければ肩の点の間 × 1.12。
    """
    s = scale.px_per_cm
    ls, rs = keypoints[L_SHOULDER].astype(float), keypoints[R_SHOULDER].astype(float)
    mid = (ls + rs) / 2
    neck = np.array([mid[0], mid[1] - NECK_ABOVE_SHOULDER * height_cm * s])

    shoulder = first(meas, "肩幅")
    sleeve = first(meas, "袖丈")
    yuki = first(meas, "裄丈")
    length = first(meas, "着丈", "着丈(後)")
    chest = first(meas, "身幅")
    expected = {
        k: v
        for k, v in (
            ("肩幅", shoulder),
            ("袖丈", sleeve),
            ("裄丈", yuki),
            ("着丈", length),
            ("身幅", chest),
        )
        if v
    }

    body_half = (body_shoulder_px or float(np.linalg.norm(ls - rs)) * 1.12) / 2
    out = Layout(px_per_cm=s, neck=neck, hem_y=None, chest_half_px=None, expected=expected)
    if length:
        out.hem_y = float(neck[1] + length * s)
    if chest:
        out.chest_half_px = chest * FRONT_WIDTH_RATIO * s / 2

    if yuki is None and sleeve and shoulder and sleeve >= SHORT_SLEEVE_CM:
        yuki = sleeve + shoulder / 2
    for side, (sh, el, wr) in {
        # 画像の左右ではなく、着る人の左右（COCO の left は画像の右側に写る）
        "left": (L_SHOULDER, L_ELBOW, L_WRIST),
        "right": (R_SHOULDER, R_ELBOW, R_WRIST),
    }.items():
        p_sh, p_el, p_wr = (keypoints[i].astype(float) for i in (sh, el, wr))
        direction = np.sign(p_sh[0] - neck[0]) or (1.0 if side == "left" else -1.0)
        # 体の肩先（首の高さの横の線ではなく、肩の点の外側）
        edge = np.array([neck[0] + direction * body_half, p_sh[1]])
        path = [neck, edge, p_el, p_wr]
        out.arm_paths[side] = path
        garment_half_px = (shoulder * s / 2) if shoulder else body_half
        # 肩の縫い目：首から 服の肩幅 ÷ 2 だけ進んだ所（体より広ければ、肩先を越えて腕の上へ落ちる）
        to_edge = float(np.linalg.norm(edge - neck))
        out.seams[side] = _point_along(path, garment_half_px + to_edge - body_half)
        arm_px = float(np.linalg.norm(p_el - edge) + np.linalg.norm(p_wr - p_el))
        arm_ok = arm_px >= ARM_MIN_RATIO * ARM_RATIO * height_cm * s
        if yuki and arm_ok:
            out.cuffs[side] = _point_along(path, yuki * s)
        elif yuki:
            name = "左" if side == "left" else "右"
            out.notes.append(
                f"{name}腕が写真の上で {arm_px / s:.0f}cm しかない"
                "（曲がっているか、カメラを向いている）。"
                "袖口の位置を決められないので、腕をまっすぐ少し開いて撮りなおす"
            )
    return out


@dataclass
class GarmentScale:
    px_per_cm: float
    top: tuple[float, float]  # 服の写真の、首の後ろ（襟の上の中心）の点


def garment_scale(mask: np.ndarray, length_cm: float) -> GarmentScale:
    """ハンガーに掛けた服の写真の形（マスク）と着丈から、服の写真の px/cm を出す。

    中心の列の近く（服の幅の 1/6）で、服のいちばん上（襟）からいちばん下（裾）までを着丈とみなす。
    """
    cols = np.flatnonzero(mask.any(axis=0))
    if cols.size == 0 or length_cm <= 0:
        raise ValueError("服の形が見つからないか、着丈がない")
    cx = (cols[0] + cols[-1]) / 2
    band = max(2, int((cols[-1] - cols[0]) / 12))
    center = mask[:, max(0, int(cx) - band) : int(cx) + band + 1]
    rows = np.flatnonzero(center.any(axis=1))
    top, bottom = int(rows[0]), int(rows[-1])
    return GarmentScale(px_per_cm=(bottom - top + 1) / length_cm, top=(float(cx), float(top)))


def width_scale(mask: np.ndarray, chest_cm: float) -> float | None:
    """服の写真の、脇の下あたり（上から 30〜45%）の胴の幅と身幅から出した px/cm（確かめ用）。

    ハンガーに掛けた服は重さで縦に伸び、横は細く写るので、着丈から出した縮尺とずれる。
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0 or not chest_cm:
        return None
    top, bottom = int(rows[0]), int(rows[-1])
    widths = []
    for y in range(int(top + 0.3 * (bottom - top)), int(top + 0.45 * (bottom - top))):
        row = mask[y]
        xs = np.flatnonzero(row)
        cx = int(xs.mean())
        if not row[cx]:
            continue
        left = right = cx
        while left > 0 and row[left - 1]:
            left -= 1
        while right < row.size - 1 and row[right + 1]:
            right += 1
        widths.append(right - left + 1)
    return float(np.median(widths)) / chest_cm if widths else None


def paste_plan(lay: Layout, g: GarmentScale) -> tuple[float, tuple[int, int]]:
    """服の写真を、人物と同じ縮尺にする倍率と、貼る位置（縮めたあとの左上の座標）。

    服の写真の首の後ろの点を、人物の首の付け根に合わせる。これで着丈（裾の高さ）は実寸どおりになる。
    """
    k = lay.px_per_cm / g.px_per_cm
    x = round(float(lay.neck[0]) - g.top[0] * k)
    y = round(float(lay.neck[1]) - g.top[1] * k)
    return k, (x, y)


@dataclass
class Measured:
    hem_y: float | None
    cuffs: dict[str, np.ndarray]


def measure_garment(garment_mask: np.ndarray, lay: Layout) -> Measured:
    """できた画像の服のマスクから、裾の高さと袖口の位置を測る。

    - 裾: 体の中心の列の近く（身幅の 1/3 の幅）で、服のいちばん下
    - 袖口: 腕の線にそって、服がつながっているいちばん先
    """
    h, w = garment_mask.shape
    cx = int(round(lay.neck[0]))
    band = int(max(2, (lay.chest_half_px or 20) / 3))
    cols = garment_mask[:, max(0, cx - band) : min(w, cx + band + 1)]
    rows = np.flatnonzero(cols.any(axis=1))
    hem = float(rows[-1]) if rows.size else None

    cuffs: dict[str, np.ndarray] = {}
    for side, path in lay.arm_paths.items():
        total = sum(float(np.linalg.norm(b - a)) for a, b in zip(path, path[1:], strict=False))
        step = max(1.0, lay.px_per_cm * 0.5)
        last = None
        d = float(np.linalg.norm(path[1] - path[0]))  # 肩先から先を探す（首から肩先は胴の上の縁）
        miss = 0
        while d <= total * 1.3:
            p = _point_along(path, d)
            x, y = int(round(p[0])), int(round(p[1]))
            inside = 0 <= x < w and 0 <= y < h and bool(garment_mask[y, x])
            if inside:
                last, miss = p, 0
            elif last is not None:
                miss += 1
                if miss * step > 3 * lay.px_per_cm:  # 3cm 続けて外なら、そこまでが袖
                    break
            d += step
        if last is not None:
            cuffs[side] = last
    return Measured(hem_y=hem, cuffs=cuffs)


def _along(path: list[np.ndarray], p: np.ndarray) -> float:
    """点 p を折れ線に落とした所の、始点からの線にそった距離（px）。"""
    best, acc, best_d = 0.0, 0.0, float("inf")
    for a, b in zip(path, path[1:], strict=False):
        seg = b - a
        n = float(np.dot(seg, seg)) or 1.0
        t = float(np.clip(np.dot(p - a, seg) / n, 0, 1))
        dist = float(np.linalg.norm(p - (a + seg * t)))
        if dist < best_d:
            best_d, best = dist, acc + t * float(np.sqrt(n))
        acc += float(np.sqrt(n))
    return best


def deviations(lay: Layout, got: Measured) -> dict[str, float]:
    """実寸から決めた位置と、測った位置のずれ（cm。正は長い・遠い）。"""
    s = lay.px_per_cm
    out: dict[str, float] = {}
    if lay.hem_y is not None and got.hem_y is not None:
        out["着丈"] = round((got.hem_y - lay.hem_y) / s, 1)
    for side, want in lay.cuffs.items():
        have = got.cuffs.get(side)
        if have is None:
            continue
        name = "袖口（左）" if side == "left" else "袖口（右）"
        path = lay.arm_paths[side]
        out[name] = round((_along(path, have) - _along(path, want)) / s, 1)
    return out


def passes(dev: dict[str, float], tolerance: float = TOLERANCE_CM) -> bool:
    return bool(dev) and all(abs(v) <= tolerance for v in dev.values())


def caption_numbers(person: dict, lay: Layout, dev: dict[str, float]) -> dict:
    """説明文に入れる数字（人物の身長・肩幅、服の実寸、いちばん大きいずれ）。"""
    return {
        "person_height": person.get("height"),
        "person_shoulder": person.get("shoulder"),
        "garment": {k: v for k, v in lay.expected.items()},
        "max_deviation_cm": max((abs(v) for v in dev.values()), default=None),
        "tolerance_cm": TOLERANCE_CM,
    }
