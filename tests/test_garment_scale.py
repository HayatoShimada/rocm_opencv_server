"""着用イメージの尺度の計算（app/garment_scale.py）。モデルを使わず、大きさの分かる合成の図で確かめる。"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app import garment_scale as gs

# 10 px/cm の人物（身長 175cm → 1750px）。肩の点は y=400、ひじ y=700、手首 y=950
H_CM = 175


def _person() -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((2000, 1200), dtype=bool)
    mask[100:1850, 450:750] = True
    kp = np.zeros((17, 2))
    kp[gs.L_SHOULDER] = (700, 400)
    kp[gs.R_SHOULDER] = (500, 400)
    kp[gs.L_ELBOW] = (700, 700)
    kp[gs.R_ELBOW] = (500, 700)
    kp[gs.L_WRIST] = (700, 950)
    kp[gs.R_WRIST] = (500, 950)
    return mask, kp


def _layout(meas: dict) -> gs.Layout:
    mask, kp = _person()
    scale = gs.person_scale(mask, H_CM)
    return gs.layout(kp, scale, H_CM, meas, body_shoulder_px=400)


def _fill(lay: gs.Layout) -> np.ndarray:
    img = Image.new("L", (1200, 2000), 0)
    draw = ImageDraw.Draw(img)
    for shape in lay.shapes():
        draw.polygon(shape, fill=255)
    return np.array(img) > 0


def test_person_scale():
    mask, _ = _person()
    s = gs.person_scale(mask, H_CM)
    assert s.px_per_cm == pytest.approx(10.0)
    assert (s.top_y, s.floor_y) == (100, 1849)


def test_layout_positions_from_measurements():
    lay = _layout({"肩幅": 40, "袖丈": 60, "着丈": 70, "身幅": 50})
    # 首の付け根は肩の点より 0.044 × 175cm = 7.7cm 上
    assert lay.neck[1] == pytest.approx(400 - 77)
    # 裾は首の付け根から着丈 70cm 下
    assert lay.hem_y == pytest.approx(323 + 700)
    # 服の肩幅が体と同じなら、縫い目は体の肩先
    assert lay.seams["left"] == pytest.approx(np.array([800, 400]), abs=0.5)
    # 袖口は首から裄丈（60 ＋ 40 ÷ 2 ＝ 80cm）。手首（530.5 ＋ 250px）を 19.5px 越える
    assert lay.cuffs["left"] == pytest.approx(np.array([700, 969.5]), abs=1)
    assert lay.chest_half_px == pytest.approx(50 * gs.FRONT_WIDTH_RATIO * 10 / 2)


def test_dropped_shoulder_moves_seam_down_the_arm():
    lay = _layout({"肩幅": 50, "袖丈": 55, "着丈": 70, "身幅": 60})
    seam = lay.seams["left"]
    # 体より 10cm 広い → 片側 5cm（50px）肩先を越えて、ひじへ向かう線の上
    edge, elbow = np.array([800, 400]), np.array([700, 700])
    assert np.linalg.norm(seam - edge) == pytest.approx(50, abs=0.5)
    a, b = elbow - edge, seam - edge
    assert a[0] * b[1] - a[1] * b[0] == pytest.approx(0, abs=1)


def test_measure_matches_layout_when_filled_exactly():
    lay = _layout({"肩幅": 44, "袖丈": 60, "着丈": 72, "身幅": 55})
    got = gs.measure_garment(_fill(lay), lay)
    dev = gs.deviations(lay, got)
    assert set(dev) == {"着丈", "袖口（左）", "袖口（右）"}
    assert all(abs(v) <= 1.0 for v in dev.values()), dev
    assert gs.passes(dev)


def test_short_hem_fails_the_check():
    lay = _layout({"肩幅": 44, "袖丈": 60, "着丈": 72, "身幅": 55})
    mask = _fill(lay)
    mask[int(lay.hem_y) - 30 :, :] = False  # 裾が 3cm 短い
    dev = gs.deviations(lay, gs.measure_garment(mask, lay))
    assert dev["着丈"] == pytest.approx(-3.0, abs=0.3)
    assert not gs.passes(dev)


def test_short_sleeve_has_no_cuff():
    lay = _layout({"肩幅": 46, "袖丈": 22, "着丈": 70, "身幅": 54})
    assert lay.cuffs == {}
    assert len(lay.shapes()) == 1  # 胴だけ


def test_bent_arm_has_no_cuff_and_a_note():
    mask, kp = _person()
    kp[gs.L_WRIST] = (650, 760)  # 前腕がカメラの方を向いて、短く写る
    lay = gs.layout(
        kp, gs.person_scale(mask, H_CM), H_CM, {"肩幅": 44, "袖丈": 60, "着丈": 72, "身幅": 55}, 400
    )
    assert "left" not in lay.cuffs and "right" in lay.cuffs
    assert lay.notes and "左腕" in lay.notes[0]


def test_garment_scale_and_paste_plan_put_the_hem_at_the_measured_length():
    # ハンガーの写真：服は 300px 幅・720px 高さ（着丈 72cm → 10 px/cm）。首の後ろは (250, 100)
    gmask = np.zeros((1000, 500), dtype=bool)
    gmask[100:820, 100:400] = True
    g = gs.garment_scale(gmask, 72)
    assert g.px_per_cm == pytest.approx(10.0)
    assert g.top == pytest.approx((249.5, 100))
    lay = _layout({"肩幅": 44, "袖丈": 60, "着丈": 72, "身幅": 55})
    k, (x, y) = gs.paste_plan(lay, g)
    # 人物も 10 px/cm なので倍率 1。裾（820）は 首の付け根 ＋ 720px に来る
    assert k == pytest.approx(1.0)
    assert y + 820 * k == pytest.approx(lay.hem_y, abs=1)
