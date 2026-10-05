import numpy as np

from app import whitebalance as wb


def _photo(wall_rgb, size=(2048, 1365)) -> np.ndarray:
    """壁の前に紺の服を掛けた写真のつもりの合成画像（BGR）。"""
    height, width = size
    image = np.empty((height, width, 3), np.uint8)
    image[:] = wall_rgb[::-1]
    garment = (
        slice(int(height * 0.2), int(height * 0.65)),
        slice(int(width * 0.2), int(width * 0.8)),
    )
    image[garment] = (60, 35, 25)
    return image


def _region_mean(image, region):
    height, width = image.shape[:2]
    x0, x1, y0, y1 = region
    crop = image[int(height * y0) : int(height * y1), int(width * x0) : int(width * x1)]
    b, g, r = crop.reshape(-1, 3).mean(axis=0)
    return np.array([r, g, b])


def test_tinted_wall_becomes_target():
    image = _photo((190, 200, 220))
    correction = wb.plan(image)
    assert correction.status == "apply"
    corrected = wb.apply_gains(image, correction.gains)
    assert np.abs(_region_mean(corrected, wb.REGIONS[0]) - wb.TARGET_RGB).max() < 1.5
    # もう一度測ると補正不要になる
    assert wb.plan(corrected).status == "ok"


def test_close_to_target_is_left_alone():
    assert wb.plan(_photo((206, 207, 207))).status == "ok"


def test_uses_next_region_when_garment_covers_reference():
    image = _photo((190, 200, 220))
    height, width = image.shape[:2]
    x0, x1, y0, y1 = wb.REGIONS[0]
    image[int(height * y0) : int(height * y1), int(width * x0) : int(width * x1)] = (60, 35, 25)
    image[int(height * y0) : int(height * y0) + 5, int(width * x0) :] = (255, 255, 255)
    correction = wb.plan(image)
    assert correction.status == "apply"
    assert correction.region != wb.REGIONS[0]


def test_review_when_no_wall():
    image = np.random.default_rng(0).integers(0, 255, (2048, 1365, 3), dtype=np.uint8)
    assert wb.plan(image).status == "review"


def test_dark_neutral_wall_is_left_alone():
    # 暗いが色かぶりのない壁は、明るさまで合わせると大きすぎるので、変えない
    correction = wb.plan(_photo((150, 150, 150)))
    assert correction.status == "ok"


def test_highlights_keep_color_ratio():
    image = np.full((10, 10, 3), (230, 220, 210), np.uint8)
    out = wb.apply_gains(image, (1.3, 1.3, 1.3))
    assert out.max() == 255
    # 比率を保って縮めるので、チャンネルの大小は変わらない
    assert out[0, 0, 0] >= out[0, 0, 1] >= out[0, 0, 2]


def test_white_garment_stays_neutral():
    # 青い壁の写真の白い服（青だけ白飛び。BGR）は、補正で黄色くならない
    image = np.full((10, 10, 3), (255, 233, 222), np.uint8)
    out = wb.apply_gains(image, (1.3, 1.237, 0.989))
    assert int(out.max()) - int(out.min()) <= 1


def test_colored_garment_keeps_color():
    # オレンジの服（赤が白飛び）は無彩色にしない
    image = np.full((10, 10, 3), (30, 120, 255), np.uint8)
    out = wb.apply_gains(image, (1.1, 1.0, 0.95))
    assert int(out[0, 0, 2]) - int(out[0, 0, 0]) > 150


def test_dark_uniform_garment_is_not_wall():
    # 範囲がすべて暗く均一な服（紺のニットのアップ）なら、壁とみなさない
    image = np.full((2048, 1365, 3), (35, 21, 14), np.uint8)
    assert wb.plan(image).status == "review"


def test_bright_bluish_wall_fixes_cast_only():
    # 暗い青っぽい壁: 明るさまで合わせると大きすぎるので、色かぶりだけ直す
    image = _photo((175, 178, 200))
    correction = wb.plan(image)
    assert correction.status == "apply"
    assert "色かぶり" in correction.reason
    after = _region_mean(wb.apply_gains(image, correction.gains), wb.REGIONS[0])
    assert after.max() - after.min() <= 3  # ほぼ無彩色（基準の壁がわずかに青寄り）
    assert abs(after.mean() - np.array([175, 178, 200]).mean()) < 6  # 明るさはほぼそのまま


def test_bright_neutral_wall_is_left_alone():
    # 真っ白で色かぶりのない壁は変えない（以前の「白い壁には触らない」）
    assert wb.plan(_photo((248, 248, 248))).status == "ok"


def test_white_wall_is_left_alone():
    # 白い壁は、青みがあっても触らない
    correction = wb.plan(_photo((225, 231, 247)))
    assert correction.status == "ok"
    assert "白い壁" in correction.reason
