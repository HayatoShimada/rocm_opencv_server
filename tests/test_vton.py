"""着用イメージの PoC のスクリプト（モデルを読まない部分）。"""

import json

import numpy as np

from scripts import vton_contact_sheet, vton_tryon


def test_front_image_prefers_front_alt():
    product = {
        "media": {
            "nodes": [
                {"alt": "シャツ（タグのアップ）", "image": {"url": "a"}},
                {"alt": "シャツ（正面）", "image": {"url": "b"}},
                {"alt": None},
            ]
        }
    }
    assert vton_tryon.front_image(product)["image"]["url"] == "b"
    product["media"]["nodes"][1]["alt"] = "シャツ（背面）"
    assert vton_tryon.front_image(product)["image"]["url"] == "a"


def test_body_width_at_counts_the_run_with_center():
    mask = np.zeros((10, 20), dtype=bool)
    mask[5, 2:4] = True  # 腕（離れている）
    mask[5, 6:15] = True  # 胴
    assert vton_tryon.body_width_at(mask, 5, 10) == 9
    assert vton_tryon.body_width_at(mask, 5, 5) is None


def test_load_people_requires_height(tmp_path):
    path = tmp_path / "people.json"
    path.write_text(json.dumps({"a": {"file": "a.jpg", "height": 170}}))
    people = vton_tryon.load_people(path)
    assert people["a"]["path"].endswith("a.jpg")
    path.write_text(json.dumps({"b": {"file": "b.jpg"}}))
    try:
        vton_tryon.load_people(path)
    except SystemExit as e:
        assert "身長" in str(e)
    else:
        raise AssertionError("身長がないのに通った")


def test_caption_uses_measured_numbers():
    job = {
        "check": {
            "passes": True,
            "caption": {
                "person_height": 170,
                "garment": {"肩幅": 48, "着丈": 74.0},
                "tolerance_cm": 2.0,
            },
        }
    }
    text = vton_contact_sheet.caption(job)
    assert "身長 170cm" in text and "肩幅 48・着丈 74cm" in text and "2cm 以内" in text


def test_largest_blob_drops_the_hanger_rod():
    mask = np.zeros((200, 200), dtype=bool)
    mask[10:12, :] = True  # 掛けるパイプ（細い線）
    mask[40:190, 50:150] = True  # 服
    out = vton_tryon.largest_blob(mask, k=5)
    assert not out[10:12].any()
    assert out[60, 100] and out.sum() > 0.9 * 150 * 100
