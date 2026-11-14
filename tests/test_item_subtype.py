import json

import pytest

from app import item_subtype
from scripts import item_subtype_report as report

VOCAB = {
    "Shirts": [
        {"code": "cotton_shirt", "label": "コットンシャツ", "default": True},
        {"code": "dress_shirt", "label": "ドレスシャツ", "default": False},
        {"code": "new_code", "label": "新しい細分類", "default": False},
    ],
    "Hats": [],
}
PRODUCT = {
    "title": "[Brooks Brothers] Oxford BD Shirts L/S",
    "description": "無地のオックスフォード",
    "brand": {"value": "Brooks Brothers"},
    "material": {"value": "コットン 100%"},
    "measurements": {"value": '{"着丈":80}'},
}


def test_questions_use_vocabulary_codes_and_descriptions():
    qs = item_subtype.questions("Shirts", VOCAB)
    criteria = qs[item_subtype.CHOICE]["criteria"]
    assert list(criteria) == ["cotton_shirt", "dress_shirt", "new_code"]
    assert "きれいめ" in criteria["dress_shirt"]
    # 説明の無い code は label を使う
    assert criteria["new_code"] == "新しい細分類"
    assert {"sleeve_short", "dressy", "napped"} <= set(qs)
    # 細分類の無い品目は質問しない
    assert item_subtype.questions("Hats", VOCAB) == {}
    assert item_subtype.questions("Trousers", VOCAB) == {}


def test_every_aux_question_has_type():
    for aux in item_subtype.AUX.values():
        for q in aux.values():
            assert q["type"] in {"noul", "score"}


def test_state_photo_has_no_text_and_never_brand():
    assert item_subtype.state(PRODUCT, "photo") == {"task": item_subtype.TASK}
    text = item_subtype.state(PRODUCT, "text")
    assert text["title"] == PRODUCT["title"] and text["material"] == "コットン 100%"
    assert text["measurements_cm"] == {"着丈": 80}
    assert "brand" not in text
    with pytest.raises(ValueError):
        item_subtype.state(PRODUCT, "both")


def test_summarize():
    out = item_subtype.summarize(
        {
            "subtype": {"choice": "dress_shirt", "confidence": 0.91234},
            "sleeve_short": {"noul": 0.1234},
            "dressy": {"score": 1.567, "confidence": 0.5},
        }
    )
    assert out == {
        "subtype": {"choice": "dress_shirt", "confidence": 0.912},
        "sleeve_short": 0.123,
        "dressy": {"score": 1.57, "confidence": 0.5},
    }


def _row(i, rule, photo, text=None, reason="題名の語", conf="high", brand="Brooks Brothers"):
    return {
        "id": i,
        "product_type": "Shirts",
        "brand": brand,
        "rule_code": rule,
        "rule_reason": reason,
        "rule_confidence": conf,
        "photo": photo,
        "text": text,
    }


def test_report_agreement_and_dress_summary():
    rs = [
        _row(1, "flannel_shirt", ("flannel_shirt", 0.9)),
        _row(2, "flannel_shirt", ("cotton_shirt", 0.5)),
        _row(3, "cotton_shirt", ("dress_shirt", 0.8), reason="既定", conf="medium"),
        _row(4, "cotton_shirt", ("cotton_shirt", 0.7), reason="既定", conf="medium"),
        _row(5, "dress_shirt", ("dress_shirt", 0.95), brand="Lands' End"),
    ]
    assert report.agreement(rs, "photo")["all"] == {"n": 5, "ok": 3, "rate": 0.6}
    tb = report.agreement(rs, "photo", report.title_based)["all"]
    assert (tb["n"], tb["ok"]) == (3, 2)
    d = report.dress_summary(rs, "photo")
    assert d["dress_count"] == 2 and d["rule_cotton_to_dress"] == 1 and d["rule_cotton_kept"] == 1
    assert d["brands_dress"] == {"Brooks Brothers": 1, "Lands' End": 1}
    assert report.confusion(rs, "photo")["Shirts"]["flannel_shirt"] == {
        "flannel_shirt": 1,
        "cotton_shirt": 1,
    }
    assert report.agreement(rs, "text") == {}  # text の判定が無ければ空


def test_rows_joins_predictions_with_guess():
    preds = {
        "g1": {
            "images": ["u"],
            "photo": {"answers": {"subtype": {"choice": "a", "confidence": 0.7}}},
        },
        "g2": {"photo": {}},
    }
    guess = {"g1": {"rule_code": "a", "product_type": "Shirts", "brand": "x"}}
    out = report.rows(preds, guess)
    assert len(out) == 1 and out[0]["photo"] == ("a", 0.7) and out[0]["text"] is None
    json.dumps(report.build(out))  # JSON にできる
