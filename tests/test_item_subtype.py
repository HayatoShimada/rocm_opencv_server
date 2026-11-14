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


def _pred(photo, text=None, handle="h"):
    def ans(c, conf, aux):
        return {"answers": {"subtype": {"choice": c, "confidence": conf}, **aux}}

    out = {"handle": handle}
    if photo:
        out["photo"] = ans(*photo)
    if text:
        out["text"] = ans(*text)
    return out


def test_candidates_payload_shape_and_filters():
    aux = {"dressy": 1.75, "sleeve_short": 0.01, "napped": {"score": 0.2, "confidence": 0.5}}
    preds = {
        "gid://1": _pred(("dress_shirt", 0.86, aux), handle="a"),
        "gid://2": _pred(("cotton_shirt", 0.49, {})),  # 0.5 未満は送らない
        "gid://3": _pred(None, ("dress_shirt", 0.99, {})),  # photo の判定が無ければ送らない
        "gid://4": {"handle": "d"},  # 判定なし
        "gid://5": _pred(
            ("flannel_shirt", 0.5, {}), ("dress_shirt", 0.9, {})
        ),  # 0.5 ちょうどは送る
    }
    body = item_subtype.candidates_payload(
        preds, ["gid://1", "gid://2", "gid://3", "gid://4", "gid://5", "gid://6"]
    )
    assert body["source"] == "clef" and body["version"] == item_subtype.VERSION
    assert body["items"][0] == {
        "productId": "gid://1",
        "handle": "a",
        "code": "dress_shirt",
        "confidence": 0.86,
        "mode": "photo",
        "aux": {"dressy": 1.75, "sleeve_short": 0.01, "napped": 0.2},
    }
    # photo の判定を使う（text の dress_shirt ではない）
    assert [(i["productId"], i["code"]) for i in body["items"]] == [
        ("gid://1", "dress_shirt"),
        ("gid://5", "flannel_shirt"),
    ]


def test_candidates_only_for_given_products():
    preds = {"gid://1": _pred(("dress_shirt", 0.9, {}))}
    assert item_subtype.candidates_payload(preds, [])["items"] == []


def test_band_counts():
    items = [
        {"productId": "a", "code": "dress_shirt", "confidence": 0.6},
        {"productId": "b", "code": "dress_shirt", "confidence": 0.8},
        {"productId": "c", "code": "dress_shirt", "confidence": 0.95},
    ]
    out = item_subtype.band_counts(items, {"a": "Shirts", "b": "Shirts", "c": "Shirts"})
    assert out == {"Shirts/dress_shirt": {"0.5以上0.8未満": 1, "0.8以上": 2}}
