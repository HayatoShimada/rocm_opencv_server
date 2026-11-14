"""商品の「細分類」（ドレスシャツ・ネルシャツ・フリースなど）を、写真の雰囲気から Clef で判定する。

細分類の語彙の正は CRM（商品タイプ → [{code, label, default}]）。ここには固定せず、
読み込んだ語彙に対して質問を組み立てる。code は変えてはいけない約束の値。
ここにあるのは、見た目で分かる言葉の説明（DESCRIPTIONS）と、判定の根拠を後で見るための補助の質問だけ。

ドレスシャツかどうかはブランドでは決めない。写真だけ（photo）と、題名・説明・素材なども渡す条件（text）の
両方で判定でき、題名の語を読んでいるだけなのか、見ているのかを比べられる。
どちらの条件でも、ブランド名は渡さない（題名に入っているものは消さない）。
"""

import json
from pathlib import Path

# 質問や渡し方を変えたら上げる（保存してある判定を取り直す）。product_attributes.VERSION とは別
VERSION = 1
MODES = ("photo", "text")
TASK = "古着・セレクトショップの商品を、見た目から細分類する"
# 判定の確からしさがこれ未満は「境目」とする
BORDERLINE = 0.6

# 見た目で分かる言葉の説明。語彙に無い code は label だけを使う
DESCRIPTIONS = {
    "short_sleeve_shirt": "半袖のシャツ（襟付き・開襟など。柄は問わない）",
    "cotton_shirt": (
        "カジュアルな長袖の綿シャツ。ワーク・ウエスタン・柄物など、ドレスにもネルにも当たらないもの"
    ),
    "dress_shirt": (
        "襟付きのきれいめな長袖シャツ。無地・細いストライプ・小さなチェック。"
        "ブロードやオックスフォードのなめらかな生地。仕事やジャケットの下に着られる"
    ),
    "flannel_shirt": "起毛した厚手の生地の長袖シャツ。大きめのチェックが多い",
    "wool_shirt": "ウールの厚手の長袖シャツ。ざらっとした毛羽のある生地。無地や霜降りが多い",
    "corduroy_shirt": "コーデュロイ（縦に畝のある生地）の長袖シャツ",
    "linen_shirt": "リネン（麻）のシャツ。しわ感があり、薄くて軽い",
    "light_jacket": "薄手から中くらいのジャケット・ブルゾン（コーチ・デニム・ミリタリーなど）",
    "blazer": "テーラードジャケット・ブレザー。襟とラペルがあり、きれいめな形",
    "fleece": "フリース（もこもこした起毛の生地のジャケット・プルオーバー）",
    "coat": "丈の長いコート（ステンカラー・トレンチ・ウール・ピーコートなど）",
    "down_jacket": "ダウンや中綿が入って膨らんだジャケット・ベスト（キルティング含む）",
    "thin_knit": "薄手のニット。編み目が細かく、体の線が出る。春秋に着る",
    "thick_knit": "厚手のニット。ざっくりした編み目・ケーブル編み・ウール。冬に着る",
    "sweatshirt": "フードの無いスウェット（トレーナー・カレッジロゴなど）",
    "hoodie": "フード付きのスウェット（パーカー）",
}

# 判定の根拠を見るための補助の質問（品目ごと）。score は criteria の並びの順に 0 から数える
AUX = {
    "Shirts": {
        "sleeve_short": {"type": "noul", "instructions": "半袖のシャツか（長袖・七分は no）"},
        "dressy": {
            "type": "score",
            "criteria": ["カジュアル（ワーク・柄物）", "ふつう", "きれいめ（ドレス・仕事向き）"],
        },
        "napped": {
            "type": "score",
            "criteria": ["なめらか・薄手の生地", "ふつう", "起毛・厚手の生地"],
        },
    },
    "Coats & Jackets": {
        "padded": {
            "type": "noul",
            "instructions": "ダウン・中綿・キルティングが入って膨らんでいるか",
        },
        "tailored": {
            "type": "noul",
            "instructions": "襟とラペルのあるテーラードな形（ブレザー・スーツのジャケット）か",
        },
        "thickness": {"type": "score", "criteria": ["薄手", "ふつう", "厚手"]},
    },
    "Sweaters": {
        "thickness": {"type": "score", "criteria": ["薄手", "ふつう", "厚手"]},
        "chunky": {
            "type": "noul",
            "instructions": "太い糸のざっくりした編み目（ケーブル編みなど）か",
        },
    },
    "Sweatshirts": {
        "hood": {"type": "noul", "instructions": "フードが付いているか"},
    },
}
CHOICE = "subtype"


def load_vocabulary(path: Path) -> dict[str, list[dict]]:
    """CRM の細分類の語彙（商品タイプ → [{code, label, default}]）。"""
    return json.loads(Path(path).read_text())


def questions(product_type: str, vocabulary: dict[str, list[dict]]) -> dict:
    """この品目の細分類を選ぶ質問と、補助の質問。細分類のない品目は空。"""
    entries = vocabulary.get(product_type)
    if not entries:
        return {}
    criteria = {e["code"]: DESCRIPTIONS.get(e["code"], e["label"]) for e in entries}
    return {
        CHOICE: {
            "type": "choice",
            "instructions": f"この{product_type}の細分類。写真の見た目（襟・生地・柄・厚み）で選ぶ",
            "criteria": criteria,
        },
        **AUX.get(product_type, {}),
    }


def state(product: dict, mode: str) -> dict:
    """Clef の state。photo は task だけ（写真で決めさせる）、text は題名・説明・素材・採寸も渡す。
    ブランドは、どちらでも渡さない。"""
    if mode not in MODES:
        raise ValueError(f"mode は {MODES} のどちらか: {mode}")
    out = {"task": TASK}
    if mode == "photo":
        return out
    out["title"] = product["title"]
    out["description"] = product.get("description") or ""
    if product.get("material"):
        out["material"] = product["material"]["value"]
    if product.get("measurements"):
        out["measurements_cm"] = json.loads(product["measurements"]["value"])
    return out


def summarize(answers: dict) -> dict:
    """答えから保存するものだけ（noul=確率、choice=選択と確からしさ、score=値）。"""
    out = {}
    for qid, a in answers.items():
        if "noul" in a:
            out[qid] = round(a["noul"], 3)
        elif "choice" in a:
            out[qid] = {"choice": a["choice"], "confidence": round(a["confidence"], 3)}
        else:
            out[qid] = {"score": round(a["score"], 2), "confidence": round(a["confidence"], 3)}
    return out


# CMS に候補として送る確からしさの下限（0.5 以上は全部送る。自動か人が確かめるかは CMS が決める）
MIN_CONFIDENCE = 0.5
# 帯（確認だけの出力用）。下限, 上限
BANDS = ((MIN_CONFIDENCE, 0.8), (0.8, 1.01))
SEND_MODE = "photo"


def candidate_item(product_id: str, pred: dict) -> dict | None:
    """送る1件。写真だけ（photo）の判定を使い、確からしさが下限未満・判定なしなら None。"""
    judged = (pred.get(SEND_MODE) or {}).get("answers") or {}
    top = judged.get(CHOICE)
    if not top or top["confidence"] < MIN_CONFIDENCE:
        return None
    aux = {
        qid: (a["score"] if isinstance(a, dict) else a)
        for qid, a in judged.items()
        if qid != CHOICE
    }
    return {
        "productId": product_id,
        "handle": pred["handle"],
        "code": top["choice"],
        "confidence": top["confidence"],
        "mode": SEND_MODE,
        "aux": aux,
    }


def candidates_payload(predictions: dict, product_ids) -> dict:
    """CMS の POST /api/products/item-subtype-candidates の本文。形が変わったらここだけ直す。"""
    items = [
        item
        for pid in product_ids
        if pid in predictions and (item := candidate_item(pid, predictions[pid]))
    ]
    return {"source": "clef", "version": VERSION, "items": items}


def band_counts(items: list[dict], product_types: dict[str, str]) -> dict:
    """細分類ごとの、確からしさの帯ごとの件数。product_types は productId → 品目。"""
    out: dict = {}
    for it in items:
        key = f"{product_types.get(it['productId'], '?')}/{it['code']}"
        for lo, hi in BANDS:
            if lo <= it["confidence"] < hi:
                label = f"{lo}以上{hi}未満" if hi < 1 else f"{lo}以上"
                out.setdefault(key, {})[label] = out.get(key, {}).get(label, 0) + 1
    return out
