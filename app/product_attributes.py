"""商品の属性（種類・色・柄・テイスト・季節・シルエットなど）を Clef で判定する。

商品の意味検索のための下ごしらえ。商品名・説明・素材などの文字と、商品の写真を Clef に渡す。
色・柄（shopify.color-pattern）と種類（productType）は、すでに入っている値と比べて
正しさを測るため、Clef には渡さない。
"""

import json

# 質問や渡し方を変えたら上げる（保存してある判定を取り直す）
VERSION = 2
TASK = "古着・セレクトショップの商品の属性を判定する（検索と絞り込みに使う）"
# 説明の長さの上限（文字）。長い説明はここで切る
DESCRIPTION_LIMIT = 800

PRODUCT_QUERY = """
query($after: String, $query: String, $first: Int!) {
  products(first: $first, after: $after, query: $query, sortKey: CREATED_AT, reverse: true) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id handle title productType tags status
      description(truncateAt: DESCRIPTION_LIMIT)
      brand: metafield(namespace: "custom", key: "brand") { value }
      material: metafield(namespace: "custom", key: "material") { value }
      measurements: metafield(namespace: "custom", key: "measurements") { value }
      colors: metafield(namespace: "shopify", key: "color-pattern") {
        references(first: 10) { nodes { ... on Metaobject { displayName } } }
      }
      media(first: 20) {
        nodes { ... on MediaImage { id image { url } } }
      }
    }
  }
}
""".replace("DESCRIPTION_LIMIT", str(DESCRIPTION_LIMIT))

# 括弧の前がテイストの名前（レポートに出す）。括弧の中に、含めないものも書く
# （v1 では、アウトドアの服にミリタリー・ワークが、国内ブランドの普段着にきれいめが付きすぎた）
TASTES = {
    "work": (
        "ワーク（作業着・ユニフォーム・Carhartt や Dickies などのワークブランド・ダック地・"
        "ペインターパンツ・カバーオール。山やキャンプ向けの服は含めない）"
    ),
    "military": (
        "ミリタリー（軍の放出品や軍の仕様の服。M-65・ファティーグ・カーゴパンツ・迷彩。"
        "軍と関係のないアウトドアやワークの服は含めない）"
    ),
    "outdoor": (
        "アウトドア（山・キャンプ・釣り向けの服。フリース・マウンテンパーカー・"
        "ナイロンジャケット・フィッシングベスト・アウトドアブランド）"
    ),
    "sports": "スポーツ（スウェット・ジャージ・チームやカレッジのもの・スポーツブランド）",
    "street": "ストリート（スケート・ヒップホップ・グラフィックの大きなプリントやロゴ）",
    "trad": (
        "トラッド・アイビー（ボタンダウン・ブレザー・チノ・オックスフォード・"
        "ケーブルやフェアアイルのニット・Brooks Brothers や Ralph Lauren）"
    ),
    "western": "ウエスタン（ウエスタンシャツ・ヨーク・スナップボタン・カウボーイ）",
    "dress": (
        "きれいめ（テーラードジャケット・スラックス・シルク・カシミヤなど、"
        "仕事や改まった場にも着られる上品な服。ショートパンツ・イージーパンツ・"
        "スウェット・Tシャツなどの普段着は含めない）"
    ),
    "minimal": "シンプル・ミニマル（無地で飾りの少ない、今のデザインの服。国内ブランドの新品など）",
}

QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "商品の種類",
        "criteria": {
            "shirts": "シャツ（長袖・半袖。ボタンやジップで前が開く）",
            "t_shirts": "Tシャツ・カットソー",
            "polos": "ポロシャツ",
            "sweatshirts": "スウェット・パーカー",
            "sweaters": "ニット・セーター（前が開かない）",
            "cardigans": "カーディガン",
            "coats_jackets": "コート・ジャケット・ブルゾン",
            "vests": "ベスト",
            "pants": "パンツ（長ズボン）",
            "shorts": "ショートパンツ",
            "hats": "帽子",
            "shoes": "靴・サンダル",
            "goods": "小物・雑貨・本",
        },
    },
    "color": {
        "type": "choice",
        "instructions": "いちばん広い面積を占める色（地の色）",
        "criteria": {
            "black": "ブラック",
            "white": "ホワイト・生成り・アイボリー",
            "gray": "グレー・チャコール",
            "navy": "ネイビー（紺）",
            "blue": "ブルー・水色・サックス",
            "green": "グリーン・カーキ・オリーブ",
            "beige": "ベージュ・タン・キャメル",
            "brown": "ブラウン（茶）",
            "red": "レッド・バーガンディ・ワイン",
            "pink": "ピンク",
            "orange": "オレンジ",
            "yellow": "イエロー・マスタード",
            "purple": "パープル",
        },
    },
    "pattern": {
        "type": "choice",
        "instructions": "柄",
        "criteria": {
            "solid": "無地",
            "check": "チェック・格子柄",
            "stripe": "ストライプ・ボーダー",
            "print": "ロゴ・プリント・刺繍（地は無地）",
            "other": "花柄・幾何学・迷彩・総柄など",
        },
    },
    **{
        f"taste_{key}": {
            "type": "noul",
            "instructions": f"このテイストにはっきり当てはまるか: {text}",
        }
        for key, text in TASTES.items()
    },
    "season": {
        "type": "choice",
        "instructions": "主に着る季節",
        "criteria": {
            "spring_summer": "春夏（薄手・半袖・涼しい素材）",
            "autumn_winter": "秋冬（厚手・ウール・フリース・中わた・起毛）",
            "all_season": "通年（季節を問わない）",
        },
    },
    "thickness": {"type": "score", "criteria": ["薄手", "ふつう", "厚手"]},
    "fit": {
        "type": "choice",
        "instructions": "シルエット（実寸と写真から）",
        "criteria": {
            "slim": "細身・タイト",
            "regular": "ふつう",
            "relaxed": "ゆったり・オーバーサイズ",
        },
    },
    "gender": {
        "type": "choice",
        "instructions": "主に想定している着る人",
        "criteria": {
            "mens": "メンズ",
            "womens": "レディース",
            "unisex": "どちらでも",
        },
    },
    # 「ユーロ古着」「アメリカ古着」で探されるため
    "origin": {
        "type": "choice",
        "instructions": "ブランドや服の系統",
        "criteria": {
            "america": "アメリカ（アメリカのブランド・アメリカ古着）",
            "europe": "ヨーロッパ（ヨーロッパのブランド・軍もの・ユーロ古着）",
            "japan": "日本（日本のブランド・日本製）",
            "unknown": "わからない",
        },
    },
}


def product_state(product: dict) -> dict:
    """Clef の state にする商品の情報。色・柄と種類は渡さない（正しさを測るため）。"""
    state = {
        "task": TASK,
        "title": product["title"],
        "description": product.get("description") or "",
    }
    for key in ("brand", "material"):
        if product.get(key):
            state[key] = product[key]["value"]
    if product.get("measurements"):
        state["measurements_cm"] = json.loads(product["measurements"]["value"])
    sizes = [t for t in product.get("tags", []) if t in SIZE_TAGS]
    if sizes:
        state["size"] = sizes
    return state


SIZE_TAGS = {"XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL", "FREE"}


def existing_colors(product: dict) -> list[str]:
    """すでに入っている色・柄（Shopify の標準の名前）。"""
    colors = product.get("colors") or {}
    return [n["displayName"] for n in colors.get("references", {}).get("nodes", []) if n]


# Shopify の色・柄の名前を、Clef の選択肢に合わせる（比べるため）
COLOR_NAMES = {
    "ブラック": "black",
    "ホワイト": "white",
    "グレー": "gray",
    "シルバー": "gray",
    "ネイビーブルー": "navy",
    "ブルー": "blue",
    "ターコイズ": "blue",
    "グリーン": "green",
    "ベージュ": "beige",
    "ブラウン": "brown",
    "レッド": "red",
    "ピンク": "pink",
    "ローズゴールド": "pink",
    "オレンジ": "orange",
    "イエロー": "yellow",
    "パープル": "purple",
}
PATTERN_NAMES = {
    "チェック柄": "check",
    "プレード": "check",
    "ストライプ": "stripe",
    "ジオメトリック": "other",
    "フローラル": "other",
    "動物": "other",
    "カモフラージュ": "other",
}
# productType を Clef の種類の選択肢に合わせる
CATEGORY_NAMES = {
    "Shirts": "shirts",
    "T-Shirts": "t_shirts",
    "Polos": "polos",
    "Sweatshirts": "sweatshirts",
    "Sweaters": "sweaters",
    "Cardigans": "cardigans",
    "Coats & Jackets": "coats_jackets",
    "Parkas": "coats_jackets",
    "Vests": "vests",
    "Trousers": "pants",
    "Pants": "pants",
    "Short Trousers": "shorts",
    "Hats": "hats",
    "Sandals": "shoes",
    "Goods": "goods",
    "Print Books": "goods",
}


def expected(product: dict) -> dict:
    """すでに入っている値から、Clef の答えとして正しいもの（の候補）を出す。

    色・柄は複数入っていることがあるので、そのどれかに当たれば正しいとする。
    柄が入っていないものは、無地かプリントのどちらか（区別できない）。
    """
    names = existing_colors(product)
    colors = {COLOR_NAMES[n] for n in names if n in COLOR_NAMES}
    patterns = {PATTERN_NAMES[n] for n in names if n in PATTERN_NAMES}
    category = CATEGORY_NAMES.get(product.get("productType") or "")
    return {
        "category": {category} if category else set(),
        "color": colors,
        "pattern": patterns or ({"solid", "print"} if names else set()),
    }


# 検索に使う属性の文（AI 検索の Worker が、商品の説明に足して Clef に渡す）。
# 確からしさがこれ以上の答えだけを使う
SEARCH_THRESHOLD = 0.5
THICKNESS_LABELS = ["薄手", "ふつうの厚さ", "厚手"]
SEARCH_SKIP = {("pattern", "other"), ("origin", "unknown"), ("fit", "regular")}


def _label(qid: str, choice: str) -> str:
    return QUESTIONS[qid]["criteria"][choice].split("（")[0]


def search_text(answers: dict) -> str:
    """判定（scripts/shopify_product_attributes.py の answers）を、検索に使う短い文にする。

    例: 「ニット・セーター / ネイビー / 無地 / トラッド・アイビー / 秋冬 / ヨーロッパ / 厚手」
    """
    parts = []
    for qid in ("category", "color", "pattern"):
        a = answers[qid]
        if a["confidence"] >= SEARCH_THRESHOLD and (qid, a["choice"]) not in SEARCH_SKIP:
            parts.append(_label(qid, a["choice"]))
    tastes = [TASTES[k].split("（")[0] for k in TASTES if answers[f"taste_{k}"] >= SEARCH_THRESHOLD]
    if tastes:
        parts.append("、".join(tastes))
    for qid in ("season", "fit", "gender", "origin"):
        a = answers[qid]
        if a["confidence"] >= SEARCH_THRESHOLD and (qid, a["choice"]) not in SEARCH_SKIP:
            parts.append(_label(qid, a["choice"]))
    thickness = answers["thickness"]["score"]
    parts.append(THICKNESS_LABELS[min(2, max(0, round(thickness)))])
    return " / ".join(parts)
