"""商品の属性（種類・色・柄・テイスト・季節・シルエットなど）を Clef で判定する。

商品の意味検索のための下ごしらえ。商品名・説明・素材などの文字と、商品の写真を Clef に渡す。
色・柄（shopify.color-pattern）と種類（productType）は、すでに入っている値と比べて
正しさを測るため、Clef には渡さない。
"""

import json

# 質問や渡し方を変えたら上げる（保存してある判定を取り直す）
VERSION = 1
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

TASTES = {
    "work": "ワーク（作業着・ワークブランド・ダック地・ペインター・カバーオール）",
    "military": "ミリタリー（軍もの・ミリタリージャケット・カーゴ・迷彩）",
    "outdoor": "アウトドア（山・キャンプ・フリース・マウンテンパーカー・ナイロン）",
    "sports": "スポーツ（スウェット・ジャージ・チームもの・カレッジ）",
    "street": "ストリート（スケート・ヒップホップ・大きなロゴやプリント）",
    "trad": "トラッド・アイビー（ボタンダウン・ブレザー・オックスフォード・ケーブルニット）",
    "western": "ウエスタン（ウエスタンシャツ・デニム・カウボーイ）",
    "dress": "きれいめ（ジャケット・スラックス・シルク・落ち着いた上品な服）",
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
        f"taste_{key}": {"type": "noul", "instructions": f"テイストが当てはまるか: {text}"}
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
