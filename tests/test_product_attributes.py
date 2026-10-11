from app import product_attributes

PRODUCT = {
    "title": "[OSH KOSH] Cotton Flannel Shirts L/S [USED]",
    "productType": "Shirts",
    "tags": ["USED", "XL", "新着"],
    "description": "深みのあるバーガンディのチェック柄。",
    "brand": {"value": "OSH KOSH"},
    "material": {"value": "コットン 100%"},
    "measurements": {"value": '{"着丈":83,"身幅":67}'},
    "colors": {"references": {"nodes": [{"displayName": "チェック柄"}, {"displayName": "レッド"}]}},
}


def test_product_state_leaves_out_colors_and_type():
    state = product_attributes.product_state(PRODUCT)
    assert state["title"] == PRODUCT["title"]
    assert state["brand"] == "OSH KOSH" and state["material"] == "コットン 100%"
    assert state["measurements_cm"] == {"着丈": 83, "身幅": 67}
    assert state["size"] == ["XL"]
    # 種類と色・柄は、正しさを測るために渡さない
    assert set(state) == {
        "task",
        "title",
        "description",
        "brand",
        "material",
        "measurements_cm",
        "size",
    }


def test_expected_maps_existing_values_to_choices():
    exp = product_attributes.expected(PRODUCT)
    assert exp == {"category": {"shirts"}, "color": {"red"}, "pattern": {"check"}}
    # 柄が入っていなければ、無地かプリント
    solid = {**PRODUCT, "colors": {"references": {"nodes": [{"displayName": "ブラック"}]}}}
    assert product_attributes.expected(solid)["pattern"] == {"solid", "print"}
    # 何も入っていなければ比べない
    assert product_attributes.expected({"title": "x"}) == {
        "category": set(),
        "color": set(),
        "pattern": set(),
    }


def test_questions_choices_cover_mapped_names():
    q = product_attributes.QUESTIONS
    assert set(product_attributes.COLOR_NAMES.values()) <= set(q["color"]["criteria"])
    assert set(product_attributes.PATTERN_NAMES.values()) <= set(q["pattern"]["criteria"])
    assert set(product_attributes.CATEGORY_NAMES.values()) <= set(q["category"]["criteria"])


def test_search_text_uses_confident_answers():
    def choice(c, conf=0.9):
        return {"choice": c, "confidence": conf}

    answers = {
        "category": choice("sweaters"),
        "color": choice("navy"),
        "pattern": choice("solid"),
        **{f"taste_{k}": 0.1 for k in product_attributes.TASTES},
        "taste_trad": 0.8,
        "taste_dress": 0.6,
        "season": choice("autumn_winter"),
        "fit": choice("regular"),  # ふつうは書かない
        "gender": choice("mens", 0.4),  # 確からしさが低いものは書かない
        "origin": choice("europe"),
        "thickness": {"score": 1.8, "confidence": 0.7},
    }
    assert product_attributes.search_text(answers) == (
        "ニット・セーター / ネイビー / 無地 / トラッド・アイビー、きれいめ"
        " / 秋冬 / ヨーロッパ / 厚手"
    )
