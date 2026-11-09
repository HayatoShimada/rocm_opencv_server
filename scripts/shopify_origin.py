"""商品の原産国を、タグのアップの写真から Clef で判定し、Shopify の「原産国」に入れる。

入れる先は在庫アイテムの countryCodeOfOrigin。米国への関税の前払い（DDP）と、
海外の税関の申告のため（85store-cms の scripts/fill-hs-codes.ts と組で使う）。

  uv run --env-file .env --group clip --group clef-local \\
      python -m scripts.shopify_origin [--product <ハンドル>]... [--apply]

- タグのアップの写真は、scripts/shopify_classify_photos.py の判定
  （data/photo-labels/predictions.json）から選ぶ
  （アップ・part=tag。確信度の高い順に、商品ごとに最大3枚）
- Clef は文字を書き出さないので、「原産国の表記が読めるか」と「どの国か（一覧から選ぶ）」を聞く
- CMS の「原産国」（custom.made_in）が入っている商品は、そちらを優先する
- 既定は確認だけ（結果を data/origin/results.json に保存）。
  --apply で、確信度の高いものだけ Shopify に入れる（--from-results で保存した結果から入れる）。
  すでに原産国が入っている在庫アイテムは変えない
"""

import argparse
import json
from pathlib import Path

from app import clef, shopify
from app.config import settings
from scripts.shopify_classify_photos import PREDICTIONS, _for_clef

OUT = Path("data/origin/results.json")
# タグの小さい文字を読むので、分類（512px）より大きく取る
WIDTH = 1600
MAX_IMAGES = 3
# これ以上なら Shopify に入れる
MIN_LEGIBLE = 0.7
MIN_CONFIDENCE = 0.7

STATE = "古着・洋服のタグ（ラベル）の写真から、原産国（Made in の表記）を読み取る"
COUNTRIES = {
    "US": "MADE IN U.S.A. / Made in USA / Made in America（ハワイ・プエルトリコを含む）",
    "CN": "Made in China / 中国製",
    "MX": "Made in Mexico / Hecho en México",
    "JP": "Made in Japan / 日本製",
    "CA": "Made in Canada / Fabriqué au Canada",
    "GB": "Made in England / Great Britain / U.K. / Scotland / Wales",
    "IE": "Made in Ireland",
    "IT": "Made in Italy",
    "FR": "Made in France",
    "PT": "Made in Portugal",
    "TR": "Made in Turkey",
    "HK": "Made in Hong Kong",
    "MO": "Made in Macau",
    "KR": "Made in Korea",
    "TW": "Made in Taiwan",
    "IN": "Made in India",
    "BD": "Made in Bangladesh",
    "LK": "Made in Sri Lanka",
    "PK": "Made in Pakistan",
    "VN": "Made in Vietnam",
    "ID": "Made in Indonesia",
    "TH": "Made in Thailand",
    "PH": "Made in Philippines",
    "MY": "Made in Malaysia",
    "GT": "Made in Guatemala / Hecho en Guatemala",
    "HN": "Made in Honduras / Hecho en Honduras",
    "SV": "Made in El Salvador / Hecho en El Salvador",
    "DO": "Made in Dominican Republic",
    "CR": "Made in Costa Rica",
    "PE": "Made in Peru / Hecho en Perú",
    "SG": "Made in Singapore",
    "MP": "Made in Saipan / Northern Mariana Islands (U.S.A.)（90年代のアメリカ古着に多い）",
    "NI": "Made in Nicaragua / Hecho en Nicaragua",
    "HT": "Made in Haiti",
    "JM": "Made in Jamaica",
    "CO": "Made in Colombia / Hecho en Colombia",
    "BR": "Made in Brazil",
    "EG": "Made in Egypt",
    "MU": "Made in Mauritius",
    "KH": "Made in Cambodia",
    "JO": "Made in Jordan",
    "IL": "Made in Israel",
    "DE": "Made in Germany / West Germany",
    "ES": "Made in Spain",
    "AU": "Made in Australia",
    "NZ": "Made in New Zealand",
    "other": "上のどれでもない国の表記がある",
    "none": "原産国の表記が写っていない・読めない",
}
QUESTIONS = {
    "legible": {
        "type": "noul",
        "instructions": (
            "写真のタグに、原産国の表記（MADE IN ○○ / Hecho en ○○ / ○○製）が"
            "はっきり写っていて読めるか。"
            "ブランド名・サイズ・洗濯表示だけで原産国が無いなら no"
        ),
    },
    "origin": {
        "type": "choice",
        "instructions": "タグに書かれている原産国はどれか（Made in の後の国名）",
        "criteria": COUNTRIES,
    },
}
# CMS の「原産国」（custom.made_in）の値 → 国コード
MADE_IN = {
    "アメリカ": "US",
    "アメリカ（ハワイ）": "US",
    "USA": "US",
    "日本": "JP",
    "中国": "CN",
    "メキシコ": "MX",
    "カナダ": "CA",
    "イギリス": "GB",
    "スコットランド": "GB",
    "イングランド": "GB",
    "アイルランド": "IE",
    "イタリア": "IT",
    "フランス": "FR",
    "ポルトガル": "PT",
    "トルコ": "TR",
    "香港": "HK",
    "韓国": "KR",
    "台湾": "TW",
    "インド": "IN",
    "バングラデシュ": "BD",
    "スリランカ": "LK",
    "ベトナム": "VN",
    "インドネシア": "ID",
    "タイ": "TH",
    "フィリピン": "PH",
    "マレーシア": "MY",
    "ペルー": "PE",
    "アラブ首長国連邦": "AE",
    "シンガポール": "SG",
    "サイパン": "MP",
    "ドイツ": "DE",
    "スペイン": "ES",
    "グアテマラ": "GT",
    "ホンジュラス": "HN",
    "エルサルバドル": "SV",
}

UPDATE_ORIGIN = """
mutation($id: ID!, $input: InventoryItemInput!) {
  inventoryItemUpdate(id: $id, input: $input) { userErrors { field message } }
}
"""

PRODUCT_QUERY = """
query($q: String) { products(first: 1, query: $q) { nodes { id handle title
  madeIn: metafield(namespace: "custom", key: "made_in") { value }
  variants(first: 50) { nodes { inventoryItem { id countryCodeOfOrigin } } } } } }
"""


def tag_images(predictions: dict, handles: set[str] | None) -> dict[str, list[dict]]:
    """商品のハンドル → タグのアップの写真（確信度の高い順）。"""
    by_product: dict[str, list[dict]] = {}
    for media_id, p in predictions.items():
        is_closeup = p["person"] < clef.THRESHOLD and p["whole"] < clef.THRESHOLD
        if not (is_closeup and p.get("part") == "tag" and p.get("part_confidence", 0) >= 0.3):
            continue
        for ref in p["products"]:
            if handles and ref["handle"] not in handles:
                continue
            by_product.setdefault(ref["handle"], []).append({"media_id": media_id, **p})
    for images in by_product.values():
        images.sort(key=lambda x: -x["part_confidence"])
    return by_product


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--product", action="append", help="商品のハンドル（省略するとタグの写真がある全商品）"
    )
    parser.add_argument(
        "--predictions", default=str(PREDICTIONS), help="写真の分類の結果（predictions.json）"
    )
    parser.add_argument("--apply", action="store_true", help="確信度の高いものを Shopify に入れる")
    parser.add_argument(
        "--from-results",
        action="store_true",
        help="判定し直さず、保存した結果（data/origin/results.json）を入れる（--apply と使う）",
    )
    args = parser.parse_args()

    client = shopify.Shopify(
        settings.shopify_store, settings.shopify_client_id, settings.shopify_client_secret
    )
    if args.from_results:
        apply_results(client, json.loads(OUT.read_text()), args.apply)
        return
    predictions = json.loads(Path(args.predictions).read_text())
    products = tag_images(predictions, set(args.product) if args.product else None)
    print(f"タグのアップの写真がある商品: {len(products)} 件", flush=True)

    model = None
    results = []
    for handle, images in sorted(products.items()):
        data = client.graphql(PRODUCT_QUERY, {"q": f"handle:{handle}"})["products"]["nodes"]
        if not data:
            continue
        product = data[0]
        items = [v["inventoryItem"] for v in product["variants"]["nodes"]]
        result = {
            "handle": handle,
            "title": product["title"],
            "images": [i["url"] for i in images[:MAX_IMAGES]],
        }
        made_in = (product.get("madeIn") or {}).get("value")
        if made_in and made_in.strip() in MADE_IN:
            result.update(
                source="custom.made_in",
                made_in=made_in,
                country=MADE_IN[made_in.strip()],
                legible=1.0,
                confidence=1.0,
            )
        else:
            if model is None:
                model = clef.from_settings(settings)
            pics = []
            for image in images[:MAX_IMAGES]:
                sep = "&" if "?" in image["url"] else "?"
                pics.append(_for_clef(client.download(f"{image['url']}{sep}width={WIDTH}")))
            answers = model.ask(STATE, QUESTIONS, pics)
            origin = answers["origin"]
            probs = origin.get("probabilities") or {}
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
            result.update(
                source="clef",
                country=origin["choice"] if isinstance(origin, dict) else origin,
                confidence=round(float(origin.get("confidence", 0)), 4),
                legible=round(float(answers["legible"]["noul"]), 4),
                top=[(k, round(v, 4)) for k, v in top],
            )
        result["ok"] = (
            result["country"] not in ("none", "other")
            and result["legible"] >= MIN_LEGIBLE
            and result["confidence"] >= MIN_CONFIDENCE
        )
        result["empty_items"] = [i["id"] for i in items if not i["countryCodeOfOrigin"]]
        results.append(result)
        mark = "○" if result["ok"] else "×"
        print(
            f"  {mark} {result['country']:5} 読める {result['legible']:.2f}"
            f" 確信 {result['confidence']:.2f}  {product['title']}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    ok = [r for r in results if r["ok"] and r["empty_items"]]
    counts: dict[str, int] = {}
    for r in ok:
        counts[r["country"]] = counts.get(r["country"], 0) + 1
    print(f"入れられる商品: {len(ok)} / {len(results)}  {counts}。保存: {OUT}")

    if args.apply:
        apply_results(client, results, True)


def apply_results(client: shopify.Shopify, results: list[dict], apply: bool) -> None:
    """確信度の高い結果を、原産国が空の在庫アイテムに入れる（入れる直前に、空かどうかを読み直す）。"""
    ok = [r for r in results if r["ok"] and r["empty_items"]]
    print(f"入れる商品: {len(ok)}" + ("" if apply else "（確認だけ）"))
    written = 0
    for r in ok:
        for item_id in r["empty_items"]:
            current = client.graphql(
                "query($id: ID!) { inventoryItem(id: $id) { countryCodeOfOrigin } }",
                {"id": item_id},
            )["inventoryItem"]
            if not current or current["countryCodeOfOrigin"]:
                continue
            if apply:
                res = client.graphql(
                    UPDATE_ORIGIN,
                    {"id": item_id, "input": {"countryCodeOfOrigin": r["country"]}},
                )
                shopify._assert_no_user_errors(
                    "inventoryItemUpdate", res["inventoryItemUpdate"]["userErrors"]
                )
            written += 1
    print(f"原産国を入れた在庫アイテム: {written}" + ("" if apply else "（確認だけ）"))


if __name__ == "__main__":
    main()
