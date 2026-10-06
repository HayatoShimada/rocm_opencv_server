"""商品画像の自動の alt を、写真の区分を付けた形にする。

例:「商品名（正面）」「商品名（タグのアップ）」。

区分は Clef の判定（scripts/shopify_classify_photos.py）から決める。
着用・全体・アップの別は、確かめた結果（scripts/photo_review.py の labels.json）があればそれを使う。
  uv run --env-file .env python -m scripts.shopify_photo_alts [--product <ハンドル>]... [--apply]
書き換えるのは、alt が空か自動の alt（商品名（n枚目）・商品名（区分））の画像だけ。
人が入れた alt と、ほかの商品と共有している画像（色違いの共通の写真など）には触らない。
--apply では、書き換える前の alt を ~/85store-shopify-originals/alts-<日時>.json に保存する。
"""

import argparse
import datetime
import json
from collections import Counter
from pathlib import Path

from app import clef, shopify
from app.config import settings
from scripts.shopify_classify_photos import DATA, load_predictions

BACKUP_DIR = Path.home() / "85store-shopify-originals"


def kind_of(media_id: str, prediction: dict, labels: dict) -> str:
    label = labels.get(media_id)
    if label:
        return clef.photo_kind(label["person"], label["whole"])
    return clef.photo_kind(
        prediction["person"] >= clef.THRESHOLD, prediction["whole"] >= clef.THRESHOLD
    )


def plan(products: list[dict], predictions: dict, labels: dict) -> tuple[list[dict], Counter]:
    """書き換える画像（id・商品・前の alt・新しい alt）と、見送った理由ごとの数を返す。"""
    used = Counter(m["id"] for p in products for m in p["media"]["nodes"] if m.get("image"))
    changes, skipped = [], Counter()
    for product in products:
        for index, media in enumerate(product["media"]["nodes"]):
            if not media.get("image"):
                continue
            alt = media.get("alt") or ""
            if used[media["id"]] > 1:
                skipped["ほかの商品と共有している"] += 1
                continue
            if alt and shopify.auto_alt_label(alt, product["title"], index) is None:
                skipped["人が入れた alt"] += 1
                continue
            prediction = predictions.get(media["id"])
            if not prediction or "view" not in prediction:
                skipped["判定がない（分類し直してください）"] += 1
                continue
            label = shopify.photo_label(
                kind_of(media["id"], prediction, labels),
                prediction["view"],
                prediction["view_confidence"],
                prediction["part"],
                prediction["part_confidence"],
            )
            new = shopify.auto_alt(product["title"], index, label)
            if new == alt:
                skipped["もう同じ alt"] += 1
                continue
            changes.append(
                {"id": media["id"], "handle": product["handle"], "before": alt, "after": new}
            )
    return changes, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", action="append", help="商品のハンドル（省略すると全商品）")
    parser.add_argument("--apply", action="store_true", help="Shopify の alt を書き換える")
    args = parser.parse_args()

    client = shopify.Shopify(
        settings.shopify_store, settings.shopify_client_id, settings.shopify_client_secret
    )
    # 共有している画像を見分けるため、商品を指定したときも全商品を読む
    products = list(client.products())
    if args.product:
        targets = set(args.product)
        products_to_change = [p for p in products if p["handle"] in targets]
    else:
        products_to_change = products
    predictions = load_predictions()
    labels_path = DATA / "labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}

    changes, skipped = plan(products, predictions, labels)
    handles = {p["handle"] for p in products_to_change}
    changes = [c for c in changes if c["handle"] in handles]
    words = Counter(c["after"].rsplit("（", 1)[-1].rstrip("）") for c in changes)
    print(f"書き換える: {len(changes)} 枚 {dict(words.most_common())}")
    print(f"見送る: {dict(skipped)}")
    for c in changes[:15]:
        print(f"  {c['before'] or '（空）'} → {c['after']}")
    if not args.apply or not changes:
        return

    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    backup = BACKUP_DIR / f"alts-{stamp}.json"
    backup.write_text(json.dumps(changes, ensure_ascii=False, indent=1))
    print(f"前の alt を保存しました: {backup}")
    # fileUpdate は商品ごとにまとめて送る（1回の件数を小さくして、処理待ちで止まらないように）
    by_product: dict[str, list[dict]] = {}
    for c in changes:
        by_product.setdefault(c["handle"], []).append({"id": c["id"], "alt": c["after"]})
    failed = 0
    for done, (handle, files) in enumerate(by_product.items(), 1):
        try:
            client.update_alts(files)
        except shopify.ShopifyError as e:
            failed += len(files)
            print(f"  失敗: {handle} {e}")
        if done % 50 == 0 or done == len(by_product):
            print(f"  {done} / {len(by_product)} 商品", flush=True)
    print(f"書き換えました: {len(changes) - failed} 枚、失敗 {failed} 枚")


if __name__ == "__main__":
    main()
