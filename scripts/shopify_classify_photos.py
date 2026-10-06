"""Shopify の全商品の画像を、Clef（Workers AI）で「着用・全体・アップ」に分ける（app/clef.py）。

  uv run --env-file .env python -m scripts.shopify_classify_photos \\
      [--product <ハンドル>]... [--force]
判定は data/photo-labels/predictions.json に保存する。
画像（?v=）・質問・モデルが変わっていなければ取り直さない。
結果が正しいかは scripts/photo_review.py の画面で確かめる。
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import cv2

from app import clef, processing, shopify
from app.config import settings

DATA = Path("data/photo-labels")
PREDICTIONS = DATA / "predictions.json"
# Clef に渡す画像の幅（Shopify の CDN で縮めて取る。人・全体の判定にはこれで足りる）
WIDTH = 512
WORKERS = 6


def load_predictions() -> dict:
    return json.loads(PREDICTIONS.read_text()) if PREDICTIONS.exists() else {}


def _key(url: str, model: str) -> str:
    # 画像を差し替えると ?v= が、質問を変えると VERSION が変わるので、判定し直す
    return f"{clef.VERSION}|{model}|{urlparse(url).query}"


def collect(client: shopify.Shopify, searches: list[str | None]) -> dict[str, dict]:
    """画像の ID ごとに、URL と使っている商品をまとめる。

    色違いで共有している画像は1回だけ判定する。
    """
    items: dict[str, dict] = {}
    for search in searches:
        for product in client.products(search):
            for position, media in enumerate(product["media"]["nodes"], 1):
                if not media.get("image"):
                    continue
                item = items.setdefault(
                    media["id"],
                    {"url": media["image"]["url"], "products": []},
                )
                ref = {"handle": product["handle"], "title": product["title"], "position": position}
                if ref not in item["products"]:
                    item["products"].append(ref)
    return items


def _for_clef(data: bytes) -> tuple[bytes, str]:
    """Clef が受け付ける形式（JPEG・PNG・WebP）にする。それ以外（GIF など）は JPEG にし直す。"""
    if data[:3] == b"\xff\xd8\xff":
        return data, "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return data, "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return data, "image/webp"
    image = processing.decode(data)
    return cv2.imencode(".jpg", image[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 90])[
        1
    ].tobytes(), "image/jpeg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", action="append", help="商品のハンドル（省略すると全商品）")
    parser.add_argument("--force", action="store_true", help="保存してある判定も取り直す")
    args = parser.parse_args()

    client = shopify.Shopify(
        settings.shopify_store, settings.shopify_client_id, settings.shopify_client_secret
    )
    model = clef.Clef(
        settings.cloudflare_account_id, settings.cloudflare_api_token, settings.clef_model
    )
    searches = [shopify.product_search(p) for p in args.product] if args.product else [None]
    items = collect(client, searches)
    predictions = load_predictions()

    todo = []
    for media_id, item in items.items():
        key = _key(item["url"], settings.clef_model)
        saved = predictions.get(media_id)
        if saved and saved.get("key") == key and not args.force:
            saved.update(url=item["url"], products=item["products"])
        else:
            todo.append((media_id, item, key))
    print(f"画像 {len(items)} 枚のうち、判定する: {len(todo)} 枚", flush=True)

    def run(media_id: str, item: dict, key: str) -> tuple[str, dict]:
        sep = "&" if "?" in item["url"] else "?"
        data = client.download(f"{item['url']}{sep}width={WIDTH}")
        labels = model.classify(*_for_clef(data))
        return media_id, {
            **item,
            "key": key,
            "person": round(labels.person, 4),
            "whole": round(labels.whole, 4),
            "view": labels.view,
            "view_confidence": round(labels.view_confidence, 4),
            "part": labels.part,
            "part_confidence": round(labels.part_confidence, 4),
        }

    failed = 0
    DATA.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = [ex.submit(run, *t) for t in todo]
        for done, future in enumerate(as_completed(futures), 1):
            try:
                media_id, prediction = future.result()
                predictions[media_id] = prediction
            except Exception as e:  # noqa: BLE001  1枚の失敗で全体を止めない
                failed += 1
                print(f"  失敗: {e}", flush=True)
            if done % 100 == 0 or done == len(futures):
                print(f"  {done} / {len(futures)}", flush=True)
                PREDICTIONS.write_text(json.dumps(predictions, ensure_ascii=False))

    # 商品から外れた画像は残さない（分類の画面に出さない）
    predictions = (
        {k: v for k, v in predictions.items() if k in items} if not args.product else predictions
    )
    PREDICTIONS.write_text(json.dumps(predictions, ensure_ascii=False))
    kinds: dict[str, int] = {}
    for p in predictions.values():
        kind = clef.photo_kind(p["person"] >= clef.THRESHOLD, p["whole"] >= clef.THRESHOLD)
        kinds[clef.KIND_LABELS[kind]] = kinds.get(clef.KIND_LABELS[kind], 0) + 1
    print(f"{kinds}、失敗 {failed} 件。保存: {PREDICTIONS}")


if __name__ == "__main__":
    main()
