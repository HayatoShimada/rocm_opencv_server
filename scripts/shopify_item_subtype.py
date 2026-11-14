"""商品の細分類（ドレスシャツ・ネルシャツ・フリースなど）を、写真の雰囲気から Clef で判定する。

質問は app/item_subtype.py。語彙（商品タイプ → 細分類）は CRM のものを JSON で受け取る。

  uv run --env-file .env --group clip --group clef-local \\
      python -m scripts.shopify_item_subtype [--subtypes <subtypes.json>] \\
      [--guess <rule-guess.json>] [--mode photo|text|both] [--limit N] [--force] [--apply]

判定のあと、CMS に送る候補（写真だけの判定で、確からしさ 0.5 以上）を
data/item-subtype/candidates.json に書き、細分類ごと・確からしさの帯ごとの件数を出す
（既定は確認だけ）。--apply で CMS の
POST /api/products/item-subtype-candidates に 50 件ずつ送る（CMS 側が上書き）。
--subtypes を省くと、CMS の GET /api/products/product-types から語彙を読む。

--mode photo は写真だけ、text は題名・説明・素材・採寸も渡す（ブランドは渡さない）、
both は両方（既定）。
対象は、語彙にある品目の販売中の商品（--guess があればその商品 ID だけ）。
写真は shopify_product_attributes.py と同じ選び方（全体・正面 + 着用）。
判定は data/item-subtype/predictions.json に、1点ごとに保存する（落ちても続きから再開できる）。
集計とレビュー用の HTML は python -m scripts.item_subtype_report。
Shopify には読むだけで、書き込まない。
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import httpx2 as httpx

from app import clef, item_subtype, shopify
from app.config import settings
from scripts.shopify_classify_photos import PREDICTIONS as PHOTO_PREDICTIONS
from scripts.shopify_classify_photos import _for_clef
from scripts.shopify_condition_drafts import CMS_URL
from scripts.shopify_product_attributes import WIDTH, WORKERS, fetch_products, pick_images

DATA = Path("data/item-subtype")
PREDICTIONS = DATA / "predictions.json"
CANDIDATES = DATA / "candidates.json"
BATCH = 50
# 判定のキャッシュのキーに入れるモデル名（モデルを読み込まずに、キャッシュと照らすため）
MODEL_NAME = "clef-flash"
SAVE_EVERY = 10


def cache_key(product: dict, urls: list[str], model: str, mode: str, qs: dict) -> str:
    """写真・渡す文字・質問・モデルのどれかが変わったら判定し直す。"""
    parsed = [urlparse(u) for u in urls]
    images = ",".join(p.path.rsplit("/", 1)[-1] + "?" + p.query for p in parsed)
    body = json.dumps([item_subtype.state(product, mode), qs], ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(body.encode()).hexdigest()[:8]
    return f"{item_subtype.VERSION}|{model}|{mode}|{images}|{digest}"


def modes_for(name: str) -> tuple[str, ...]:
    return item_subtype.MODES if name == "both" else (name,)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtypes", type=Path, help="細分類の語彙の JSON（省くと CMS から読む）")
    parser.add_argument("--guess", type=Path, help="対象を、この JSON の商品 ID に絞る（比べる用）")
    parser.add_argument("--mode", choices=("photo", "text", "both"), default="photo")
    parser.add_argument("--limit", type=int, help="先頭から N 点だけ（試し用）")
    parser.add_argument("--force", action="store_true", help="保存してある判定も取り直す")
    parser.add_argument("--apply", action="store_true", help="候補を CMS に送る")
    args = parser.parse_args()

    http = httpx.Client(timeout=60, follow_redirects=True)
    vocabulary = (
        item_subtype.load_vocabulary(args.subtypes) if args.subtypes else vocabulary_from_cms(http)
    )
    client = shopify.Shopify(
        settings.shopify_store, settings.shopify_client_id, settings.shopify_client_secret
    )
    products = [
        p for p in fetch_products(client, "status:active") if p.get("productType") in vocabulary
    ]
    if args.guess:
        wanted = set(json.loads(args.guess.read_text()))
        products = [p for p in products if p["id"] in wanted]
    products = products[: args.limit] if args.limit else products
    photos = json.loads(PHOTO_PREDICTIONS.read_text()) if PHOTO_PREDICTIONS.exists() else {}
    predictions = json.loads(PREDICTIONS.read_text()) if PREDICTIONS.exists() else {}

    # 判定が要る商品が無ければ、モデルを読み込まない（キャッシュだけで候補を作る）
    model = None
    modes = modes_for(args.mode)
    todo = []
    for product in products:
        urls = pick_images(product, photos)
        qs = item_subtype.questions(product["productType"], vocabulary)
        keys = {m: cache_key(product, urls, MODEL_NAME, m, qs) for m in modes}
        saved = predictions.get(product["id"], {})
        need = [m for m in modes if args.force or saved.get(m, {}).get("key") != keys[m]]
        if need:
            todo.append((product, urls, qs, {m: keys[m] for m in need}))
    print(
        f"商品 {len(products)} 点のうち、判定する: {len(todo)} 点（{'・'.join(modes)}）", flush=True
    )

    if todo:
        t0 = time.time()
        model = clef.from_settings(settings)
        print(f"モデル: {model.model}（読み込み {time.time() - t0:.1f} 秒）", flush=True)
        if model.model != MODEL_NAME:
            parser.error(f"モデルが {MODEL_NAME} ではありません: {model.model}")

    def run(product: dict, urls: list[str], qs: dict, keys: dict) -> dict:
        images = []
        for url in urls:
            sep = "&" if "?" in url else "?"
            images.append(_for_clef(client.download(f"{url}{sep}width={WIDTH}")))
        out = {}
        for mode, key in keys.items():
            started = time.time()
            answers = model.ask(item_subtype.state(product, mode), qs, images)
            out[mode] = {
                "key": key,
                "seconds": round(time.time() - started, 2),
                "answers": item_subtype.summarize(answers),
            }
        return out

    DATA.mkdir(parents=True, exist_ok=True)
    failed = 0
    started = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = {ex.submit(run, *t): t for t in todo}
        for done, future in enumerate(as_completed(futures), 1):
            product, urls = futures[future][:2]
            try:
                result = future.result()
                entry = predictions.setdefault(product["id"], {})
                entry.update(
                    handle=product["handle"],
                    title=product["title"],
                    productType=product["productType"],
                    images=urls,
                    **result,
                )
            except Exception as e:  # noqa: BLE001  1点の失敗で全体を止めない
                failed += 1
                print(f"  失敗: {product['handle']}: {e}", flush=True)
            if done % SAVE_EVERY == 0 or done == len(futures):
                PREDICTIONS.write_text(json.dumps(predictions, ensure_ascii=False, indent=1))
                print(f"  {done} / {len(futures)}（{time.time() - started:.0f} 秒）", flush=True)
    print(f"失敗 {failed} 件。判定: {PREDICTIONS.resolve()}")
    send_candidates(http, predictions, products, args.apply)


def vocabulary_from_cms(http: httpx.Client) -> dict[str, list[dict]]:
    """CMS（CRM の語彙のキャッシュ）から、細分類のある品目の語彙を読む。"""
    res = http.get(f"{CMS_URL}/api/products/product-types")
    res.raise_for_status()
    body = res.json()
    if not body.get("available"):
        raise SystemExit("CMS が語彙を返せません。--subtypes <JSON> を渡してください")
    vocabulary = {t["name"]: t["subtypes"] for t in body["types"] if t.get("subtypes")}
    if not vocabulary:
        raise SystemExit("CMS に細分類の語彙がありません。--subtypes <JSON> を渡してください")
    return vocabulary


def send_candidates(
    http: httpx.Client, predictions: dict, products: list[dict], apply: bool
) -> None:
    """写真だけの判定から候補を作って保存し、件数を出す。apply のときだけ CMS に送る。"""
    payload = item_subtype.candidates_payload(predictions, [p["id"] for p in products])
    CANDIDATES.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    items = payload["items"]
    types = {p["id"]: p["productType"] for p in products}
    print(f"CMS に送る候補: {len(items)} / {len(products)}" + ("" if apply else "（確認だけ）"))
    for key, bands in sorted(item_subtype.band_counts(items, types).items()):
        print(f"  {key}: " + "・".join(f"{b} {n}" for b, n in bands.items()))
    print(f"保存: {CANDIDATES.resolve()}")
    if not apply or not items:
        return
    saved, skipped = 0, []
    for i in range(0, len(items), BATCH):
        res = http.post(
            f"{CMS_URL}/api/products/item-subtype-candidates",
            json={**payload, "items": items[i : i + BATCH]},
        )
        res.raise_for_status()
        body = res.json()
        saved += body.get("saved", 0)
        skipped += body.get("skipped") or []
    print(f"入れた: {saved}  飛ばした: {len(skipped)}")
    for s in skipped:
        print(f"  飛ばした {s.get('productId')}: {s.get('reason')}")


if __name__ == "__main__":
    main()
