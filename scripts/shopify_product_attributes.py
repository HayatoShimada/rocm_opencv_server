"""商品の属性（種類・色・柄・テイスト・季節・シルエットなど）を Clef で判定する。

質問は app/product_attributes.py にある。

  uv run --env-file .env --group clip --group clef-local \\
      python -m scripts.shopify_product_attributes [--sample 30] [--product <ハンドル>]... [--force]
既定は手元の GPU で判定する。CLEF_BACKEND=workers-ai なら Workers AI。
販売中の商品だけを対象にする。--sample では、種類が偏らないように選ぶ。
写真は、写真の分類（scripts/shopify_classify_photos.py の data/photo-labels）から
「全体・正面」と「着用」を1枚ずつ選んで渡す（分類が無ければ1枚目）。
判定は data/product-attributes/predictions.json に、確かめるための一覧を index.html に書く。
Shopify には書き込まない。
"""

import argparse
import hashlib
import html
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

from app import clef, product_attributes, shopify
from app.config import settings
from scripts.shopify_classify_photos import PREDICTIONS as PHOTO_PREDICTIONS
from scripts.shopify_classify_photos import _for_clef

DATA = Path("data/product-attributes")
PREDICTIONS = DATA / "predictions.json"
WIDTH = 512
WORKERS = 3
SEED = 85


def fetch_products(client: shopify.Shopify, search: str) -> list[dict]:
    products, after = [], None
    while True:
        data = client.graphql(
            product_attributes.PRODUCT_QUERY, {"after": after, "query": search, "first": 10}
        )["products"]
        products += data["nodes"]
        if not data["pageInfo"]["hasNextPage"]:
            return products
        after = data["pageInfo"]["endCursor"]


def sample(products: list[dict], n: int) -> list[dict]:
    """種類ごとに分けて、順に1点ずつ取る（数の多い種類ばかりにならないように）。"""
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for p in products:
        groups[p.get("productType") or ""].append(p)
    for items in groups.values():
        rng.shuffle(items)
    picked = []
    while len(picked) < n and any(groups.values()):
        for key in sorted(groups, key=lambda k: -len(groups[k])):
            if groups[key] and len(picked) < n:
                picked.append(groups[key].pop())
    return picked


def pick_images(product: dict, photos: dict) -> list[str]:
    """渡す写真の URL。全体・正面を1枚と、着用を1枚。"""
    media = [m for m in product["media"]["nodes"] if m.get("image")]
    if not media:
        return []
    labeled = [(m, photos[m["id"]]) for m in media if m["id"] in photos]
    whole = [
        (p["whole"] * (p["view_confidence"] if p.get("view") == "front" else 0.3), m)
        for m, p in labeled
        if p["person"] < clef.THRESHOLD and p["whole"] >= clef.THRESHOLD
    ]
    worn = [(p["person"], m) for m, p in labeled if p["person"] >= clef.THRESHOLD]
    urls = [max(whole, key=lambda t: t[0])[1]["image"]["url"]] if whole else []
    urls = urls or [media[0]["image"]["url"]]
    if worn:
        urls.append(max(worn, key=lambda t: t[0])[1]["image"]["url"])
    return urls


def _key(product: dict, urls: list[str], model: str) -> str:
    # 写真・説明・質問・モデルのどれかが変わったら判定し直す
    images = ",".join(urlparse(u).path.rsplit("/", 1)[-1] + "?" + urlparse(u).query for u in urls)
    text = json.dumps(product_attributes.product_state(product), ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(text.encode()).hexdigest()[:8]
    return f"{product_attributes.VERSION}|{model}|{images}|{digest}"


def summarize(answers: dict) -> dict:
    """答えから、確かめるときに見るものだけを取り出す。"""
    out = {}
    for qid, a in answers.items():
        if "noul" in a:
            out[qid] = round(a["noul"], 3)
        elif "choice" in a:
            out[qid] = {"choice": a["choice"], "confidence": round(a["confidence"], 3)}
        else:
            out[qid] = {"score": round(a["score"], 2), "confidence": round(a["confidence"], 3)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=30, help="判定する商品の数（既定 30）")
    parser.add_argument("--product", action="append", help="商品のハンドル（--sample より優先）")
    parser.add_argument("--force", action="store_true", help="保存してある判定も取り直す")
    args = parser.parse_args()

    client = shopify.Shopify(
        settings.shopify_store, settings.shopify_client_id, settings.shopify_client_secret
    )
    if args.product:
        products = [
            p for h in args.product for p in fetch_products(client, shopify.product_search(h))
        ]
    else:
        products = sample(fetch_products(client, "status:active"), args.sample)
    photos = json.loads(PHOTO_PREDICTIONS.read_text()) if PHOTO_PREDICTIONS.exists() else {}
    predictions = json.loads(PREDICTIONS.read_text()) if PREDICTIONS.exists() else {}

    t0 = time.time()
    model = clef.from_settings(settings)
    print(f"モデル: {model.model}（読み込み {time.time() - t0:.1f} 秒）", flush=True)

    todo = []
    for product in products:
        urls = pick_images(product, photos)
        key = _key(product, urls, model.model)
        saved = predictions.get(product["id"])
        if not (saved and saved.get("key") == key and not args.force):
            todo.append((product, urls, key))
    print(f"商品 {len(products)} 点のうち、判定する: {len(todo)} 点", flush=True)

    def run(product: dict, urls: list[str], key: str) -> dict:
        images = []
        for url in urls:
            sep = "&" if "?" in url else "?"
            images.append(_for_clef(client.download(f"{url}{sep}width={WIDTH}")))
        state = product_attributes.product_state(product)
        started = time.time()
        answers = model.ask(state, product_attributes.QUESTIONS, images)
        return {
            "key": key,
            "handle": product["handle"],
            "title": product["title"],
            "productType": product.get("productType"),
            "colors": product_attributes.existing_colors(product),
            "description": product.get("description") or "",
            "material": (product.get("material") or {}).get("value"),
            "images": urls,
            "seconds": round(time.time() - started, 2),
            "answers": summarize(answers),
        }

    DATA.mkdir(parents=True, exist_ok=True)
    failed = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = {ex.submit(run, *t): t[0] for t in todo}
        for done, future in enumerate(as_completed(futures), 1):
            product = futures[future]
            try:
                predictions[product["id"]] = future.result()
            except Exception as e:  # noqa: BLE001  1点の失敗で全体を止めない
                failed += 1
                print(f"  失敗: {product['handle']}: {e}", flush=True)
            if done % 10 == 0 or done == len(futures):
                print(f"  {done} / {len(futures)}", flush=True)
    PREDICTIONS.write_text(json.dumps(predictions, ensure_ascii=False, indent=1))

    ids = [p["id"] for p in products if p["id"] in predictions]
    by_id = {p["id"]: p for p in products}
    report = write_report([(by_id[i], predictions[i]) for i in ids])
    print(f"失敗 {failed} 件。レポート: {report.resolve()}")


def score(rows: list[tuple[dict, dict]]) -> dict[str, tuple[int, int]]:
    """種類・色・柄で、すでに入っている値と合っている数（合った数, 比べた数）。"""
    result = Counter()
    for product, pred in rows:
        exp = product_attributes.expected(product)
        for qid in ("category", "color", "pattern"):
            if exp[qid]:
                result[f"{qid}_n"] += 1
                result[f"{qid}_ok"] += pred["answers"][qid]["choice"] in exp[qid]
    return {q: (result[f"{q}_ok"], result[f"{q}_n"]) for q in ("category", "color", "pattern")}


def _choice_label(qid: str, choice: str) -> str:
    criteria = product_attributes.QUESTIONS[qid]["criteria"]
    if isinstance(criteria, dict):
        return criteria.get(choice, choice).split("（")[0]
    return choice


def write_report(rows: list[tuple[dict, dict]]) -> Path:
    e = html.escape
    acc = score(rows)
    seconds = [pred["seconds"] for _, pred in rows]
    head = " / ".join(
        f"{name}: {ok}/{n}（{ok / n:.0%}）"
        for name, (ok, n) in zip(("種類", "色", "柄"), acc.values(), strict=True)
        if n
    )
    if seconds:
        head += f" ・ 1点 平均 {sum(seconds) / len(seconds):.2f} 秒"
    cards = []
    for product, pred in rows:
        a = pred["answers"]
        exp = product_attributes.expected(product)
        lines = []
        for qid, name, existing in (
            ("category", "種類", product.get("productType") or "—"),
            ("color", "色", "・".join(pred["colors"]) or "—"),
            ("pattern", "柄", "・".join(pred["colors"]) or "—"),
        ):
            choice = a[qid]["choice"]
            mark = "" if not exp[qid] else ("ok" if choice in exp[qid] else "ng")
            lines.append(
                f'<tr class="{mark}"><th>{name}</th><td>{e(_choice_label(qid, choice))}'
                f" <small>{a[qid]['confidence']:.2f}</small></td>"
                f"<td><small>入力済み: {e(existing)}</small></td></tr>"
            )
        tastes = [
            f"{_taste(k)} <small>{a[f'taste_{k}']:.2f}</small>"
            for k in product_attributes.TASTES
            if a[f"taste_{k}"] >= 0.3
        ]
        lines.append(f"<tr><th>テイスト</th><td colspan=2>{'・'.join(tastes) or '—'}</td></tr>")
        for qid, name in (("season", "季節"), ("fit", "シルエット"), ("gender", "対象")):
            lines.append(
                f"<tr><th>{name}</th><td colspan=2>{e(_choice_label(qid, a[qid]['choice']))}"
                f" <small>{a[qid]['confidence']:.2f}</small></td></tr>"
            )
        thick = a["thickness"]["score"]
        lines.append(
            f"<tr><th>厚さ</th><td colspan=2>{thick:.2f}"
            " <small>（0 薄手 〜 2 厚手）</small></td></tr>"
        )
        imgs = "".join(f'<img src="{e(u)}&width=240" loading=lazy>' for u in pred["images"])
        cards.append(
            f'<article><div class=imgs>{imgs}</div><div><h2><a href="https://shop.85-store.com/products/'
            f'{e(pred["handle"])}" target=_blank>{e(pred["title"])}</a></h2>'
            f"<p>{e(pred['description'][:160])}</p>"
            f"<p><small>素材: {e(pred['material'] or '—')}</small></p>"
            f"<table>{''.join(lines)}</table></div></article>"
        )
    page = f"""<!doctype html><html lang=ja><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>商品の属性（Clef）</title>
<style>{STYLE}</style>
<h1>商品の属性（Clef）</h1><p>{e(head)}（種類・色・柄は、入力済みの値と比べた正しさ）</p>
{"".join(cards)}</html>"""
    path = DATA / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


STYLE = """
body{font-family:system-ui,sans-serif;margin:16px;color:#222;background:#fafafa}
article{display:flex;gap:16px;background:#fff;border:1px solid #ddd;border-radius:8px;
  padding:12px;margin:12px 0}
.imgs{display:flex;gap:6px;flex:none}
.imgs img{width:160px;height:200px;object-fit:cover;border-radius:4px}
h2{font-size:15px;margin:0 0 4px} p{margin:4px 0;font-size:13px;color:#555}
table{border-collapse:collapse;font-size:13px;margin-top:6px}
th{text-align:left;padding:2px 10px 2px 0;color:#666;font-weight:normal}
td{padding:2px 10px 2px 0} small{color:#888}
tr.ok td:first-of-type{color:#18794e} tr.ng td:first-of-type{color:#c62828;font-weight:bold}
@media (max-width:640px){article{flex-direction:column}}
"""


def _taste(key: str) -> str:
    return product_attributes.TASTES[key].split("（")[0]


if __name__ == "__main__":
    main()
