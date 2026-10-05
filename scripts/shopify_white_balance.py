"""単品の商品写真のホワイトバランスと明るさを、基準の画像の壁の色に揃える（app/whitebalance.py）。

写真の種類は CLIP で判定し（app/classify.py）、単品の写真だけを補正する。
  uv run --env-file .env --group clip python -m scripts.shopify_white_balance \\
      [--product <ハンドル>]... [--report <ディレクトリ>] [--apply]
--apply を付けないときは、判定と補正の結果をレポート（HTML）に書き出すだけ。
--apply では、差し替える前に元の画像を ~/85store-shopify-originals/<日付>/ に保存する。
"""

import argparse
import datetime
import html
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from app import processing, shopify
from app import whitebalance as wb
from app.classify import KIND_LABELS, VERSION, Classification, Classifier
from app.config import settings

CACHE = Path("data/classify-cache.json")
# これ未満の確からしさで「単品」と判定したものは補正しない
MIN_CONFIDENCE = 0.6
CHUNK = 64
STATUS_LABELS = {"apply": "補正する", "ok": "補正不要", "review": "要確認", "skip": "対象外"}
THUMB_WIDTH = 300


@dataclass
class Item:
    media_id: str
    url: str
    handles: list[str]
    classification: Classification | None = None
    correction: wb.Correction | None = None
    status: str = ""
    reason: str = ""
    error: str = ""


def _key(item: Item) -> str:
    # 画像を差し替えると ?v= が、説明文やモデルを変えると VERSION が変わるので、判定し直す
    return f"{VERSION}|{item.media_id}|{urlparse(item.url).query}"


def _filename(url: str) -> str:
    return unquote(Path(urlparse(url).path).name)


def collect(client: shopify.Shopify, searches: list[str | None]) -> list[Item]:
    items: dict[str, Item] = {}
    for search in searches:
        for product in client.products(search):
            for media in product["media"]["nodes"]:
                if not (media.get("image") or {}).get("url"):
                    continue
                item = items.setdefault(media["id"], Item(media["id"], media["image"]["url"], []))
                item.handles.append(product["handle"])
    return list(items.values())


def load(client: shopify.Shopify, item: Item) -> tuple[bytes, np.ndarray]:
    data = client.download(item.url)
    image = processing.to_srgb(processing.decode(data), processing.icc_profile(data))
    return data, image


def decide(item: Item, image: np.ndarray) -> None:
    c = item.classification
    if c.kind != "single":
        item.status, item.reason = "skip", KIND_LABELS[c.kind]
        return
    height, width = image.shape[:2]
    if width >= height:
        # 店の単品の写真はすべて縦長。横長はディテール（服の一部）とみなす
        item.status, item.reason = "skip", "横長（単品は縦長）"
        return
    if c.confidence < MIN_CONFIDENCE:
        item.status, item.reason = "review", f"単品か確かでない（{c.confidence:.2f}）"
        return
    item.correction = wb.plan(image)
    item.status, item.reason = item.correction.status, item.correction.reason


def thumbnail(image: np.ndarray, region=None) -> np.ndarray:
    height, width = image.shape[:2]
    small = cv2.resize(image, (THUMB_WIDTH, round(height * THUMB_WIDTH / width)), cv2.INTER_AREA)
    if region:
        h, w = small.shape[:2]
        x0, x1, y0, y1 = region
        top_left, bottom_right = (int(w * x0), int(h * y0)), (int(w * x1), int(h * y1))
        cv2.rectangle(small, top_left, bottom_right, (0, 0, 255), 2)
    return small


def write_thumbs(report: Path, item: Item, image: np.ndarray) -> None:
    region = item.correction.region if item.correction else None
    name = item.media_id.rsplit("/", 1)[-1]
    cv2.imwrite(str(report / "img" / f"{name}_b.jpg"), thumbnail(image, region))
    if item.correction and item.correction.gains and item.status in ("apply", "review"):
        after = wb.apply_gains(thumbnail(image), item.correction.gains)
        cv2.imwrite(str(report / "img" / f"{name}_a.jpg"), after)


def card(item: Item, report: Path) -> str:
    name = item.media_id.rsplit("/", 1)[-1]
    imgs = f'<img src="img/{name}_b.jpg" alt="前">'
    if (report / "img" / f"{name}_a.jpg").exists():
        imgs += f'<img src="img/{name}_a.jpg" alt="後">'
    c, w = item.classification, item.correction
    lines = [f"<b>{html.escape(STATUS_LABELS[item.status])}</b>: {html.escape(item.reason)}"]
    if c:
        scores = " ".join(f"{KIND_LABELS[k]} {v:.2f}" for k, v in c.scores.items())
        lines.append(f"{KIND_LABELS[c.kind]}（{c.confidence:.2f}）<small>{scores}</small>")
    if w and w.measured_rgb:
        rgb = ", ".join(f"{v:.0f}" for v in w.measured_rgb)
        gains = ", ".join(f"{g:.2f}" for g in w.gains or ())
        lines.append(f"壁 RGB({rgb}) ΔE {w.delta_e:.1f} ゲイン({gains})")
    if item.error:
        lines.append(f'<span class="err">{html.escape(item.error)}</span>')
    lines.append(f"<small>{html.escape(', '.join(item.handles))} / {_filename(item.url)}</small>")
    return f'<div class="card"><div class="imgs">{imgs}</div><p>{"<br>".join(lines)}</p></div>'


def write_report(report: Path, items: list[Item], applied: bool) -> None:
    counts = {s: sum(1 for i in items if i.status == s) for s in STATUS_LABELS}
    kinds = {
        k: sum(1 for i in items if i.classification and i.classification.kind == k)
        for k in KIND_LABELS
    }
    summary = " / ".join(f"{STATUS_LABELS[s]} {n}" for s, n in counts.items())
    kind_summary = " / ".join(f"{KIND_LABELS[k]} {n}" for k, n in kinds.items())
    target = ", ".join(f"{v:.1f}" for v in wb.TARGET_RGB)
    sections = []
    for status in ["apply", "review", "ok", "skip"]:
        group = sorted(
            (i for i in items if i.status == status),
            key=lambda i: (
                i.classification.kind if i.classification else "",
                -(i.correction.delta_e or 0) if i.correction else 0,
            ),
        )
        if group:
            cards = "".join(card(i, report) for i in group)
            sections.append(
                f"<h2>{STATUS_LABELS[status]}（{len(group)}）</h2><div class=grid>{cards}</div>"
            )
    page = f"""<!doctype html><html lang=ja><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>ホワイトバランスの確認</title>
<style>body{{font:14px sans-serif;margin:16px;background:#fafafa;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(620px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px}}
.imgs{{display:flex;gap:6px}}.imgs img{{width:300px;height:auto}}
small{{color:#666}}.err{{color:#c00}}</style>
<h1>ホワイトバランスの確認{"（実行済み）" if applied else "（確認だけ）"}</h1>
<p>{summary}<br>種類: {kind_summary}<br>
目標の壁の色 RGB({target})。赤い枠が測った範囲。左が補正前、右が補正後。</p>
{"".join(sections)}</html>"""
    (report / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", action="append", help="商品のハンドル（省略すると全商品）")
    parser.add_argument("--report", default="data/white-balance-report")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-gain", type=float, help=f"ゲインの上限（既定 {wb.GAIN_RANGE[1]}）")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.max_gain:
        wb.GAIN_RANGE = (wb.GAIN_RANGE[0], args.max_gain)

    client = shopify.Shopify(
        settings.shopify_store,
        settings.shopify_client_id,
        settings.shopify_client_secret,
        settings.shopify_admin_token,
    )
    report = Path(args.report)
    (report / "img").mkdir(parents=True, exist_ok=True)
    originals = Path.home() / "85store-shopify-originals" / datetime.date.today().isoformat()
    cache: dict = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    searches = [shopify.product_search(p) for p in args.product] if args.product else [None]
    items = collect(client, searches)
    print(f"{'実行します' if args.apply else '確認だけ（--apply で実行）'}: 画像 {len(items)} 枚")
    classifier: Classifier | None = None

    with ThreadPoolExecutor(args.workers) as pool:
        for start in range(0, len(items), CHUNK):
            chunk = items[start : start + CHUNK]
            loaded = list(pool.map(lambda i: _try(load, client, i), chunk))
            ok = [(i, r) for i, r in zip(chunk, loaded, strict=True) if not isinstance(r, str)]
            for item, result in zip(chunk, loaded, strict=True):
                if isinstance(result, str):
                    item.status, item.reason, item.error = "review", "取得できない", result
            todo = [(i, img) for i, (_, img) in ok if _key(i) not in cache]
            if todo:
                classifier = classifier or Classifier()
                for (item, _), c in zip(
                    todo, classifier.classify([img for _, img in todo]), strict=True
                ):
                    cache[_key(item)] = c.__dict__
            for item, (_, image) in ok:
                item.classification = Classification(**cache[_key(item)])
                decide(item, image)
                write_thumbs(report, item, image)
            if args.apply:
                targets = [(i, d, img) for i, (d, img) in ok if i.status == "apply"]
                for item, error in pool.map(
                    lambda t: (t[0], _try(replace, client, originals, *t)), targets
                ):
                    if error:
                        item.error = error
            CACHE.parent.mkdir(exist_ok=True)
            CACHE.write_text(json.dumps(cache))
            print(f"  {min(start + CHUNK, len(items))} / {len(items)}")

    write_report(report, items, args.apply)
    counts = {STATUS_LABELS[s]: sum(1 for i in items if i.status == s) for s in STATUS_LABELS}
    errors = sum(1 for i in items if i.error)
    print(f"{counts}、失敗 {errors} 件。レポート: {(report / 'index.html').resolve()}")


def replace(client, originals: Path, item: Item, data: bytes, image: np.ndarray) -> None:
    originals.mkdir(parents=True, exist_ok=True)
    name = item.media_id.rsplit("/", 1)[-1]
    (originals / f"{name}_{_filename(item.url)}").write_bytes(data)
    corrected = processing.encode(
        wb.apply_gains(image, item.correction.gains), "jpeg", settings.fit_quality
    )
    filename = Path(_filename(item.url)).stem + ".jpg"
    client.replace_image(item.media_id, client.stage_upload(corrected, filename, "image/jpeg"))


def _try(fn, *args):
    """失敗したら例外の文字列を返す（1枚の失敗で全体を止めない）。

    成功したら戻り値（replace は None）。
    """
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    main()
