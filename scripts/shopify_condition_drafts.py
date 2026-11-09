"""古着の写真から Clef で傷・汚れを探し、「状態メモ」の候補を 85store-cms に入れる。

ショップの商品ページの状態メモ（custom.condition_note）は、古着 257 点のうち
39 点にしか入っていない。候補は CMS の商品の画面の「状態メモの候補」に出て、
人が直してから状態メモに入れる（自動では公開しない）。

  uv run --env-file .env --group clip --group clef-local \\
      python -m scripts.shopify_condition_drafts [--limit 50] [--apply]
  # CMS を通さず、この商品だけ試す
  uv run ... python -m scripts.shopify_condition_drafts --product <ハンドル>

- 対象は CMS の GET /api/products/condition-drafts/targets
  （売り場に出ている古着で、状態メモも候補も無い商品）
- Clef は文を書かないので、傷・汚れの種類ごとに「写っているか」（noul）、
  いちばん目立つものの場所（choice）、目立つか（noul）を聞き、決まった型の文（1〜2 文）にする。
  小さな傷は写真をまとめて聞くと埋もれるので、1 枚ずつ聞いていちばん確かな答えを取る
- 状態ランク（SS〜C）と矛盾しないように、SS・S では確信度が高いものだけを書く。
  B・C なのに何も見つからないときは候補を入れない（人が書く）
- 既定は確認だけ（data/condition/results.json に保存）。
  --apply で CMS の POST /api/products/condition-drafts に送る。--from-results で保存した結果を送る
- 限界（2026-11-09 に、人が書いた状態メモ 9 点と比べた）: 正面・背面の写真だけでは、
  小さな穴・虫食い・破れは見つけられない（4 点中 0 点）。シミは 1 点見つけた。
  「目立った傷や汚れは見受けられません」は A の商品と合った。
  傷・汚れのアップの写真があると見つけやすい
"""

import argparse
import json
from pathlib import Path

import httpx2 as httpx

from app import clef, shopify
from app.config import settings
from scripts.shopify_classify_photos import _for_clef

OUT = Path("data/condition/results.json")
CMS_URL = "https://cms.85-store.com"
WIDTH = 1200
MAX_IMAGES = 4

STATE = (
    "古着屋のオンラインストアの商品写真から、服や小物の状態（シミ・汚れ・穴・色あせ・毛玉・ほつれ）を確かめる。"
    "写真の明るさ・影・柄・生地の織り・プリント・もとからのデザイン（ダメージ加工・ウォッシュ）は、傷や汚れに数えない"
)
# 種類 → (Clef への質問, 状態メモに書くことば)
KINDS = {
    "stain": ("シミ・汚れ・変色（黄ばみ・黒ずみを含む）が写っているか", "シミ"),
    "hole": ("穴・破れ・裂けが写っているか", "穴"),
    "fade": ("部分的な色あせ・色落ち（日焼け・襟や袖口の退色）が写っているか", "色あせ"),
    "pilling": ("毛玉・毛羽立ちがはっきり写っているか", "毛玉"),
    "fray": ("ほつれ・糸のほどけ・すり切れが写っているか", "ほつれ"),
}
LOCATIONS = {
    "front": "前身頃",
    "back": "後ろ身頃",
    "collar": "襟まわり",
    "left_sleeve": "左袖",
    "right_sleeve": "右袖",
    "cuff": "袖口",
    "hem": "裾",
    "pocket": "ポケットまわり",
    "waist": "ウエストまわり",
    "leg": "脚の部分（パンツ）",
    "crotch": "股まわり（パンツ）",
    "inside": "裏地・内側",
    "unknown": "場所が分からない・傷や汚れが写っていない",
}
QUESTIONS = {
    **{
        kind: {"type": "noul", "instructions": f"写真のどれかに、{text}"}
        for kind, (text, _) in KINDS.items()
    },
    "location": {
        "type": "choice",
        "instructions": "いちばん目立つ傷・汚れは、商品のどこにあるか",
        "criteria": LOCATIONS,
    },
    "noticeable": {
        "type": "noul",
        "instructions": (
            "いちばん目立つ傷・汚れは、商品全体の写真でも分かるくらい目立つか"
            "（アップでやっと分かる程度なら no）"
        ),
    },
}
# 状態ランクごとに、傷・汚れとして書く確率のしきい値（SS・S は新品同様なので、確かなものだけ）
THRESHOLD = {"SS": 0.85, "S": 0.8, "A": 0.6, "B": 0.5, "C": 0.45}
NONE_TEXT = "目立った傷や汚れは見受けられません。"
# 場所を書くことば（「左袖にシミがあります」）
LOCATION_TEXT = {k: v.split("（")[0] for k, v in LOCATIONS.items() if k != "unknown"}
# 写真の区分の、優先する順（傷・汚れのアップを先に）
LABEL_ORDER = ("傷", "汚れ", "ダメージ", "アップ", "背面", "正面")


def pick_images(images: list[dict]) -> list[dict]:
    def rank(image: dict) -> int:
        label = image.get("label") or ""
        for i, word in enumerate(LABEL_ORDER):
            if word in label:
                return i
        return len(LABEL_ORDER)

    return sorted(images, key=rank)[:MAX_IMAGES]


def compose(answers: dict, condition: str | None) -> dict:
    """Clef の答えから、状態メモの候補の文と findings を作る。"""
    threshold = THRESHOLD.get((condition or "").upper(), 0.6)
    probs = {kind: float(answers[kind]["noul"]) for kind in KINDS}
    found = sorted(
        ((kind, p) for kind, p in probs.items() if p >= threshold), key=lambda kp: -kp[1]
    )
    location = answers["location"]
    location_key = location.get("choice") if isinstance(location, dict) else location
    location_confidence = float(location.get("confidence", 0)) if isinstance(location, dict) else 0
    noticeable = float(answers["noticeable"]["noul"])

    findings = [
        {
            "kind": kind,
            "confidence": round(p, 3),
            "location": LOCATION_TEXT.get(location_key) if i == 0 else None,
            "severity": ("noticeable" if noticeable >= 0.5 else "small") if i == 0 else None,
        }
        for i, (kind, p) in enumerate(found)
    ]
    if not found:
        # B・C（ダメージのあるランク）なのに何も見つからないときは、候補を入れない
        if (condition or "").upper() in ("B", "C"):
            return {"draft": None, "findings": [], "probs": probs, "reason": "rank_needs_damage"}
        return {"draft": NONE_TEXT, "findings": [], "probs": probs}

    first_kind = found[0][0]
    size = "目立つ" if noticeable >= 0.5 else "小さな"
    where = LOCATION_TEXT.get(location_key) if location_confidence >= 0.4 else None
    sentences = [
        f"{where}に{size}{KINDS[first_kind][1]}があります。"
        if where
        else f"{size}{KINDS[first_kind][1]}があります。"
    ]
    others = [KINDS[kind][1] for kind, _ in found[1:3]]
    if others:
        sentences.append(f"ほかに{'・'.join(others)}も見られます。")
    return {"draft": "".join(sentences), "findings": findings, "probs": probs}


def judge(model, client: shopify.Shopify, product: dict) -> dict:
    images = pick_images(product["images"])
    pics = []
    for image in images:
        sep = "&" if "?" in image["url"] else "?"
        pics.append(_for_clef(client.download(f"{image['url']}{sep}width={WIDTH}")))
    # 小さな傷は、写真をまとめて聞くと埋もれるので、1 枚ずつ聞いていちばん確かな答えを取る
    best: dict = {}
    best_image: dict = {}
    for image, pic in zip(images, pics, strict=True):
        answers = model.ask(STATE, QUESTIONS, [pic])
        for key, value in answers.items():
            score = value.get("noul") if "noul" in value else value.get("confidence", 0)
            if key not in best or score > (
                best[key].get("noul") if "noul" in best[key] else best[key].get("confidence", 0)
            ):
                best[key] = value
                best_image[key] = image["url"]
    result = compose(best, product.get("condition"))
    for f in result["findings"]:
        f["imageUrl"] = best_image.get(f["kind"])
    return {
        "id": product.get("id"),
        "shopifyProductId": product.get("shopifyProductId"),
        "title": product["title"],
        "condition": product.get("condition"),
        "images": [i["url"] for i in images],
        **result,
    }


def targets_from_cms(http: httpx.Client, limit: int) -> list[dict]:
    res = http.get(f"{CMS_URL}/api/products/condition-drafts/targets", params={"limit": limit})
    res.raise_for_status()
    return res.json()["docs"]


def target_from_shopify(client: shopify.Shopify, handle: str) -> dict:
    data = client.graphql(
        """query($q: String) { products(first: 1, query: $q) { nodes { id title
          condition: metafield(namespace: "custom", key: "condition") { value }
          media(first: 12) { nodes { ... on MediaImage { image { url } alt } } } } } }""",
        {"q": f"handle:{handle}"},
    )["products"]["nodes"]
    if not data:
        raise SystemExit(f"商品がありません: {handle}")
    p = data[0]
    return {
        "shopifyProductId": p["id"],
        "title": p["title"],
        "condition": (p.get("condition") or {}).get("value"),
        "images": [
            {"url": m["image"]["url"], "label": m.get("alt")}
            for m in p["media"]["nodes"]
            if m.get("image")
        ],
    }


def send(http: httpx.Client, results: list[dict], apply: bool) -> None:
    items = [
        {
            **({"id": r["id"]} if r.get("id") else {"shopifyProductId": r["shopifyProductId"]}),
            "draft": r["draft"],
            "findings": r["findings"],
            "source": "clef",
        }
        for r in results
        if r.get("draft")
    ]
    print(f"CMS に送る候補: {len(items)} / {len(results)}" + ("" if apply else "（確認だけ）"))
    if not apply or not items:
        return
    res = http.post(f"{CMS_URL}/api/products/condition-drafts", json={"items": items})
    res.raise_for_status()
    body = res.json()
    print(f"入れた: {body.get('updated')}  飛ばした: {len(body.get('skipped') or [])}")
    for s in body.get("skipped") or []:
        print(f"  飛ばした {s.get('ref')}: {s.get('reason')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50, help="CMS から読む商品の数")
    parser.add_argument(
        "--product", action="append", help="CMS を通さず、この商品だけ試す（ハンドル）"
    )
    parser.add_argument("--apply", action="store_true", help="候補を CMS に送る")
    parser.add_argument(
        "--from-results",
        action="store_true",
        help="判定し直さず、保存した結果を送る（--apply と使う）",
    )
    args = parser.parse_args()

    http = httpx.Client(timeout=60, follow_redirects=True)
    if args.from_results:
        send(http, json.loads(OUT.read_text()), args.apply)
        return

    client = shopify.Shopify(
        settings.shopify_store, settings.shopify_client_id, settings.shopify_client_secret
    )
    if args.product:
        targets = [target_from_shopify(client, h) for h in args.product]
    else:
        targets = targets_from_cms(http, args.limit)
    print(f"対象: {len(targets)} 件", flush=True)

    model = clef.from_settings(settings)
    results = []
    for product in targets:
        if not product.get("images"):
            continue
        r = judge(model, client, product)
        results.append(r)
        top = " ".join(f"{k}={v:.2f}" for k, v in r["probs"].items())
        skipped = f"（入れない: {r.get('reason', '')}）"
        print(
            f"  [{r['condition'] or '-'}] {r['draft'] or skipped}  {top}  {r['title'][:40]}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"保存: {OUT}")
    if not args.product:
        send(http, results, args.apply)


if __name__ == "__main__":
    main()
