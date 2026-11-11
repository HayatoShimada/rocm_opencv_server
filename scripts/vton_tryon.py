"""着用イメージ（AI の合成）を、服の実寸どおりの大きさで作る PoC。Shopify・CMS には書き込まない。

  uv run --env-file .env --group clip --group clef-local --group vton \\
      python -m scripts.vton_tryon --handles <ハンドル>,<ハンドル> \\
      [--stage all|prepare|generate|check]

85-Store の原則（タグではなく、測った尺度で服を見せる）に合わせて、AI に服の大きさを決めさせない。
1. prepare: 人物の写真から骨格の点（ViTPose）と人の形（SAM2）を取り、身長から px/cm を出す。
   服の実寸（custom.measurements）から、首の付け根・肩の縫い目・袖口・裾の位置を決める
   （app/garment_scale.py）。
   ハンガーに掛けた服の正面の写真から服の形（SAM2）を切り抜き、着丈から服の写真の px/cm を出して、
   人物と同じ縮尺で首の付け根に合わせて貼る（下敷き）。ここで服の大きさと丈が実寸どおりに決まる
   （灰色の範囲だけを渡す方法は、AI が灰色のまま残したので、やめた。2026-11-11）
2. generate: 下敷きと服の写真を Qwen-Image-Edit に渡し、
   貼った服を自然に着ている見た目にだけ直させる。
   大きさ・丈・柄は変えさせない。範囲の外は元の写真に戻す（顔・手・背景は変えない）
3. check: できた画像の服の形（SAM2）から、裾と袖口の位置を測りなおし、
   実寸から決めた位置とのずれ（cm）を出す。
   TOLERANCE_CM（2cm）以内のものだけを「使える」とする

人物は data/vton/people/people.json に書く（写真を使う同意を取った人だけ）:
  {"sample": {"file": "sample.png", "height": 175, "shoulder": 44}}
出力は data/vton/<人物>/<ハンドル>/（guide.png・mask.png・out.jpg）と data/vton/results.json。
GPU は 24GB なので、段階ごとにモデルを入れ替える。Clef・snoo と同時に動かさない。
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app import garment_scale as gs
from app import shopify
from app.config import settings

ROOT = Path("data/vton")
PEOPLE = ROOT / "people" / "people.json"
RESULTS = ROOT / "results.json"

PRODUCT_QUERY = """
query($q: String!) {
  products(first: 1, query: $q) {
    nodes {
      handle title productType
      measurements: metafield(namespace: "custom", key: "measurements") { value }
      media(first: 20) { nodes { ... on MediaImage { alt image { url width height } } } }
    }
  }
}
"""


def load_people(path: Path = PEOPLE) -> dict[str, dict]:
    people = json.loads(path.read_text())
    for name, p in people.items():
        p["path"] = str(path.parent / p["file"])
        if not p.get("height"):
            raise SystemExit(f"{name}: 身長（height）がない")
    return people


def front_image(product: dict) -> dict | None:
    """正面の写真（alt に「（正面）」がある画像。無ければ 1 枚目）。"""
    images = [n for n in product["media"]["nodes"] if n.get("image")]
    for n in images:
        if "（正面）" in (n.get("alt") or ""):
            return n
    return images[0] if images else None


def fetch_product(client: shopify.Shopify, handle: str) -> dict | None:
    nodes = client.graphql(PRODUCT_QUERY, {"q": f"handle:{handle}"})["products"]["nodes"]
    if not nodes:
        return None
    p = nodes[0]
    raw = (p.get("measurements") or {}).get("value")
    p["measurements"] = json.loads(raw) if raw else {}
    return p


def body_width_at(mask: np.ndarray, y: int, cx: int) -> int | None:
    """人の形の、高さ y で中心 cx を含む連続した幅（px）。"""
    row = mask[int(np.clip(y, 0, mask.shape[0] - 1))]
    if not row[cx]:
        return None
    left = cx
    while left > 0 and row[left - 1]:
        left -= 1
    right = cx
    while right < row.size - 1 and row[right + 1]:
        right += 1
    return right - left + 1


def garment_shape(seg, garment: Image.Image) -> np.ndarray:
    """ハンガーに掛けた服の写真から、服の形（ハンガーを除く）を取る。

    中心を正、四隅（背景の板）とフックのあたりを負にして指す。そのうえで、形のいちばん上の点（フック）から
    ハンガーの形を取り、服の形から引く（ハンガーが入ると、着丈から出す縮尺がずれる）。
    """
    w, h = garment.size
    corners = [
        (w * 0.04, h * 0.04),
        (w * 0.96, h * 0.04),
        (w * 0.04, h * 0.96),
        (w * 0.96, h * 0.96),
    ]
    points = [(w / 2, h * 0.5), (w / 2, h * 0.03), *corners]
    mask = seg(garment, points=points, labels=[1, 0, 0, 0, 0, 0])
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return mask
    top = int(rows[0])
    hook = (float(np.flatnonzero(mask[top]).mean()), float(top + 2))
    hanger = seg(garment, points=[hook], labels=[1])
    if 0 < hanger.sum() < 0.2 * mask.sum():
        mask &= ~hanger
    return largest_blob(mask, k=max(3, int(w * 0.02)))


def largest_blob(mask: np.ndarray, k: int) -> np.ndarray:
    """細い物（掛けるパイプ・フック）をオープニングで消し、いちばん大きいかたまりだけを残す。"""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if n <= 1:
        return opened.astype(bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def out_dir(person: str, handle: str) -> Path:
    return ROOT / person / handle


def prepare(handles: list[str], people: dict[str, dict]) -> list[dict]:
    from app import vton

    client = shopify.Shopify(
        settings.shopify_store,
        client_id=settings.shopify_client_id,
        client_secret=settings.shopify_client_secret,
        admin_token=settings.shopify_admin_token,
    )
    pose, seg = vton.Pose(), vton.Segmenter()
    jobs: list[dict] = []
    for name, person in people.items():
        img = Image.open(person["path"]).convert("RGB")
        kp, scores = pose(img)
        xs, ys = kp[:, 0], kp[:, 1]
        pad = 0.15 * (ys.max() - ys.min())
        box = (xs.min() - pad, ys.min() - pad * 1.5, xs.max() + pad, ys.max() + pad)
        positives = [tuple(kp[i]) for i in (gs.L_SHOULDER, gs.R_SHOULDER, gs.L_HIP, gs.R_HIP)]
        pmask = seg(img, points=positives, box=box)
        scale = gs.person_scale(pmask, person["height"])
        mid = (kp[gs.L_SHOULDER] + kp[gs.R_SHOULDER]) / 2
        body_px = body_width_at(pmask, int(mid[1]), int(mid[0]))
        if person.get("shoulder"):
            body_px = person["shoulder"] * scale.px_per_cm
        for handle in handles:
            product = fetch_product(client, handle)
            if not product:
                print(f"{handle}: 商品が見つからない")
                continue
            meas = product["measurements"]
            if meas.get("sizes"):
                print(f"{handle}: サイズ違いの表は PoC では対象外")
                continue
            lay = gs.layout(kp, scale, person["height"], meas, body_shoulder_px=body_px)
            if lay.hem_y is None or lay.chest_half_px is None:
                print(f"{handle}: 着丈か身幅がない")
                continue
            front = front_image(product)
            if not front:
                print(f"{handle}: 写真がない")
                continue
            d = out_dir(name, handle)
            d.mkdir(parents=True, exist_ok=True)
            garment = Image.open(io.BytesIO(client.download(front["image"]["url"]))).convert("RGB")
            garment.thumbnail((1600, 1600))
            garment.save(d / "garment.jpg", quality=92)
            gmask = garment_shape(seg, garment)
            Image.fromarray((gmask * 255).astype(np.uint8)).save(d / "garment_shape.png")
            g = gs.garment_scale(gmask, gs.first(meas, "着丈", "着丈(後)"))
            k, at = gs.paste_plan(lay, g)
            by_width = gs.width_scale(gmask, gs.first(meas, "身幅"))
            if by_width and abs(by_width / g.px_per_cm - 1) > 0.15:
                gap = by_width / g.px_per_cm - 1
                lay.notes.append(
                    f"服の写真の縮尺が、着丈からと身幅からで {gap:+.0%} ちがう"
                    "（ハンガーで形が変わっている）。丈は着丈に合わせた。平置きの写真なら正確になる"
                )
            guide, pasted = vton.collage(img, garment, gmask, k, at)
            guide.save(d / "guide.png")
            pasted.save(d / "mask.png")
            jobs.append(
                {
                    "person": name,
                    "handle": handle,
                    "title": product["title"],
                    "type": product.get("productType"),
                    "measurements": meas,
                    "person_meta": {k: person.get(k) for k in ("height", "shoulder", "chest")},
                    "px_per_cm": round(scale.px_per_cm, 3),
                    "keypoints": kp.round(1).tolist(),
                    "pose_scores": scores.round(3).tolist(),
                    "notes": lay.notes,
                    "paste": {
                        "by_width_px_per_cm": round(by_width, 3) if by_width else None,
                        "scale": round(k, 4),
                        "at": list(at),
                        "garment_px_per_cm": round(g.px_per_cm, 3),
                    },
                    "layout": {
                        "neck": lay.neck.round(1).tolist(),
                        "hem_y": round(lay.hem_y, 1),
                        "chest_half_px": round(lay.chest_half_px, 1),
                        "seams": {k: v.round(1).tolist() for k, v in lay.seams.items()},
                        "cuffs": {k: v.round(1).tolist() for k, v in lay.cuffs.items()},
                    },
                }
            )
            print(f"{name} × {handle}: 下敷きを作った（{scale.px_per_cm:.2f} px/cm）")
            for note in lay.notes:
                print(f"  注意: {note}")
    del pose, seg
    vton.free_gpu()
    return jobs


def generate(jobs: list[dict], people: dict[str, dict], seed: int) -> None:
    from app import vton

    t0 = time.time()
    model = vton.TryOnModel.load()
    load_s = round(time.time() - t0, 1)
    for job in jobs:
        d = out_dir(job["person"], job["handle"])
        original = Image.open(people[job["person"]]["path"]).convert("RGB")
        guide = Image.open(d / "guide.png").convert("RGB")
        garment = Image.open(d / "garment.jpg").convert("RGB")
        t = time.time()
        raw = model.refine(guide, garment, seed=seed)
        raw.save(d / "raw.jpg", quality=92)
        grow = int(job["px_per_cm"] * 3)
        out = vton.composite(original, raw, Image.open(d / "mask.png"), grow)
        out.save(d / "out.jpg", quality=92)
        job["generate"] = {
            "seconds": round(time.time() - t, 1),
            "load_seconds": load_s,
            "seed": seed,
        }
        job["generate"]["vram_gb"] = vton.vram_gb()
        print(f"{job['person']} × {job['handle']}: {job['generate']['seconds']} 秒")
    del model
    vton.free_gpu()


def rebuild_layout(job: dict, people: dict[str, dict]) -> gs.Layout:
    person = people[job["person"]]
    kp = np.array(job["keypoints"])
    s = job["px_per_cm"]
    scale = gs.PersonScale(px_per_cm=s, top_y=0, floor_y=0)
    body_px = (person.get("shoulder") or 0) * s or None
    return gs.layout(kp, scale, person["height"], job["measurements"], body_shoulder_px=body_px)


def check(jobs: list[dict], people: dict[str, dict]) -> None:
    from app import vton

    seg = vton.Segmenter()
    for job in jobs:
        d = out_dir(job["person"], job["handle"])
        if not (d / "out.jpg").exists():
            continue
        lay = rebuild_layout(job, people)
        img = Image.open(d / "out.jpg").convert("RGB")
        kp = np.array(job["keypoints"])
        cx = float(lay.neck[0])
        chest_y = float(lay.seams["left"][1]) + lay.chest_half_px * 0.6
        positives = [(cx, chest_y), (cx, (chest_y + lay.hem_y) / 2)]
        negatives = [tuple(kp[gs.NOSE]), tuple(kp[gs.L_ANKLE]), tuple(kp[gs.R_ANKLE])]
        gmask = seg(img, points=positives + negatives, labels=[1, 1, 0, 0, 0])
        Image.fromarray((gmask * 255).astype(np.uint8)).save(d / "garment_mask.png")
        dev = gs.deviations(lay, gs.measure_garment(gmask, lay))
        job["check"] = {
            "deviations_cm": dev,
            "passes": gs.passes(dev),
            "caption": gs.caption_numbers(job["person_meta"], lay, dev),
        }
        ok = "使える" if job["check"]["passes"] else "使わない"
        print(f"{job['person']} × {job['handle']}: ずれ {dev} → {ok}")
    del seg
    vton.free_gpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--handles", required=True, help="カンマ区切りのハンドル")
    ap.add_argument("--stage", choices=["all", "prepare", "generate", "check"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    handles = [h.strip() for h in args.handles.split(",") if h.strip()]
    people = load_people()

    jobs = json.loads(RESULTS.read_text()) if RESULTS.exists() and args.stage != "all" else []
    if args.stage in ("all", "prepare"):
        jobs = prepare(handles, people)
    if args.stage in ("all", "generate"):
        generate([j for j in jobs if j["handle"] in handles], people, args.seed)
    if args.stage in ("all", "check"):
        check([j for j in jobs if j["handle"] in handles], people)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(jobs, ensure_ascii=False, indent=2))
    print(f"{RESULTS} に書いた（{len(jobs)} 件）")


if __name__ == "__main__":
    main()
