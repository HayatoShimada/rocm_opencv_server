"""85store-cms の「写真から下書きを作る」（撮影のまとまり）を、Clef で商品ごとに分ける。

人が何着分かの写真を CMS にまとめて上げ、「Clef に分けてもらう」を押すと「分け待ち」になる。
このスクリプトがそれを取り、写真ごとに区分（正面・背面・タグのアップ…）を判定して、
商品ごとに分けた案を返す。人が CMS の画面で案を確かめて直し、商品ごとに保留の下書きを作る。

  uv run --env-file .env --group clip --group clef-local \\
      python -m scripts.cms_photo_batches       # 分け待ちを 1 つ取って処理する（無ければ終わる）
  uv run ... python -m scripts.cms_photo_batches --dry-run   # 取らずに判定し、案を表示するだけ
  uv run ... python -m scripts.cms_photo_batches --eval-brands 40   # ブランドの当たり方を測る

- 1 分ごとに systemd のタイマー（deploy/cms-photo-batches.*）で起こす。
  分け待ちが無ければ Clef を読み込まずに終わる。GPU の空きが足りない（snoo などが使う）ときは、
  その回は見送る（まとまりは分け待ちのまま）
- CMS の API（src/photo-intake/batch-endpoints.ts）: GET /api/photoBatches/queue →
  POST /:id/claim → POST /:id/result（失敗は /:id/error）。
  写真は CMS の縮小版（/api/photoSources/:id/preview。DNG も JPEG で来る）
- 分け方: 撮影時刻の順に並べ、次のどちらかで新しい商品にする
  1. 人が写っていない全体の正面（Clef: whole・view=front が確か）の写真が、すでに正面のある
     商品のあとに来て、その正面と別の服だった（同じ服かは 2 枚を並べて Clef に聞く。
     撮り直しは同じ商品）
  2. 前の写真から GAP_MINUTES 分以上あいた
  新しい商品が正面で始まったら、前の商品の終わりのアップの写真を 1 枚ずつ、前と次のどちらの正面と
  同じ服かを聞き、次の服なら移す（正面より先にタグなどを撮ったとき）
- 品目は CMS が渡す一覧（85crm の品目）から、全体の正面の写真で選ばせる
- 状態メモの候補は scripts/shopify_condition_drafts.py と同じ質問・文の型
  （状態ランクはまだ無いので既定のしきい値）
- ブランドは --brand のときだけ（登録されているブランドから選ばせる。
  当たり方を --eval-brands で測ってから使う）
- CMS へは PC の端末として tailnet の whois で通る（トークンは要らない）
"""

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import httpx2 as httpx

from app import clef, processing, shopify
from app.config import settings
from scripts import shopify_condition_drafts as condition

CMS_URL = os.getenv("CMS_URL", "https://cms.85-store.com").rstrip("/")
GAP_MINUTES = 5
# 区分を判定するときの大きさ（scripts/shopify_classify_photos.py と同じ）
CLASSIFY_WIDTH = 512
# 正面とみなす確からしさ（app/shopify.py の MIN_VIEW_CONFIDENCE と同じ）
FRONT_CONFIDENCE = 0.6
# 品目・ブランドを入れる確からしさ
PRODUCT_TYPE_CONFIDENCE = 0.4
BRAND_CONFIDENCE = 0.6
# 2 枚が同じ服とみなす確からしさ（2026-11-14 の 112 枚で、同じ服 0.91 以上・別の服 0.22 以下）
SAME_GARMENT = 0.5
SAME_STATE = "古着屋の商品撮影で、続けて撮った2枚の写真を見比べる"
SAME_QUESTIONS = {
    "same": {
        "type": "noul",
        "instructions": (
            "2枚の写真に写っている服は、同じ1着か（向き・寄り方・明るさの違いは気にしない。"
            "色・柄・形・ディテールが同じなら yes。似ているが別の服なら no）"
        ),
    }
}
# Clef（BF16）が使う VRAM の目安。空きがこれより少なければ見送る
NEED_VRAM_BYTES = 21 * 1024**3
OUT = Path("data/photo-batches")


@dataclass
class Photo:
    id: int
    filename: str
    taken_at: datetime | None
    image: bytes = b""
    # Clef に渡す小さい JPEG（区分の判定・同じ服かの比べに使う）
    small: bytes = b""
    kind: str = "closeup"
    view: str = "other"
    view_confidence: float = 0.0
    part: str = "other"
    part_confidence: float = 0.0
    label: str = "ディテール"

    @property
    def main_front(self) -> bool:
        """人が写っていない全体の正面（商品の代表になる写真）。"""
        return (
            self.kind == "whole"
            and self.view == "front"
            and self.view_confidence >= FRONT_CONFIDENCE
        )


@dataclass
class Group:
    photos: list[Photo] = field(default_factory=list)

    @property
    def front(self) -> Photo | None:
        return next((p for p in self.photos if p.main_front), None)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


SameFn = Callable[[Photo, Photo], float]


def split_groups(
    photos: list[Photo], same: SameFn, gap_minutes: float = GAP_MINUTES
) -> list[Group]:
    """撮影の順に並んだ写真を、商品ごとに分ける（1 着ずつ続けて撮っている前提）。

    same(a, b) は 2 枚が同じ服の確からしさ（Clef に聞く。テストでは偽物）。
    """
    groups: list[Group] = []
    last: datetime | None = None
    for photo in photos:
        gap = (
            photo.taken_at is not None
            and last is not None
            and (photo.taken_at - last).total_seconds() >= gap_minutes * 60
        )
        current = groups[-1] if groups else None
        other_front = (
            current is not None
            and photo.main_front
            and current.front is not None
            and same(current.front, photo) < SAME_GARMENT
        )
        if current is None or gap or other_front:
            groups.append(Group())
            if other_front and current is not None:
                pull_back(current, groups[-1], photo, same)
        groups[-1].photos.append(photo)
        if photo.taken_at is not None:
            last = photo.taken_at
    return groups


def pull_back(prev: Group, new: Group, front: Photo, same: SameFn) -> None:
    """前の商品の終わりのアップが次の服（front）のものなら、次の商品へ移す（後ろから 1 枚ずつ）。"""
    reference = prev.front
    while reference is not None and len(prev.photos) > 1 and prev.photos[-1].kind == "closeup":
        photo = prev.photos[-1]
        if same(photo, front) <= same(photo, reference):
            break
        new.photos.insert(0, prev.photos.pop())


def group_confidence(group: Group) -> float:
    """分け方の確からしさ。全体の正面がちょうど 1 枚あれば高い（画面は 0.6 未満を目立たせる）。"""
    front = group.front
    if front is None:
        return 0.4
    return 0.9 if front.view_confidence >= 0.8 else 0.7


def downscale(data: bytes, width: int) -> bytes:
    image = processing.decode(data)
    h, w = image.shape[:2]
    if w > width:
        image = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", image[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def label_photo(model, photo: Photo) -> None:
    photo.small = downscale(photo.image, CLASSIFY_WIDTH)
    labels = model.classify(photo.small, "image/jpeg")
    photo.kind = labels.kind
    photo.view, photo.view_confidence = labels.view, labels.view_confidence
    photo.part, photo.part_confidence = labels.part, labels.part_confidence
    photo.label = shopify.photo_label(
        labels.kind, labels.view, labels.view_confidence, labels.part, labels.part_confidence
    )


def ask_choice(model, state: str, instructions: str, options: list[str], images: list[bytes]):
    """選択肢から 1 つ選ばせる。選択肢の名前（& や空白を含む）はキーにせず、番号で聞く。"""
    criteria = {f"o{i}": name for i, name in enumerate(options)}
    answers = model.ask(
        state,
        {"choice": {"type": "choice", "instructions": instructions, "criteria": criteria}},
        [(img, "image/jpeg") for img in images],
    )
    answer = answers["choice"]
    key = answer.get("choice")
    index = int(key[1:]) if isinstance(key, str) and key[1:].isdigit() else -1
    return (options[index] if 0 <= index < len(options) else None), float(
        answer.get("confidence", 0)
    )


def judge_product_type(model, group: Group, product_types: list[str]):
    main = (
        group.front or next((p for p in group.photos if p.kind == "whole"), None) or group.photos[0]
    )
    return ask_choice(
        model,
        "古着屋のオンラインストアの商品写真から、商品の品目を選ぶ",
        "この商品の品目（服の種類）",
        product_types,
        [main.small or downscale(main.image, CLASSIFY_WIDTH)],
    )


NO_BRAND = "（どれでもない・読めない）"


def judge_brand(model, group: Group, brands: list[dict]):
    tags = [p for p in group.photos if p.part == "tag" and p.kind == "closeup"]
    if not tags:
        return None, 0.0
    names = [b["name"] for b in brands] + [NO_BRAND]
    name, confidence = ask_choice(
        model,
        "古着屋の商品写真の、服のタグ（ラベル）に書かれたブランド名を読む",
        "タグに書かれているブランド",
        names,
        [downscale(p.image, 1024) for p in tags[:2]],
    )
    brand = next((b for b in brands if b["name"] == name), None)
    return (brand["id"] if brand else None), confidence


def judge_condition(model, group: Group) -> dict:
    """shopify_condition_drafts.py と同じ聞き方（1 枚ずつ聞き、いちばん確かな答えを取る）。"""
    images = condition.pick_images([{"label": p.label, "photo": p} for p in group.photos])
    best: dict = {}

    def score(value: dict) -> float:
        return value.get("noul") if "noul" in value else value.get("confidence", 0)

    for image in images:
        pic = downscale(image["photo"].image, condition.WIDTH)
        answers = model.ask(condition.STATE, condition.QUESTIONS, [(pic, "image/jpeg")])
        for key, value in answers.items():
            if key not in best or score(value) > score(best[key]):
                best[key] = value
    if not best:
        return {"draft": None, "findings": []}
    return condition.compose(best, None)


def vram_free() -> int | None:
    """GPU の VRAM の空き（バイト）。分からなければ None。"""
    for card in Path("/sys/class/drm").glob("card*/device"):
        total, used = card / "mem_info_vram_total", card / "mem_info_vram_used"
        if total.exists() and used.exists():
            return int(total.read_text()) - int(used.read_text())
    return None


class Cms:
    def __init__(self, http: httpx.Client | None = None):
        self.http = http or httpx.Client(timeout=120, follow_redirects=True)

    def queue(self) -> dict:
        res = self.http.get(f"{CMS_URL}/api/photoBatches/queue", params={"limit": 1})
        res.raise_for_status()
        return res.json()

    def claim(self, batch_id: int) -> bool:
        res = self.http.post(f"{CMS_URL}/api/photoBatches/{batch_id}/claim")
        if res.status_code == 409:
            return False
        res.raise_for_status()
        return True

    def preview(self, path: str) -> bytes:
        res = self.http.get(f"{CMS_URL}{path}")
        res.raise_for_status()
        return res.content

    def result(self, batch_id: int, groups: list[dict]) -> dict:
        res = self.http.post(
            f"{CMS_URL}/api/photoBatches/{batch_id}/result", json={"groups": groups}
        )
        res.raise_for_status()
        return res.json()

    def error(self, batch_id: int, message: str) -> None:
        self.http.post(
            f"{CMS_URL}/api/photoBatches/{batch_id}/error", json={"message": message[:2000]}
        )


def build_groups(model, batch: dict, choices: dict, cms: Cms, brand: bool) -> list[dict]:
    photos = [
        Photo(s["id"], s.get("filename") or "", parse_time(s.get("takenAt")))
        for s in batch["sources"]
    ]
    for photo, source in zip(photos, batch["sources"], strict=True):
        photo.image = cms.preview(source["preview"])
        label_photo(model, photo)
        print(
            f"  {photo.filename}: {photo.label}  ({photo.kind} "
            f"{photo.view}={photo.view_confidence:.2f} {photo.part}={photo.part_confidence:.2f})",
            flush=True,
        )

    def same(a: Photo, b: Photo) -> float:
        pics = [(p.small, "image/jpeg") for p in (a, b)]
        return float(model.ask(SAME_STATE, SAME_QUESTIONS, pics)["same"]["noul"])

    out = []
    for i, group in enumerate(split_groups(photos, same), start=1):
        item: dict = {
            "key": f"g{i}",
            "sources": [{"id": p.id, "label": p.label} for p in group.photos],
            "confidence": group_confidence(group),
        }
        if group.front is None:
            item["note"] = "全体の正面の写真が見つかりませんでした"
        if choices.get("productTypes"):
            product_type, confidence = judge_product_type(model, group, choices["productTypes"])
            item["productTypeConfidence"] = round(confidence, 3)
            if product_type and confidence >= PRODUCT_TYPE_CONFIDENCE:
                item["productType"] = product_type
        if brand and choices.get("brands"):
            brand_id, confidence = judge_brand(model, group, choices["brands"])
            item["brandConfidence"] = round(confidence, 3)
            if brand_id and confidence >= BRAND_CONFIDENCE:
                item["brand"] = brand_id
        cond = judge_condition(model, group)
        if cond.get("draft"):
            item["conditionDraft"] = cond["draft"]
            item["findings"] = cond.get("findings", [])
        print(
            f"#{i} {len(group.photos)}枚  正面={'あり' if group.front else 'なし'}  "
            f"品目={item.get('productType', '-')}({item.get('productTypeConfidence', 0):.2f})  "
            f"状態={item.get('conditionDraft', '-')}",
            flush=True,
        )
        out.append(item)
    return out


def run_once(dry_run: bool, brand: bool) -> int:
    cms = Cms()
    queue = cms.queue()
    if not queue.get("docs"):
        return 0
    batch = queue["docs"][0]
    print(f"まとまり #{batch['id']} {batch.get('title')}（{len(batch['sources'])}枚）", flush=True)
    free = vram_free()
    if free is not None and free < NEED_VRAM_BYTES:
        print(f"GPU の空きが足りないので見送ります（空き {free / 1024**3:.1f}GB）", flush=True)
        return 0
    if not dry_run and not cms.claim(batch["id"]):
        print("ほかが取ったので見送ります", flush=True)
        return 0
    try:
        model = clef.from_settings(settings)
        groups = build_groups(model, batch, queue, cms, brand)
    except Exception as e:  # noqa: BLE001 失敗は CMS に知らせて、人が自分で分けられるようにする
        if not dry_run:
            cms.error(batch["id"], f"Clef で分けられませんでした: {e}")
        raise
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"batch-{batch['id']}.json").write_text(json.dumps(groups, ensure_ascii=False, indent=1))
    if dry_run:
        print(f"{len(groups)}商品に分けました（確認だけ。CMS には送っていません）")
        return 0
    res = cms.result(batch["id"], groups)
    print(f"{res.get('groups')}商品に分けて、CMS に返しました", flush=True)
    return 0


def eval_brands(limit: int) -> int:
    """今ある商品（ブランドとタグのアップの写真があるもの）で、ブランドの当たり方を測る。"""
    http = httpx.Client(timeout=120, follow_redirects=True)
    brands = http.get(f"{CMS_URL}/api/brands", params={"limit": 2000, "depth": 0}).json()["docs"]
    res = http.get(
        f"{CMS_URL}/api/products",
        params={
            "limit": 1000,
            "depth": 0,
            "where[brand][exists]": "true",
            "where[status][equals]": "active",
        },
    ).json()["docs"]
    targets = []
    for p in res:
        tags = [
            i
            for i in p.get("images") or []
            if i.get("autoAltLabel") == "タグのアップ" and i.get("shopifyUrl")
        ]
        if tags:
            targets.append((p, tags))
        if len(targets) >= limit:
            break
    print(f"対象: {len(targets)} 件（ブランド {len(brands)} 件から選ぶ）", flush=True)
    model = clef.from_settings(settings)
    hit = picked = 0
    for p, tags in targets:
        photos = []
        for t in tags[:2]:
            sep = "&" if "?" in t["shopifyUrl"] else "?"
            data = http.get(f"{t['shopifyUrl']}{sep}width=1024").content
            photos.append(Photo(0, "", None, image=data, kind="closeup", part="tag"))
        brand_id, confidence = judge_brand(model, Group(photos), brands)
        ok = brand_id == p["brand"]
        if confidence >= BRAND_CONFIDENCE and brand_id:
            picked += 1
            hit += ok
        name = next((b["name"] for b in brands if b["id"] == brand_id), "-")
        truth = next((b["name"] for b in brands if b["id"] == p["brand"]), "?")
        print(
            f"  {'○' if ok else '×'} {confidence:.2f} 判定={name}  正解={truth}  {p['title'][:40]}",
            flush=True,
        )
    print(
        f"入れる（確からしさ {BRAND_CONFIDENCE} 以上）: {picked}/{len(targets)}"
        f"  そのうち正解: {hit}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="取らずに判定だけして表示する（CMS は変えない）"
    )
    parser.add_argument(
        "--brand", action="store_true", help="タグの写真からブランドも選ぶ（測ってから使う）"
    )
    parser.add_argument(
        "--eval-brands", type=int, metavar="N", help="今ある商品 N 件で、ブランドの当たり方を測る"
    )
    args = parser.parse_args()
    if args.eval_brands:
        return eval_brands(args.eval_brands)
    return run_once(args.dry_run, args.brand)


if __name__ == "__main__":
    sys.exit(main())
