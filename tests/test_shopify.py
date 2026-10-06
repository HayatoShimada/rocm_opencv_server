import json

import cv2
import httpx2 as httpx
import numpy as np
import pytest

from app import shopify

STORE = "test.myshopify.com"


def _jpeg(width: int, height: int) -> bytes:
    return cv2.imencode(".jpg", np.full((height, width, 3), 128, np.uint8))[1].tobytes()


def _media(media_id: str, width: int, height: int) -> dict:
    url = f"https://cdn.shopify.com/{media_id}.jpg"
    image = {"url": url, "width": width, "height": height}
    return {"id": media_id, "alt": "", "mimeType": "image/jpeg", "image": image}


class FakeShopify:
    """Shopify の Admin API・CDN・staged upload の先を1つにまとめた偽物。"""

    def __init__(self, media: list[dict]):
        self.media = {m["id"]: m for m in media}
        self.uploaded: dict[str, bytes] = {}
        self.updated: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.shopify.com":
            media_id = request.url.path.strip("/").removesuffix(".jpg")
            image = self.media[media_id]["image"]
            return httpx.Response(200, content=_jpeg(image["width"], image["height"]))
        if request.url.host == "upload.example":
            self.uploaded["last"] = request.content
            return httpx.Response(201)
        body = json.loads(request.content)
        query, variables = body["query"], body["variables"]
        if "products(" in query:
            product = {"id": "gid://shopify/Product/1", "handle": "shirt", "title": "シャツ"}
            product["media"] = {"nodes": list(self.media.values())}
            page = {"hasNextPage": False, "endCursor": None}
            data = {"products": {"pageInfo": page, "nodes": [product]}}
        elif "stagedUploadsCreate" in query:
            target = {"url": "https://upload.example/", "resourceUrl": "https://upload.example/r"}
            target["parameters"] = [{"name": "key", "value": "k"}]
            data = {"stagedUploadsCreate": {"stagedTargets": [target], "userErrors": []}}
        elif "fileUpdate" in query:
            self.updated += variables["files"]
            data = {"fileUpdate": {"files": [], "userErrors": []}}
        else:
            image = {"url": "https://cdn.shopify.com/new.jpg", "width": 2048, "height": 1536}
            data = {"node": {"fileStatus": "READY", "image": image}}
        return httpx.Response(200, json={"data": data})


@pytest.fixture
def fake():
    return FakeShopify([_media("big", 4000, 3000), _media("small", 1000, 800)])


@pytest.fixture
def client(fake):
    http = httpx.Client(transport=httpx.MockTransport(fake.handler))
    return shopify.Shopify(STORE, admin_token="token", http=http)


def test_product_search():
    assert shopify.product_search("gid://shopify/Product/123") == "id:123"
    assert shopify.product_search("123") == "id:123"
    assert shopify.product_search("shirt") == 'handle:"shirt"'


def test_dry_run_does_not_upload(client, fake):
    result = shopify.fit_product_images(client, None, 2048, 90, apply=False)
    assert result.checked == 2
    assert [r.media_id for r in result.resized] == ["big"]
    assert fake.updated == []


def test_apply_replaces_only_large_images(client, fake):
    result = shopify.fit_product_images(client, None, 2048, 90, apply=True)
    assert fake.updated == [{"id": "big", "originalSource": "https://upload.example/r"}]
    assert result.alt_filled == 0
    assert result.resized[0].after == (2048, 1536)
    assert result.resized[0].url_after == "https://cdn.shopify.com/new.jpg"
    assert result.errors == []


def test_fit_image_resizes_to_max_side():
    data, mime_type = shopify.fit_image(_jpeg(4000, 3000), "image/jpeg", 2048, 90)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert mime_type == "image/jpeg"
    assert image.shape == (1536, 2048, 3)


def test_fill_alt_before_replacing(client, fake):
    fake.media["small"]["alt"] = "正面"
    result = shopify.fit_product_images(client, None, 2048, 90, apply=True, fill_alt=True)
    assert fake.updated[0] == {"id": "big", "alt": "シャツ（1枚目）"}
    assert fake.updated[1]["id"] == "big" and "originalSource" in fake.updated[1]
    assert result.alt_filled == 1


def test_fit_image_converts_to_srgb(monkeypatch):
    calls = []
    monkeypatch.setattr(
        shopify.processing, "to_srgb", lambda image, icc: calls.append(icc) or image
    )
    monkeypatch.setattr(shopify.processing, "icc_profile", lambda data: b"icc")
    shopify.fit_image(_jpeg(4000, 3000), "image/jpeg", 2048, 90)
    assert calls == [b"icc"]


def test_photo_label_uses_view_and_part_when_confident():
    assert shopify.photo_label("whole", "back", 0.9, "tag", 0.9) == "背面"
    assert shopify.photo_label("worn", "front", 0.9, "other", 0.0) == "着用"
    assert shopify.photo_label("worn", "back", 0.9, "other", 0.0) == "着用・背面"
    assert shopify.photo_label("closeup", "front", 0.9, "tag", 0.5) == "タグのアップ"
    # 確からしさが低い・どれでもないときは決めつけない
    assert shopify.photo_label("whole", "back", 0.4, "other", 0.0) == "全体"
    assert shopify.photo_label("whole", "other", 0.9, "other", 0.0) == "全体"
    assert shopify.photo_label("closeup", "front", 0.9, "tag", 0.2) == "ディテール"
    assert shopify.photo_label("closeup", "front", 0.9, "other", 0.9) == "ディテール"


def test_auto_alt_label_recognizes_automatic_alts():
    title = "[River] Wool Check Pants"
    assert shopify.auto_alt(title, 2) == f"{title}（3枚目）"
    assert shopify.auto_alt(title, 2, "正面") == f"{title}（正面）"
    assert shopify.auto_alt_label(f"{title}（3枚目）", title, 2) == ""
    assert shopify.auto_alt_label(f"{title}（タグのアップ）", title, 2) == "タグのアップ"
    # 人が入れた alt・位置の違う n枚目・別の商品名・一覧にない語は、自動の alt ではない
    assert shopify.auto_alt_label("手で入れた説明", title, 2) is None
    assert shopify.auto_alt_label(f"{title}（2枚目）", title, 2) is None
    assert shopify.auto_alt_label("別の商品（正面）", title, 2) is None
    assert shopify.auto_alt_label(f"{title}（裏地のアップ）", title, 2) is None
