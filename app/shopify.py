"""Shopify の商品画像を、長辺が上限に収まるように縮めて差し替える。

差し替えは fileUpdate の originalSource で行う（MediaImage の ID・並び順・alt はそのまま）。
85store-cms の src/shopify/client.ts・upload.ts と同じ作法で Admin GraphQL を呼ぶ。
"""

import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from urllib.parse import urlparse

import httpx2 as httpx

from app import processing

API_VERSION = "2026-01"
# 1回の問い合わせのコストが上限（1000）を超えないよう、商品は10件ずつ取る（10 × 画像50）
PAGE_SIZE = 10

PRODUCTS_QUERY = """
query($after: String, $query: String, $first: Int!) {
  products(first: $first, after: $after, query: $query) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id handle title
      media(first: 50) {
        nodes { ... on MediaImage { id alt mimeType image { url width height } } }
      }
    }
  }
}
"""

STAGED_UPLOAD = """
mutation($input: [StagedUploadInput!]!) {
  stagedUploadsCreate(input: $input) {
    stagedTargets { url resourceUrl parameters { name value } }
    userErrors { field message }
  }
}
"""

FILE_UPDATE = """
mutation($files: [FileUpdateInput!]!) {
  fileUpdate(files: $files) { files { id fileStatus } userErrors { field message } }
}
"""

MEDIA_STATUS = """
query($id: ID!) { node(id: $id) { ... on MediaImage { fileStatus image { url width height } } } }
"""


class ShopifyError(RuntimeError):
    pass


class Shopify:
    def __init__(
        self,
        store: str,
        client_id: str = "",
        client_secret: str = "",
        admin_token: str = "",
        http: httpx.Client | None = None,
    ):
        if not store or not ((client_id and client_secret) or admin_token):
            raise ShopifyError("SHOPIFY_STORE と SHOPIFY_CLIENT_ID/SECRET を設定してください")
        self.store = store
        self._client_id = client_id
        self._client_secret = client_secret
        self._admin_token = admin_token
        self._http = http or httpx.Client(timeout=120, follow_redirects=True)
        self._cached: tuple[str, float] | None = None

    def _access_token(self) -> str:
        if self._admin_token and not self._client_id:
            return self._admin_token
        if self._cached and time.time() < self._cached[1] - 5 * 60:
            return self._cached[0]
        res = self._http.post(
            f"https://{self.store}/admin/oauth/access_token",
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "grant_type": "client_credentials",
            },
        )
        if res.status_code != 200:
            raise ShopifyError(f"Shopify のトークンを取得できません: {res.status_code}")
        body = res.json()
        self._cached = (body["access_token"], time.time() + body.get("expires_in", 86_399))
        return self._cached[0]

    def graphql(self, query: str, variables: dict | None = None) -> dict:
        url = f"https://{self.store}/admin/api/{API_VERSION}/graphql.json"
        for attempt in range(6):
            res = self._http.post(
                url,
                json={"query": query, "variables": variables or {}},
                headers={"X-Shopify-Access-Token": self._access_token()},
            )
            if res.status_code == 401 and attempt == 0:
                self._cached = None
                continue
            if (res.status_code == 429 or res.status_code >= 500) and attempt < 5:
                time.sleep(2**attempt)
                continue
            if res.status_code != 200:
                raise ShopifyError(f"Shopify API: {res.status_code} {res.text}")
            body = res.json()
            errors = body.get("errors") or []
            throttled = any(e.get("extensions", {}).get("code") == "THROTTLED" for e in errors)
            if throttled and attempt < 5:
                time.sleep(2 * 2**attempt)
                continue
            if errors:
                raise ShopifyError("Shopify API: " + " / ".join(e["message"] for e in errors))
            # 残りのコストが少なくなったら少し待つ（CMS・85crm も同じ API を使うため、使い切らない）
            throttle = body.get("extensions", {}).get("cost", {}).get("throttleStatus")
            if throttle and throttle["currentlyAvailable"] < 200:
                time.sleep(200 / throttle["restoreRate"])
            return body["data"]
        raise ShopifyError("Shopify API: やり直しても応答がありません")

    def products(self, search: str | None = None) -> Iterator[dict]:
        after = None
        while True:
            data = self.graphql(
                PRODUCTS_QUERY, {"after": after, "query": search, "first": PAGE_SIZE}
            )["products"]
            yield from data["nodes"]
            if not data["pageInfo"]["hasNextPage"]:
                return
            after = data["pageInfo"]["endCursor"]

    def download(self, url: str) -> bytes:
        res = self._http.get(url)
        if res.status_code != 200:
            raise ShopifyError(f"画像を取得できません: {res.status_code} {url}")
        return res.content

    def stage_upload(self, data: bytes, filename: str, mime_type: str) -> str:
        res = self.graphql(
            STAGED_UPLOAD,
            {
                "input": [
                    {
                        "resource": "IMAGE",
                        "filename": filename,
                        "mimeType": mime_type,
                        "httpMethod": "POST",
                        "fileSize": str(len(data)),
                    }
                ]
            },
        )["stagedUploadsCreate"]
        _assert_no_user_errors("stagedUploadsCreate", res["userErrors"])
        target = res["stagedTargets"][0]
        form = {p["name"]: p["value"] for p in target["parameters"]}
        upload = self._http.post(
            target["url"], data=form, files={"file": (filename, data, mime_type)}
        )
        if upload.status_code not in (200, 201, 204):
            raise ShopifyError(f"画像のアップロードに失敗しました: {upload.status_code}")
        return target["resourceUrl"]

    def replace_image(self, media_id: str, resource_url: str, timeout: float = 90) -> dict:
        """画像の中身を差し替え、Shopify の処理が終わるのを待って新しい image を返す。"""
        files = [{"id": media_id, "originalSource": resource_url}]
        res = self.graphql(FILE_UPDATE, {"files": files})
        _assert_no_user_errors("fileUpdate", res["fileUpdate"]["userErrors"])
        deadline = time.time() + timeout
        while True:
            node = self.graphql(MEDIA_STATUS, {"id": media_id})["node"]
            if node["fileStatus"] == "READY" and node["image"]:
                return node["image"]
            if node["fileStatus"] == "FAILED":
                raise ShopifyError(f"Shopify で画像の処理に失敗しました: {media_id}")
            if time.time() > deadline:
                raise ShopifyError(f"Shopify の画像の処理が終わりません: {media_id}")
            time.sleep(2)


def _assert_no_user_errors(label: str, errors: list[dict]) -> None:
    if errors:
        detail = " / ".join(f"{'.'.join(e.get('field') or [])} {e['message']}" for e in errors)
        raise ShopifyError(f"{label}: {detail}")


def product_search(identifier: str) -> str:
    """商品の ID（gid か数字）またはハンドルを、products の検索条件にする。"""
    if identifier.startswith("gid://"):
        identifier = identifier.rsplit("/", 1)[-1]
    if identifier.isdigit():
        return f"id:{identifier}"
    return f'handle:"{identifier}"'


@dataclass
class Resized:
    product: str
    media_id: str
    before: tuple[int, int]
    after: tuple[int, int] | None = None
    url_after: str | None = None
    bytes_before: int | None = None
    bytes_after: int | None = None


@dataclass
class FitResult:
    checked: int = 0
    resized: list[Resized] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def fit_image(data: bytes, mime_type: str, max_side: int, quality: int) -> tuple[bytes, str]:
    """画像を縮めてエンコードし直す。透過のある PNG は PNG のまま、ほかは JPEG にする。"""
    if mime_type == "image/png":
        image = processing.decode(data, keep_alpha=True)
        if processing.has_alpha(image):
            return processing.encode(processing.fit_long_side(image, max_side), "png"), "image/png"
    fitted = processing.fit_long_side(processing.decode(data), max_side)
    return processing.encode(fitted, "jpeg", quality), "image/jpeg"


def _upload_filename(url: str, mime_type: str) -> str:
    stem = PurePosixPath(urlparse(url).path).stem or "image"
    return f"{stem}.{'png' if mime_type == 'image/png' else 'jpg'}"


def fit_product_images(
    shopify: Shopify,
    search: str | None,
    max_side: int,
    quality: int,
    apply: bool,
    log: Callable[[str], None] = lambda _: None,
) -> FitResult:
    """長辺が max_side を超える商品画像だけを縮めて差し替える（apply が False なら数えるだけ）。"""
    result = FitResult()
    for product in shopify.products(search):
        for media in product["media"]["nodes"]:
            image = media.get("image")
            if not image or not image.get("width"):
                continue
            result.checked += 1
            size = (image["width"], image["height"])
            if max(size) <= max_side:
                continue
            item = Resized(product=product["handle"], media_id=media["id"], before=size)
            result.resized.append(item)
            label = f"{product['handle']} {media['id']} {size[0]}x{size[1]}"
            if not apply:
                log(f"縮める: {label}")
                continue
            try:
                original = shopify.download(image["url"])
                mime_type = media.get("mimeType") or ""
                data, mime_type = fit_image(original, mime_type, max_side, quality)
                resource_url = shopify.stage_upload(
                    data, _upload_filename(image["url"], mime_type), mime_type
                )
                new_image = shopify.replace_image(media["id"], resource_url)
            except (ShopifyError, processing.ImageError, httpx.HTTPError) as e:
                result.resized.remove(item)
                result.errors.append(f"{label}: {e}")
                log(f"失敗: {label}: {e}")
                continue
            item.after = (new_image["width"], new_image["height"])
            item.url_after = new_image["url"]
            item.bytes_before, item.bytes_after = len(original), len(data)
            log(
                f"縮めた: {label} → {item.after[0]}x{item.after[1]}"
                f"（{len(original) // 1024}KB → {len(data) // 1024}KB）"
            )
    return result
