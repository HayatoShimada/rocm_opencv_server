"""Shopify の商品画像を、長辺が上限に収まるように縮めて差し替える。

差し替えは fileUpdate の originalSource で行う（MediaImage の ID・並び順・alt はそのまま）。
85store-cms の src/shopify/client.ts・upload.ts と同じ作法で Admin GraphQL を呼ぶ。
"""

import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any
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
        wait = 0.5
        while True:
            node = self.graphql(MEDIA_STATUS, {"id": media_id})["node"]
            if node["fileStatus"] == "READY" and node["image"]:
                return node["image"]
            if node["fileStatus"] == "FAILED":
                raise ShopifyError(f"Shopify で画像の処理に失敗しました: {media_id}")
            if time.time() > deadline:
                raise ShopifyError(f"Shopify の画像の処理が終わりません: {media_id}")
            time.sleep(wait)
            wait = min(wait * 2, 2)

    def update_alts(self, files: list[dict]) -> None:
        res = self.graphql(FILE_UPDATE, {"files": files})
        _assert_no_user_errors("fileUpdate", res["fileUpdate"]["userErrors"])


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


def auto_alt(title: str, index: int, label: str | None = None) -> str:
    """alt が空の画像に入れる文言（85store-cms の src/shopify/mapping.ts の autoAlt と同じ形）。

    写真の区分（photo_label）が分かれば「商品名（正面）」、分からなければ「商品名（n枚目）」。
    """
    return f"{title}（{label or f'{index + 1}枚目'}）"


# 写真の区分の語（85store-cms の AUTO_ALT_LABELS と同じ。変えるときは両方直す）
VIEW_LABELS = {
    "whole": {"front": "正面", "back": "背面", "side": "横"},
    "worn": {"front": "着用", "back": "着用・背面", "side": "着用・横"},
}
PART_LABELS = {
    "tag": "タグのアップ",
    "print": "ロゴ・プリントのアップ",
    "fabric": "生地のアップ",
    "fastener": "ボタン・ジッパーのアップ",
    "collar": "襟元のアップ",
    "hem": "袖口・裾のアップ",
    "pocket": "ポケットのアップ",
    "damage": "傷・汚れのアップ",
}
ALT_LABELS = [
    "正面", "背面", "横", "全体", "着用", "着用・背面", "着用・横",
    *PART_LABELS.values(),
    "ディテール",
]  # fmt: skip
# 確からしさがこれ未満の向き・部分は使わず、「全体」「着用」「ディテール」とだけ書く
MIN_VIEW_CONFIDENCE = 0.6
MIN_PART_CONFIDENCE = 0.3


def photo_label(
    kind: str, view: str, view_confidence: float, part: str, part_confidence: float
) -> str:
    """Clef の判定（app/clef.py）から、alt に付ける区分の語を決める。"""
    if kind == "closeup":
        if part_confidence >= MIN_PART_CONFIDENCE and part in PART_LABELS:
            return PART_LABELS[part]
        return "ディテール"
    fallback = "着用" if kind == "worn" else "全体"
    if view_confidence >= MIN_VIEW_CONFIDENCE:
        return VIEW_LABELS[kind].get(view, fallback)
    return fallback


def auto_alt_label(alt: str, title: str, index: int) -> str | None:
    """自動で付けた alt なら区分の語（「n枚目」なら ""）を、人が入れた alt なら None を返す。"""
    if alt == auto_alt(title, index):
        return ""
    if alt.startswith(f"{title}（") and alt.endswith("）"):
        label = alt[len(title) + 1 : -1]
        if label in ALT_LABELS:
            return label
    return None


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
    alt_filled: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def fit_image(data: bytes, mime_type: str, max_side: int, quality: int) -> tuple[bytes, str]:
    """画像を縮め、色を sRGB にしてエンコードし直す。透過のある PNG は PNG のまま、ほかは JPEG。

    縮小は CPU で行う（1枚ずつなら GPU との転送のほうが重い。4284x5712 で CPU 5ms・GPU 15ms）。
    """
    icc = processing.icc_profile(data)
    if mime_type == "image/png":
        image = processing.decode(data, keep_alpha=True)
        if processing.has_alpha(image):
            fitted = processing.fit_long_side(image, max_side, device=False)
            return processing.encode(processing.to_srgb(fitted, icc), "png"), "image/png"
    fitted = processing.fit_long_side(processing.decode(data), max_side, device=False)
    return processing.encode(processing.to_srgb(fitted, icc), "jpeg", quality), "image/jpeg"


def _upload_filename(url: str, mime_type: str) -> str:
    stem = PurePosixPath(urlparse(url).path).stem or "image"
    return f"{stem}.{'png' if mime_type == 'image/png' else 'jpg'}"


def _in_parallel(
    workers: int, fn: Callable, items: list
) -> Iterator[tuple[Any, Any, BaseException | None]]:
    """items を並列に処理し、終わった順に (item, 戻り値, 例外) を返す。"""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            error = future.exception()
            yield futures[future], None if error else future.result(), error


def fit_product_images(
    shopify: Shopify,
    search: str | None,
    max_side: int,
    quality: int,
    apply: bool,
    fill_alt: bool = False,
    workers: int = 6,
    log: Callable[[str], None] = lambda _: None,
) -> FitResult:
    """長辺が max_side を超える商品画像を縮めて差し替え、fill_alt なら空の alt を商品名で埋める。

    apply が False なら数えるだけ。画像は workers 枚ずつ並列に処理する
    （時間のほとんどは Shopify とのやりとりと、差し替えた画像の処理待ち）。
    """
    result = FitResult()
    alts: list[tuple[str, list[dict]]] = []
    targets: list[tuple[Resized, dict, str]] = []
    for product in shopify.products(search):
        missing = []
        for index, media in enumerate(product["media"]["nodes"]):
            image = media.get("image")
            if not image or not image.get("width"):
                continue
            result.checked += 1
            if fill_alt and not media.get("alt"):
                missing.append({"id": media["id"], "alt": auto_alt(product["title"], index)})
            size = (image["width"], image["height"])
            if max(size) > max_side:
                item = Resized(product=product["handle"], media_id=media["id"], before=size)
                targets.append((item, image, media.get("mimeType") or ""))
        if missing:
            alts.append((product["handle"], missing))

    if not apply:
        for handle, files in alts:
            log(f"alt を入れる: {handle} {len(files)} 枚（{files[0]['alt']} など）")
        for item, _, _ in targets:
            log(f"縮める: {item.product} {item.media_id} {item.before[0]}x{item.before[1]}")
        result.alt_filled = sum(len(files) for _, files in alts)
        result.resized = [item for item, _, _ in targets]
        return result

    # alt を先に入れる（処理中の画像は更新できないので、差し替えより前に）
    for (handle, files), _, error in _in_parallel(
        workers, lambda a: shopify.update_alts(a[1]), alts
    ):
        if error:
            result.errors.append(f"{handle} の alt: {error}")
            log(f"失敗: {handle} の alt: {error}")
        else:
            result.alt_filled += len(files)

    def fit_one(target: tuple[Resized, dict, str]) -> tuple[dict, int, int]:
        item, image, mime_type = target
        original = shopify.download(image["url"])
        data, mime_type = fit_image(original, mime_type, max_side, quality)
        resource_url = shopify.stage_upload(
            data, _upload_filename(image["url"], mime_type), mime_type
        )
        return shopify.replace_image(item.media_id, resource_url), len(original), len(data)

    for (item, _, _), done, error in _in_parallel(workers, fit_one, targets):
        label = f"{item.product} {item.media_id} {item.before[0]}x{item.before[1]}"
        if error:
            result.errors.append(f"{label}: {error}")
            log(f"失敗: {label}: {error}")
            continue
        new_image, item.bytes_before, item.bytes_after = done
        item.after = (new_image["width"], new_image["height"])
        item.url_after = new_image["url"]
        result.resized.append(item)
        log(
            f"縮めた: {label} → {item.after[0]}x{item.after[1]}"
            f"（{item.bytes_before // 1024}KB → {item.bytes_after // 1024}KB）"
        )
    return result
