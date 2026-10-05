"""Shopify の商品画像のうち、長辺が上限を超えるものを縮めて差し替える。

.env の SHOPIFY_* を読み込んで（docker では docker compose run --rm server python -m ...）、
  uv run --env-file .env python -m scripts.shopify_fit_images [--product <ID|ハンドル>] [--apply]
--apply を付けないときは、縮める画像を表示するだけ。--product を省くと全商品。
"""

import argparse

from app import shopify
from app.config import settings
from app.gpu import init_opencl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", action="append", help="商品の ID（gid・数字）またはハンドル")
    parser.add_argument("--max-side", type=int, default=settings.fit_max_side)
    parser.add_argument("--quality", type=int, default=settings.fit_quality)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    init_opencl(settings.use_opencl)
    client = shopify.Shopify(
        settings.shopify_store,
        settings.shopify_client_id,
        settings.shopify_client_secret,
        settings.shopify_admin_token,
    )
    mode = "実行します" if args.apply else "確認だけ（--apply で実行）"
    print(f"{mode}: 長辺 {args.max_side}px まで")
    searches = [shopify.product_search(p) for p in args.product] if args.product else [None]
    checked = resized = 0
    errors: list[str] = []
    for search in searches:
        result = shopify.fit_product_images(
            client, search, args.max_side, args.quality, args.apply, log=print
        )
        checked += result.checked
        resized += len(result.resized)
        errors += result.errors
    verb = "縮めた" if args.apply else "縮める"
    print(f"画像 {checked} 枚のうち {verb}: {resized} 枚、失敗: {len(errors)} 枚")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
