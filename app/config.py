import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # OpenCL (ROCm) を使うか。false にすると常に CPU で処理する
    use_opencl: bool = _env_bool("USE_OPENCL", True)
    # アップロード画像の上限サイズ（バイト）
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
    # 内部 API（/v1/shopify/...）の認証。X-Internal-Token と比べる。空なら内部 API は使えない
    api_token: str = os.getenv("API_TOKEN", "")
    # Shopify Admin API（Client Credentials。読み取りの確認だけなら SHOPIFY_ADMIN_TOKEN も使える）
    shopify_store: str = os.getenv("SHOPIFY_STORE", "")
    shopify_client_id: str = os.getenv("SHOPIFY_CLIENT_ID", "")
    shopify_client_secret: str = os.getenv("SHOPIFY_CLIENT_SECRET", "")
    shopify_admin_token: str = os.getenv("SHOPIFY_ADMIN_TOKEN", "")
    # 商品画像の長辺の上限（CMS の productPhotos と同じ 2048）と JPEG の品質
    fit_max_side: int = int(os.getenv("FIT_MAX_SIDE", "2048"))
    fit_quality: int = int(os.getenv("FIT_QUALITY", "90"))


settings = Settings()
