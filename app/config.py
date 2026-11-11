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
    # 写真の分類（Clef）を動かす場所。local（手元の GPU。既定）か workers-ai
    clef_backend: str = os.getenv("CLEF_BACKEND", "local")
    # 手元で動かすときの重みの場所（Hugging Face の Cloudflare/clef-flash をダウンロードしたもの）
    clef_model_path: str = os.getenv("CLEF_MODEL_PATH", "~/models/clef-flash")
    # Cloudflare Workers AI（CLEF_BACKEND=workers-ai のとき。トークンは Workers AI の権限）
    cloudflare_account_id: str = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
    cloudflare_api_token: str = os.getenv("CLOUDFLARE_API_TOKEN", "")
    clef_model: str = os.getenv("CLEF_MODEL", "@cf/cloudflare/clef-flash")
    # 着用イメージ（AI の合成。app/vton.py）。
    # Qwen-Image-Edit-2509 の本体以外・GGUF の本体・Lightning の LoRA
    vton_model_path: str = os.getenv("VTON_MODEL_PATH", "~/models/qwen-image-edit-2509")
    vton_gguf: str = os.getenv(
        "VTON_GGUF", "~/models/qwen-image-edit-2509-gguf/Qwen-Image-Edit-2509-Q4_K_M.gguf"
    )
    vton_lora: str = os.getenv(
        "VTON_LORA",
        "~/models/qwen-image-lightning/Qwen-Image-Edit-2509/"
        "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors",
    )
    vton_steps: int = int(os.getenv("VTON_STEPS", "8"))
    vton_sam_path: str = os.getenv("VTON_SAM_PATH", "~/models/sam2.1-hiera-large")


settings = Settings()
