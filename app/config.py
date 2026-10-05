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


settings = Settings()
