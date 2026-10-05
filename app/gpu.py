"""OpenCL (ROCm) デバイスの初期化と情報取得。

OpenCV の CUDA モジュールは AMD GPU では使えないため、T-API (cv2.UMat) 経由で
ROCm の OpenCL ランタイムに処理を流す。
"""

import cv2


def init_opencl(enabled: bool) -> bool:
    """OpenCL を有効化し、実際に使える状態になったかを返す。"""
    cv2.ocl.setUseOpenCL(enabled and cv2.ocl.haveOpenCL())
    return cv2.ocl.useOpenCL()


def device_info() -> dict:
    info: dict = {
        "opencv_version": cv2.__version__,
        "opencl_available": cv2.ocl.haveOpenCL(),
        "opencl_enabled": cv2.ocl.useOpenCL(),
        "device": None,
    }
    if not info["opencl_available"]:
        return info

    device = cv2.ocl.Device.getDefault()
    info["device"] = {
        "name": device.name(),
        "vendor": device.vendorName(),
        "version": device.version(),
        "driver_version": device.driverVersion(),
        "compute_units": device.maxComputeUnits(),
        "global_memory_mb": device.globalMemSize() // (1024 * 1024),
    }
    return info
