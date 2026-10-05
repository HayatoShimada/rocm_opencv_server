"""CPU と OpenCL (ROCm) の処理時間を比較する簡易ベンチマーク。

uv run python scripts/bench.py [--size 3840x2160] [--iterations 50]
"""

import argparse
import time

import cv2
import numpy as np


def run(label: str, use_opencl: bool, image: np.ndarray, iterations: int) -> None:
    cv2.ocl.setUseOpenCL(use_opencl)
    src = cv2.UMat(image) if use_opencl else image

    def pipeline():
        blurred = cv2.GaussianBlur(src, (15, 15), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges.get() if use_opencl else edges

    pipeline()  # ウォームアップ（OpenCL カーネルのコンパイルを除外する）
    start = time.perf_counter()
    for _ in range(iterations):
        pipeline()
    elapsed = (time.perf_counter() - start) / iterations * 1000
    print(f"{label:>8}: {elapsed:8.2f} ms/iter")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="3840x2160")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    width, height = map(int, args.size.split("x"))
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    print(f"OpenCV {cv2.__version__}, image {width}x{height}")
    run("CPU", False, image, args.iterations)
    if cv2.ocl.haveOpenCL():
        print(f"device: {cv2.ocl.Device.getDefault().name()}")
        run("OpenCL", True, image, args.iterations)
    else:
        print("OpenCL が利用できません（ROCm OpenCL ランタイムを確認してください）")


if __name__ == "__main__":
    main()
