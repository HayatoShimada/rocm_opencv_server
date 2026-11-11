"""着用イメージ（AI の合成）のモデルの部分。尺度の計算は app/garment_scale.py。

- 骨格の点: ViTPose（usyd-community/vitpose-base-simple。Apache 2.0）。COCO の 17 点
- 人・服の形: SAM2（~/models/sam2.1-hiera-large）。点と箱で指して、形（マスク）を取る
- 生成: Qwen-Image-Edit-2509（Apache 2.0）。本体は GGUF の量子化（24GB に入れるため）、
  Lightning の LoRA で 8 ステップ。文章の部分は CPU に置き、使うときだけ GPU へ
  （enable_model_cpu_offload）

GPU は 24GB なので、段階ごとに読み込んで、使い終わったら外す（unload）。
Clef（約 20GB）や snoo と同時には動かさない（scripts/snoo_phase.sh offline で空けてから）。
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings

POSE_MODEL = "usyd-community/vitpose-base-simple"

PROMPT = " ".join(
    [
        "Image 1 shows a person with a garment photo pasted over the body at its real size.",
        "Make the person naturally wear this garment: drape it on the body with natural folds,",
        "shadows and lighting, arms in the sleeves. Keep the garment's size, length, hem height,",
        "sleeve length and outline exactly as pasted - do not shorten, lengthen, widen or",
        "narrow it.",
        "Keep its color, fabric, pattern, print, buttons and collar exactly as in image 2.",
        "Remove the hanger. Keep the face, hair, hands, legs, shoes, pose and background",
        "unchanged.",
    ]
)


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def free_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_gb() -> float:
    import torch

    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0


class Pose:
    """ViTPose。人が1人だけ写っている写真の、COCO の 17 点（画像の座標）と確からしさを返す。"""

    def __init__(self) -> None:
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(POSE_MODEL)
        self.model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL).to(_device()).eval()

    def __call__(self, image: Image.Image, box: tuple[float, float, float, float] | None = None):
        w, h = image.size
        x0, y0, x1, y1 = box or (0, 0, w, h)
        boxes = [[[x0, y0, x1 - x0, y1 - y0]]]  # COCO の形（x, y, 幅, 高さ）
        inputs = self.processor(image, boxes=boxes, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.processor.post_process_pose_estimation(out, boxes=boxes)[0][0]
        return res["keypoints"].cpu().numpy(), res["scores"].cpu().numpy()


class Segmenter:
    """SAM2。点（正・負）と箱で指した物の形を返す。"""

    def __init__(self, path: str | None = None) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        p = str(Path(path or settings.vton_sam_path).expanduser())
        self.processor = Sam2Processor.from_pretrained(p)
        self.model = Sam2Model.from_pretrained(p).to(_device()).eval()

    def __call__(
        self,
        image: Image.Image,
        points: list[tuple[float, float]] | None = None,
        labels: list[int] | None = None,
        box: tuple[float, float, float, float] | None = None,
    ) -> np.ndarray:
        kwargs = {}
        if points:
            kwargs["input_points"] = [[[[float(v) for v in p] for p in points]]]
            kwargs["input_labels"] = [[[int(v) for v in (labels or [1] * len(points))]]]
        if box:
            kwargs["input_boxes"] = [[[float(v) for v in box]]]
        inputs = self.processor(images=image, return_tensors="pt", **kwargs).to(self.model.device)
        with self.torch.no_grad():
            out = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])
        return masks[0][0, 0].numpy().astype(bool)


@dataclass
class TryOnModel:
    pipe: object

    @classmethod
    def load(cls) -> TryOnModel:
        import torch
        from diffusers import (
            GGUFQuantizationConfig,
            QwenImageEditPlusPipeline,
            QwenImageTransformer2DModel,
        )

        base = Path(settings.vton_model_path).expanduser()
        transformer = QwenImageTransformer2DModel.from_single_file(
            str(Path(settings.vton_gguf).expanduser()),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=str(base / "transformer"),
            torch_dtype=torch.bfloat16,
        )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            str(base), transformer=transformer, torch_dtype=torch.bfloat16
        )
        if settings.vton_lora:
            pipe.load_lora_weights(str(Path(settings.vton_lora).expanduser()))
        pipe.enable_model_cpu_offload()
        return cls(pipe=pipe)

    def refine(self, guide: Image.Image, garment: Image.Image, seed: int = 0) -> Image.Image:
        import torch

        out = self.pipe(
            image=[guide, garment],
            prompt=PROMPT,
            negative_prompt=" ",
            true_cfg_scale=1.0,
            num_inference_steps=settings.vton_steps,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        return out.resize(guide.size, Image.LANCZOS)


def composite(
    original: Image.Image, generated: Image.Image, mask: Image.Image, grow_px: int
) -> Image.Image:
    """下敷きの範囲（少し広げる）の外は、元の写真に戻す（顔・手・背景を AI に変えさせない）。"""
    from PIL import ImageFilter

    grown = mask.filter(ImageFilter.MaxFilter(grow_px * 2 + 1)).filter(
        ImageFilter.GaussianBlur(grow_px / 3)
    )
    return Image.composite(generated.convert("RGB"), original.convert("RGB"), grown)


def collage(
    person: Image.Image,
    garment: Image.Image,
    garment_mask: np.ndarray,
    k: float,
    at: tuple[int, int],
) -> tuple[Image.Image, Image.Image]:
    """服の写真を倍率 k で縮め、形（マスク）で切り抜いて、人物の写真の at に貼る。

    貼った範囲のマスクも返す。
    """
    w, h = garment.size
    size = (max(1, round(w * k)), max(1, round(h * k)))
    g = garment.convert("RGB").resize(size, Image.LANCZOS)
    alpha = Image.fromarray((garment_mask * 255).astype(np.uint8)).resize(size, Image.LANCZOS)
    out = person.convert("RGB").copy()
    out.paste(g, at, alpha)
    mask = Image.new("L", person.size, 0)
    mask.paste(alpha, at)
    return out, mask
