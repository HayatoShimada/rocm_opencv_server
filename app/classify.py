"""商品写真の種類（単品・着用・ディテール・その他）を CLIP で判定する（ゼロショット）。

画像と、種類ごとの説明文（英語。CLIP は英語で学習している）の類似度を比べ、いちばん近い種類にする。
torch・open_clip は重いので、使うときだけ読み込む（uv sync --group clip）。
GPU がなければ CPU で動く。
"""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

Kind = Literal["single", "worn", "detail", "other"]
KIND_LABELS: dict[Kind, str] = {
    "single": "単品",
    "worn": "着用",
    "detail": "ディテール",
    "other": "その他",
}
# 種類ごとの説明文。埋め込みを平均して種類の代表にする
PROMPTS: dict[Kind, list[str]] = {
    "single": [
        "a product photo of an entire garment hanging on a hanger in front of a plain wall",
        "a full-length photo of clothing on a hanger against a white wall, empty wall around it",
        "a photo of a whole pair of pants hanging on a hanger on a wall, no person",
        "a photo of a whole garment laid flat",
    ],
    "worn": [
        "a photo of a person wearing clothes",
        "a fashion photo of a model wearing an outfit",
        "a photo of a man wearing a shirt and pants",
        "a photo of a woman wearing clothes",
    ],
    "detail": [
        "a close-up photo of fabric texture",
        "a close-up photo of a clothing tag or label",
        "a close-up photo of a button, zipper or stitching",
        "a close-up photo of a print or embroidery on clothing",
        "a close-up photo of the collar and neck tag of a garment",
        "a cropped close-up photo showing only part of a garment",
    ],
    "other": [
        "a size chart with text",
        "a logo graphic on a plain background",
    ],
}
MODEL = os.getenv("CLIP_MODEL", "ViT-L-14")
PRETRAINED = os.getenv("CLIP_PRETRAINED", "laion2b_s32b_b82k")
# モデルと説明文の版。変えたら判定し直す（判定のキャッシュのキーに入れる）
VERSION = hashlib.sha1(json.dumps([MODEL, PRETRAINED, PROMPTS]).encode()).hexdigest()[:8]


@dataclass
class Classification:
    kind: Kind
    confidence: float
    scores: dict[str, float]


class Classifier:
    def __init__(self, model: str = MODEL, pretrained: str = PRETRAINED):
        import open_clip
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(model)
        self.kinds: list[Kind] = list(PROMPTS)
        with torch.no_grad():
            classes = []
            for kind in self.kinds:
                text = self.model.encode_text(tokenizer(PROMPTS[kind]).to(self.device))
                text = text / text.norm(dim=-1, keepdim=True)
                mean = text.mean(dim=0)
                classes.append(mean / mean.norm())
            self.text = torch.stack(classes)

    def classify(self, images: list[np.ndarray]) -> list[Classification]:
        """BGR の画像をまとめて判定する。"""
        from PIL import Image

        torch = self.torch
        batch = torch.stack(
            [self.preprocess(Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))) for i in images]
        ).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            probs = (100.0 * features @ self.text.T).softmax(dim=-1).cpu().numpy()
        return [_result(self.kinds, p) for p in probs]


def _result(kinds: list[Kind], probs: np.ndarray) -> Classification:
    best = int(np.argmax(probs))
    scores = {k: round(float(p), 4) for k, p in zip(kinds, probs, strict=True)}
    return Classification(kinds[best], float(probs[best]), scores)
