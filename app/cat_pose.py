"""猫の関節（39 点）を DeepLabCut の SuperAnimal-Quadruped（HRNet-w32）で推定する。

DeepLabCut の本体は入れず、Hugging Face の重み（REPO。研究・非商用に限る）を
timm の hrnet_w32 と 1x1 の転置畳み込みに読み込んで動かす。
猫の箱（app/snoo.py の検出）を正方形に切り出して 256x256 で推定する（top-down）。
長毛でも鼻・目・耳・足先・しっぽは取れるが、背中・脇腹の点はぶれやすい。1 匹 12 ms ほど。
"""

import cv2
import numpy as np

REPO = "mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped"
WEIGHTS = "superanimal_quadruped_hrnet_w32.pt"
# DeepLabCut の modelzoo/project_configs/superanimal_quadruped.yaml の順（thai は原文のまま）
NAMES = [
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "right_earbase", "right_earend", "right_antler_base", "right_antler_end",
    "left_eye", "left_earbase", "left_earend", "left_antler_base", "left_antler_end",
    "neck_base", "neck_end", "throat_base", "throat_end",
    "back_base", "back_end", "back_middle", "tail_base", "tail_end",
    "front_left_thai", "front_left_knee", "front_left_paw",
    "front_right_thai", "front_right_knee", "front_right_paw",
    "back_left_paw", "back_left_thai", "back_right_thai",
    "back_left_knee", "back_right_knee", "back_right_paw",
    "belly_bottom", "body_middle_right", "body_middle_left",
]  # fmt: skip
SIZE = 256
# 箱の縁で足先やしっぽが切れないよう、箱の長いほうの辺を少し広げて切り出す
MARGIN = 1.15
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class CatPose:
    def __init__(self, device: str = "cuda"):
        import timm
        import torch
        from huggingface_hub import hf_hub_download

        self.torch = torch
        self.device = device
        state = torch.load(hf_hub_download(REPO, WEIGHTS), map_location="cpu", weights_only=True)
        state = state["model"]
        prefix = "backbone.model."
        self.backbone = timm.create_model(
            "hrnet_w32", pretrained=False, features_only=True, feature_location=""
        )
        self.backbone.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        )
        # HRNet のいちばん細かい枝（32 チャンネル・1/4）だけを使う（interpolate_branches: false）
        self.branch = self.backbone.feature_info.channels().index(32)
        self.head = torch.nn.ConvTranspose2d(32, len(NAMES), 1)
        self.head.weight.data = state["heads.bodypart.heatmap_head.deconv_layers.0.weight"]
        self.head.bias.data = state["heads.bodypart.heatmap_head.deconv_layers.0.bias"]
        self.backbone = self.backbone.to(device).eval()
        self.head = self.head.to(device).eval()
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def estimate(self, image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
        """BGR の画像と猫の箱（幅・高さに対する比）から、関節ごとの [x, y, 確からしさ] を返す。

        形は (39, 3)。x・y は画素。
        """
        torch = self.torch
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * MARGIN
        scale = SIZE / side
        warp = np.array([[scale, 0, SIZE / 2 - cx * scale], [0, scale, SIZE / 2 - cy * scale]])
        crop = cv2.warpAffine(image, warp.astype(np.float32), (SIZE, SIZE))
        x = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).float().div_(255)
        with torch.inference_mode():
            heatmaps = self.head(self.backbone((x - self.mean) / self.std)[self.branch])[0]
        n, hh, ww = heatmaps.shape
        scores, at = heatmaps.reshape(n, -1).max(dim=1)
        stride = SIZE / ww
        u = ((at % ww).float() + 0.5) * stride
        v = ((at // ww).float() + 0.5) * stride
        px = (u - SIZE / 2) / scale + cx
        py = (v - SIZE / 2) / scale + cy
        return torch.stack([px, py, scores.clamp(0, 1)], dim=1).cpu().numpy()
