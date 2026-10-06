"""商品写真に人が写っているか・商品の全体が写っているかを、
Cloudflare Workers AI の Clef で判定する。

Clef は、質問（はい・いいえ など）ごとに選択肢の確率を返す判定用のモデル（文章は生成しない）。
画像は base64 で渡す（URL は受け付けない。1枚 4MiB・最大4枚）。
"""

import base64
import time
from dataclasses import dataclass

import httpx2 as httpx

DEFAULT_MODEL = "@cf/cloudflare/clef-flash"
# 質問や渡し方を変えたら上げる（保存してある判定を取り直す）
VERSION = 2
STATE = "オンラインストアの商品写真を分類する"
QUESTIONS = {
    "person": {
        "type": "noul",
        "instructions": (
            "人が服を着ている写真（着用・モデル）か。"
            "手や指だけが写っている（商品を持つ・広げる・タグを指す）写真は no"
        ),
    },
    "whole": {
        "type": "noul",
        "instructions": (
            "商品（服・小物）の全体が1枚に収まっているか。"
            "一部だけのアップ（タグ・生地・ボタン・プリントの寄り）なら no"
        ),
    },
}
# 確率がこれ以上なら「はい」とみなす
THRESHOLD = 0.5


class ClefError(Exception):
    pass


@dataclass(frozen=True)
class PhotoLabels:
    """はいの確率（0〜1）。person は人が着ているか（手・指だけなら低い）。"""

    person: float
    whole: float

    @property
    def kind(self) -> str:
        return photo_kind(self.person >= THRESHOLD, self.whole >= THRESHOLD)


def photo_kind(person: bool, whole: bool) -> str:
    """写真の区分。人が写っていれば着用、いなければ全体かアップ。"""
    if person:
        return "worn"
    return "whole" if whole else "closeup"


KIND_LABELS = {"worn": "着用", "whole": "全体", "closeup": "アップ"}


class Clef:
    def __init__(self, account_id: str, api_token: str, model: str = DEFAULT_MODEL, http=None):
        if not account_id or not api_token:
            raise ClefError("CLOUDFLARE_ACCOUNT_ID と CLOUDFLARE_API_TOKEN を設定してください")
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        self.model = model.rsplit("/", 1)[-1]
        self._http = http or httpx.Client(timeout=60)
        self._headers = {"Authorization": f"Bearer {api_token}"}

    def classify(self, image: bytes, content_type: str) -> PhotoLabels:
        payload = {
            "model": self.model,
            "state": STATE,
            "questions": QUESTIONS,
            "images": [{"content_type": content_type, "base64": base64.b64encode(image).decode()}],
        }
        wait = 1.0
        for _ in range(6):
            res = self._http.post(self.url, json=payload, headers=self._headers)
            # 混んでいるとき・一時的な失敗は、少し待ってやり直す
            if res.status_code == 429 or res.status_code >= 500:
                time.sleep(wait)
                wait = min(wait * 2, 16)
                continue
            body = res.json()
            if not body.get("success"):
                errors = " / ".join(e.get("message", "") for e in body.get("errors", []))
                raise ClefError(f"Workers AI: {res.status_code} {errors}")
            answers = body["result"]["answers"]
            return PhotoLabels(person=answers["person"]["noul"], whole=answers["whole"]["noul"])
        raise ClefError("Workers AI: やり直しても応答がありません")
