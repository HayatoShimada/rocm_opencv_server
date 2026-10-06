"""Clef の分類（scripts/shopify_classify_photos.py）が正しいかを、全画像について確かめる画面。

  uv run python -m scripts.photo_review [--port 8010]
http://127.0.0.1:8010 を開く。
確かめた結果（正しい区分）は data/photo-labels/labels.json に保存する。
手元だけで開く（127.0.0.1 で待ち受ける）。
"""

import argparse
import datetime
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import clef

DATA = Path(os.getenv("PHOTO_LABELS_DIR", "data/photo-labels"))
PAGE = Path(__file__).with_name("photo_review.html")


class Label(BaseModel):
    person: bool
    whole: bool


class LabelsIn(BaseModel):
    # null は「未確認に戻す」
    labels: dict[str, Label | None]


def _read(name: str) -> dict:
    path = DATA / name
    return json.loads(path.read_text()) if path.exists() else {}


def _write(name: str, data: dict) -> None:
    # 書きかけで壊れないよう、別のファイルに書いてから置き換える
    path = DATA / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(path)


app = FastAPI(title="商品写真の分類の確認")


@app.get("/")
def page() -> FileResponse:
    return FileResponse(PAGE)


@app.get("/api/items")
def items() -> dict:
    predictions = _read("predictions.json")
    return {
        "threshold": clef.THRESHOLD,
        "items": [
            {
                "id": media_id,
                "url": p["url"],
                "products": p["products"],
                "person": p["person"],
                "whole": p["whole"],
            }
            for media_id, p in predictions.items()
        ],
        "labels": _read("labels.json"),
    }


@app.post("/api/labels")
def save_labels(body: LabelsIn) -> dict:
    labels = _read("labels.json")
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    for media_id, label in body.labels.items():
        if label is None:
            labels.pop(media_id, None)
        else:
            labels[media_id] = {**label.model_dump(), "reviewed_at": now}
    DATA.mkdir(parents=True, exist_ok=True)
    _write("labels.json", labels)
    return {"saved": len(body.labels), "reviewed": len(labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    print(f"http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
