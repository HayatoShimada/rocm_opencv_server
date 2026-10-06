import json

import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient

from app import clef


def _client(handler) -> clef.Clef:
    return clef.Clef("acc", "token", http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_classify_sends_image_and_reads_answers():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        answers = {
            "person": {"type": "noul", "noul": 0.9},
            "whole": {"type": "noul", "noul": 0.2},
            "view": {"type": "choice", "choice": "back", "confidence": 0.8, "probabilities": {}},
            "part": {"type": "choice", "choice": "tag", "confidence": 0.4, "probabilities": {}},
        }
        return httpx.Response(200, json={"success": True, "result": {"answers": answers}})

    labels = _client(handler).classify(b"\xff\xd8\xff", "image/jpeg")
    assert seen["url"].endswith("/accounts/acc/ai/run/@cf/cloudflare/clef-flash")
    assert seen["body"]["model"] == "clef-flash"
    assert set(seen["body"]["questions"]) == {"person", "whole", "view", "part"}
    assert seen["body"]["images"][0] == {"content_type": "image/jpeg", "base64": "/9j/"}
    assert labels == clef.PhotoLabels(0.9, 0.2, "back", 0.8, "tag", 0.4)
    assert labels.kind == "worn"


def test_classify_retries_when_busy(monkeypatch):
    monkeypatch.setattr(clef.time, "sleep", lambda _: None)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, json={"success": False, "errors": []})
        answers = {"person": {"noul": 0.0}, "whole": {"noul": 0.1}}
        return httpx.Response(200, json={"success": True, "result": {"answers": answers}})

    assert _client(handler).classify(b"x", "image/png").kind == "closeup"
    assert len(calls) == 3


def test_classify_raises_on_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"success": False, "errors": [{"message": "bad"}]})

    with pytest.raises(clef.ClefError, match="bad"):
        _client(handler).classify(b"x", "image/png")


def test_photo_kind():
    assert clef.photo_kind(True, False) == "worn"
    assert clef.photo_kind(False, True) == "whole"
    assert clef.photo_kind(False, False) == "closeup"


def test_review_saves_and_clears_labels(tmp_path, monkeypatch):
    from scripts import photo_review

    monkeypatch.setattr(photo_review, "DATA", tmp_path)
    (tmp_path / "predictions.json").write_text(
        json.dumps(
            {
                "gid://shopify/MediaImage/1": {
                    "url": "https://cdn.shopify.com/a.jpg?v=1",
                    "products": [{"handle": "a", "title": "A", "position": 1}],
                    "person": 0.1,
                    "whole": 0.9,
                }
            }
        )
    )
    client = TestClient(photo_review.app)
    assert "写真の分類の確認" in client.get("/").text
    data = client.get("/api/items").json()
    assert data["items"][0]["whole"] == 0.9 and data["labels"] == {}

    media_id = "gid://shopify/MediaImage/1"
    res = client.post("/api/labels", json={"labels": {media_id: {"person": False, "whole": False}}})
    assert res.json() == {"saved": 1, "reviewed": 1}
    label = client.get("/api/items").json()["labels"][media_id]
    assert label["whole"] is False and label["reviewed_at"]

    client.post("/api/labels", json={"labels": {media_id: None}})
    assert client.get("/api/items").json()["labels"] == {}


def test_from_settings_picks_backend(tmp_path):
    from dataclasses import replace

    from app.config import settings

    workers = replace(settings, clef_backend="workers-ai", cloudflare_account_id="a")
    workers = replace(workers, cloudflare_api_token="t")
    assert isinstance(clef.from_settings(workers), clef.Clef)
    # 手元: 重みがなければ、どこを見たかを伝える
    with pytest.raises(clef.ClefError, match="重みがありません"):
        clef.from_settings(replace(settings, clef_backend="local", clef_model_path=str(tmp_path)))
    with pytest.raises(clef.ClefError, match="local か workers-ai"):
        clef.from_settings(replace(settings, clef_backend="gpu"))


def test_labels_from_answers_defaults_missing_questions():
    labels = clef.labels_from_answers({"person": {"noul": 0.1}, "whole": {"noul": 0.9}})
    assert labels == clef.PhotoLabels(0.1, 0.9)
    assert labels.kind == "whole"
