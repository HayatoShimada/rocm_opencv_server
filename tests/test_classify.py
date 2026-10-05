import numpy as np

from app.classify import PROMPTS, VERSION, _result


def test_result_picks_highest_score():
    kinds = list(PROMPTS)
    probs = np.array([0.1, 0.7, 0.15, 0.05])
    result = _result(kinds, probs)
    assert result.kind == kinds[1]
    assert result.confidence == 0.7
    assert set(result.scores) == set(kinds)


def test_version_changes_with_prompts(monkeypatch):
    import importlib

    from app import classify

    monkeypatch.setenv("CLIP_MODEL", "ViT-B-32")
    assert importlib.reload(classify).VERSION != VERSION
    monkeypatch.delenv("CLIP_MODEL")
    importlib.reload(classify)
