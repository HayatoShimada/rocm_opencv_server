"""scripts/snoo_dev.py（make status・deploy-*）と scripts/snoo_replay.py（make replay）の、
外を呼ばない所。
"""

import json
from datetime import datetime

import numpy as np
import pytest

from app.snoo_mapper import VggtOut
from scripts import snoo_dev, snoo_replay

LINE = (
    "2026-10-29 23:02:27,662 INFO 60.1 fps、受けてから返すまで 中央値 5.0 ms・最大 8.2 ms｜"
    "p50/p95 ms: 受信 0.02/0.03  デコード 0.45/0.49  検出 4.45/5.16  CLIP 0.00/0.00  "
    "送信 0.03/0.05｜clear 0 回（回っていて飛ばした 0）"
)


def test_latency_line_is_parsed():
    r = snoo_dev.parse_latency(LINE)
    assert r["time"] == datetime(2026, 10, 29, 23, 2, 27)
    assert (r["fps"], r["median_ms"], r["max_ms"]) == (60.1, 5.0, 8.2)
    assert r["stages"]["検出"] == (4.45, 5.16)
    assert list(r["stages"]) == ["受信", "デコード", "検出", "CLIP", "送信"]
    assert r["rest"].startswith("clear 0 回")


def test_latency_line_without_clearance():
    r = snoo_dev.parse_latency(LINE.split("｜clear")[0])
    assert r["stages"]["送信"] == (0.03, 0.05)
    assert r["rest"] == ""


def test_last_latency_skips_other_lines():
    older = LINE.replace("23:02:27", "23:02:17").replace("60.1 fps", "9.8 fps")
    text = "\n".join([older, LINE, "2026-10-29 23:02:30,000 WARNING 何か"])
    assert snoo_dev.last_latency(text)["fps"] == 60.1
    assert snoo_dev.last_latency("INFO 待ち受け") is None
    assert snoo_dev.parse_latency("INFO 0.0.0.0:8710/udp で待ち受け") is None


def test_latency_text_shows_age():
    r = snoo_dev.parse_latency(LINE)
    text = snoo_dev.latency_text(r, now=datetime(2026, 10, 29, 23, 2, 37))
    assert text.startswith("p50/p95（10 秒前）: 60.1 fps")
    assert "検出 4.45/5.16" in text
    assert "フレームが届いていない" in snoo_dev.latency_text(None)


def test_detector_stage_follows_the_log():
    stages = snoo_dev.DETECTOR_STAGES
    assert snoo_dev.stage("", stages) == "python を起動している"
    log = "HEAD https://huggingface.co/PekingU/rtdetr_v2_r18vd/resolve/main/config.json\n"
    assert "RT-DETR" in snoo_dev.stage(log, stages)
    log += "Loaded built-in ViT-B-16 model config.\n"
    log += "W1029 .venv/lib/python3.12/site-packages/torch/_dynamo/variables/tensor.py"
    assert "torch.compile" in snoo_dev.stage(log, stages)
    log += "\nINFO clear_cm を見積もる（カメラの高さ 8.0 cm）"
    assert "深度" in snoo_dev.stage(log, stages)
    log += "\nINFO clear_cm を見積もる準備ができた\nINFO 0.0.0.0:8710/udp で待ち受け（cuda）"
    assert snoo_dev.stage(log, stages) == "待ち受け"


def test_map_stage_follows_the_log():
    stages = snoo_dev.MAP_STAGES
    log = "INFO 0.0.0.0:8720 で待ち受け（受ける送り元: ...）\n"
    assert snoo_dev.stage(log, stages) == "VGGT を読んでいる"
    log += "INFO VGGT を読んだ（6.8 秒）\nINFO:     Uvicorn running on http://0.0.0.0:8720"
    assert snoo_dev.stage(log, stages) == "待ち受け"


def test_traceback_is_found():
    log = "INFO 起動\nTraceback (most recent call last):\n  File x\nNameError: y\n"
    assert snoo_dev.find_traceback(log).endswith("NameError: y\n")
    assert snoo_dev.find_traceback("INFO 待ち受け") is None


def test_image_is_stale_when_an_input_is_newer():
    assert snoo_dev.image_stale(100.0, {"uv.lock": 50.0, "Dockerfile": 150.0}) == ["Dockerfile"]
    assert snoo_dev.image_stale(100.0, {"uv.lock": 50.0}) == []


def test_docker_time():
    t = snoo_dev._docker_time("2026-10-29T13:21:03.732464743Z")
    assert t == pytest.approx(
        datetime.fromisoformat("2026-10-29T13:21:03.732464+00:00").timestamp()
    )


def test_live_lines():
    live = {
        "enabled": True,
        "senders": [
            {
                "sender": "192.168.0.188",
                "seq": 7,
                "age_s": 0.5,
                "infer_ms": 4.9,
                "dets": [{}],
                "cmd": {"moving": True},
                "clear_cm": 42.0,
            }
        ],
    }
    (line,) = snoo_dev.live_lines(live)
    assert "192.168.0.188 0.5 秒前" in line and "猫 1" in line and "走っている" in line
    assert "clear 42.0 cm" in line
    assert "まだ届いていない" in snoo_dev.live_lines({"enabled": True, "senders": []})[0]
    assert "書いていない" in snoo_dev.live_lines({"enabled": False})[0]


def test_session_lines():
    sessions = [{"id": "explore_b", "stations": 3, "latest": 2, "mtime": 1793281614.0}]
    state = {
        "reply_station": 2,
        "reply": {"action": "move", "steer_deg": 5.0, "segments": [], "map_url": "/x"},
        "timing_s": {"vggt": 6.4, "map": 0.4},
    }
    lines = snoo_dev.session_lines(sessions, state)
    assert lines[0] == "セッション: 1 個"
    assert lines[1].startswith("いちばん新しい: explore_b（3 か所")
    assert lines[2] == '最後の返事（ステーション 2）: {"action": "move", "steer_deg": 5.0}'
    assert lines[3] == "かかった秒: vggt 6.4 map 0.4"
    assert snoo_dev.session_lines([], None) == ["セッション: 0 個"]


# --- replay ---


def _vggt(files: list[str]) -> VggtOut:
    n = len(files)
    return VggtOut(
        extrinsic=np.arange(n, dtype=float)[:, None, None] * np.ones((n, 3, 4)),
        intrinsic=np.ones((n, 3, 3)),
        depth=np.arange(n, dtype=float)[:, None, None] * np.ones((n, 4, 5)),
        depth_conf=np.ones((n, 4, 5)),
        files=files,
    )


def test_npz_station_and_pick():
    v = _vggt(["/a/s/003/x.jpg", "/a/s/005/y.jpg", "/a/s/005/z.jpg"])
    assert snoo_replay.npz_station(v) == 5
    w = snoo_replay.pick(v, [("005", "z.jpg"), ("003", "x.jpg")])
    assert w.depth[:, 0, 0].tolist() == [2.0, 0.0]
    assert w.files == ["/a/s/005/z.jpg", "/a/s/003/x.jpg"]
    with pytest.raises(snoo_replay.ReplayError, match="vggt.npz に無いコマ"):
        snoo_replay.pick(v, [("005", "q.jpg")])


def test_prepare_rolls_the_session_back_to_before_the_station(tmp_path):
    src, out = tmp_path / "s", tmp_path / "out" / "s"
    for i in range(4):
        d = src / f"{i:03d}"
        d.mkdir(parents=True)
        (d / "meta.json").write_text("{}")
    pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    (src / "poses.json").write_text(json.dumps({str(i): pose for i in range(4)}))
    (src / "reverse.json").write_text(json.dumps({"1": {}, "2": {}}))
    (src / "vggt.npz").write_bytes(b"x")
    snoo_replay.prepare(src, out, 2)
    assert sorted(p.name for p in out.iterdir()) == [
        "000",
        "001",
        "002",
        "poses.json",
        "reverse.json",
    ]
    assert list(json.loads((out / "poses.json").read_text())) == ["0", "1"]
    assert list(json.loads((out / "reverse.json").read_text())) == ["1"]
    assert (src / "003").is_dir()  # もとは変えない
    with pytest.raises(snoo_replay.ReplayError):
        snoo_replay.prepare(src, src, 2)


def test_previous_reply_only_for_the_same_station(tmp_path):
    assert snoo_replay.previous_reply(tmp_path, 3) is None
    reply = {"action": "done", "reason": "unreachable"}
    plan = {"window": {"stations": [1, 3]}, "reply": reply, "next": {"action": "x"}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    assert snoo_replay.previous_reply(tmp_path, 3) == reply
    assert snoo_replay.previous_reply(tmp_path, 4) is None


def test_same_reply_ignores_map_url():
    a = {"action": "move", "map_url": "/v1/snoo/map/a/topdown.png"}
    assert snoo_replay.same_reply(a, {"action": "move", "map_url": "/v1/snoo/map/b/topdown.png"})
    assert not snoo_replay.same_reply(a, {"action": "done"})
