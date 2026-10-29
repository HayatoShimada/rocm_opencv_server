import json
import socket
import threading
import time

import cv2
import numpy as np
import pytest

from app import snoo
from app.snoo import HEADER, Detection, FrameError, SnooMemory, encode_reply, parse_frame


def _packet(seq=7, t=123456789012, jpeg=b"\xff\xd8jpeg"):
    return HEADER.pack(b"SNO1", seq, t, 640, 480) + jpeg


def test_parse_frame():
    frame = parse_frame(_packet())
    assert (frame.seq, frame.t_capture_ns, frame.width, frame.height) == (7, 123456789012, 640, 480)
    assert frame.jpeg == b"\xff\xd8jpeg"


@pytest.mark.parametrize("data", [b"SNO1", HEADER.pack(b"XXXX", 1, 2, 640, 480)])
def test_parse_frame_rejects_bad_packets(data):
    with pytest.raises(FrameError):
        parse_frame(data)


def test_reply_echoes_header_and_allows_null_snoo():
    frame = parse_frame(_packet())
    reply = json.loads(encode_reply(frame, [Detection((0.1, 0.2, 0.3, 0.4), 0.912345)], 8.21))
    assert reply == {
        "v": 1,
        "seq": 7,
        "t_capture_ns": 123456789012,
        "infer_ms": 8.21,
        "dets": [{"box": [0.1, 0.2, 0.3, 0.4], "cat": 0.9123, "snoo": None}],
    }


class _FakeDetector:
    device = "cpu"

    def detect(self, image, memory=None, stages=None):
        assert image.shape == (480, 640, 3)
        if stages is not None:
            stages.mark("検出")
        return [Detection((0.4, 0.3, 0.6, 0.7), 0.9)]


def test_serve_replies_even_when_jpeg_is_broken():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()
    threading.Thread(
        target=snoo.serve, args=("127.0.0.1", port, _FakeDetector()), daemon=True
    ).start()

    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(2)
    jpeg = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))[1].tobytes()
    for _ in range(50):  # サーバーが bind するまで送り直す
        client.sendto(_packet(seq=1, jpeg=jpeg), ("127.0.0.1", port))
        try:
            reply = json.loads(client.recv(65535))
            break
        except TimeoutError:
            continue
    assert reply["seq"] == 1 and reply["dets"][0]["cat"] == 0.9

    client.sendto(_packet(seq=2, jpeg=b"not a jpeg"), ("127.0.0.1", port))
    reply = json.loads(client.recv(65535))
    assert reply["seq"] == 2 and reply["dets"] == []


def test_background_boxes_stay_off_the_cat():
    from scripts.snoo_gallery import _background_boxes, _overlaps

    cat = (0.3, 0.3, 0.7, 0.7)
    boxes = _background_boxes([cat], np.random.default_rng(0), 6)
    assert boxes
    assert not any(_overlaps(b, cat) for b in boxes)
    assert _overlaps((0.0, 0.0, 0.3, 0.3), cat)  # 猫の箱の外でも、広げた毛の分にかかれば使わない


def test_snoo_memory_reuses_overlapping_boxes_for_a_few_frames():
    memory = SnooMemory(frames=2)
    first = [Detection((0.40, 0.30, 0.60, 0.70), 0.9)]
    assert memory.reuse(first) == [0]
    first[0].snoo = 0.8
    memory.remember(first, [0])
    for _ in range(2):  # 少しずれた同じ猫は CLIP を回さない。重ならない箱は回す
        dets = [Detection((0.42, 0.31, 0.62, 0.71), 0.9), Detection((0.0, 0.0, 0.1, 0.1), 0.5)]
        assert memory.reuse(dets) == ([1] if _ == 0 else [])
        assert dets[0].snoo == 0.8
        if _ == 0:
            dets[1].snoo = 0.1
            memory.remember(dets, [1])
    dets = [Detection((0.42, 0.31, 0.62, 0.71), 0.9)]
    assert memory.reuse(dets) == [0]  # frames 回使い回したら CLIP をかけ直す


def test_stages_summary_is_in_a_fixed_order_with_p50_and_p95():
    from app.snoo import Stages

    stages = Stages()
    stages.samples["送信"] = [0.1]
    stages.samples["デコード"] = [float(v) for v in range(1, 21)]  # 1..20
    assert stages.summary() == "デコード 10.50/19.00  送信 0.10/0.10"
    assert stages.summary() == ""  # 出したら消える
    stages.mark("検出")  # start の前は記録しない
    assert stages.summary() == ""


def test_serve_does_not_count_a_failed_decode_as_send_time(monkeypatch):
    from app.snoo import Stages

    def decode(jpeg):
        if jpeg == b"slow broken":
            time.sleep(0.05)
            raise FrameError("JPEG を読めない")
        return np.zeros((480, 640, 3), np.uint8)

    monkeypatch.setattr(snoo, "decode_jpeg", decode)
    stages = Stages()
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()
    threading.Thread(
        target=snoo.serve, args=("127.0.0.1", port, _FakeDetector(), stages), daemon=True
    ).start()

    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(2)
    for _ in range(50):  # サーバーが bind するまで送り直す
        client.sendto(_packet(seq=1, jpeg=b"\xff\xd8ok"), ("127.0.0.1", port))
        try:
            client.recv(65535)
            break
        except TimeoutError:
            continue
    client.sendto(b"XXXX", ("127.0.0.1", port))  # 読めないヘッダーには返さない
    client.sendto(_packet(seq=2, jpeg=b"slow broken"), ("127.0.0.1", port))
    assert json.loads(client.recv(65535))["dets"] == []
    for _ in range(100):  # 返事の後の mark を待つ
        if len(stages.samples["送信"]) >= 2:
            break
        time.sleep(0.01)
    assert len(stages.samples["デコード"]) == len(stages.samples["送信"]) - 1
    assert max(stages.samples["送信"]) < 40  # 50 ms かかったデコードは入っていない


def test_allow_list_accepts_only_listed_senders():
    from app.snoo import _allowed, parse_allow

    allow = parse_allow("192.168.0.188, 100.64.0.0/10")
    assert _allowed(("192.168.0.188", 5000), allow)
    assert _allowed(("100.73.14.8", 5000), allow)
    assert not _allowed(("192.168.0.50", 5000), allow)
    assert _allowed(("192.168.0.50", 5000), [])  # 空ならすべて


def test_serve_ignores_senders_not_in_the_allow_list():
    from app.snoo import parse_allow

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()
    threading.Thread(
        target=snoo.serve,
        args=("127.0.0.1", port, _FakeDetector()),
        kwargs={"allow": parse_allow("10.0.0.1")},
        daemon=True,
    ).start()
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(0.3)
    jpeg = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))[1].tobytes()
    for _ in range(5):
        client.sendto(_packet(seq=1, jpeg=jpeg), ("127.0.0.1", port))
        with pytest.raises(TimeoutError):  # 127.0.0.1 は受ける送り元に無いので返事が無い
            client.recv(65535)


def test_serve_touches_the_active_file_while_frames_arrive(tmp_path):
    active = tmp_path / "active"
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()
    threading.Thread(
        target=snoo.serve,
        args=("127.0.0.1", port, _FakeDetector()),
        kwargs={"active_file": str(active)},
        daemon=True,
    ).start()
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(0.3)
    jpeg = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))[1].tobytes()
    for _ in range(50):
        client.sendto(_packet(seq=1, jpeg=jpeg), ("127.0.0.1", port))
        try:
            client.recv(65535)
            break
        except TimeoutError:
            continue
    assert active.exists() and time.time() - active.stat().st_mtime < 5
