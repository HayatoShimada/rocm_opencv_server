import json
import socket
import threading

import cv2
import numpy as np
import pytest

from app import snoo
from app.snoo import HEADER, Detection, FrameError, encode_reply, parse_frame


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

    def detect(self, image):
        assert image.shape == (480, 640, 3)
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
