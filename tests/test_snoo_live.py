"""app/snoo_live.py（最新のフレームと返事をファイルに書く）を、作り物の検出で試す（GPU なし）。"""

import json
import threading
import time

import cv2
import numpy as np
from test_snoo import _ask, _FakeDetector, _packet, _packet_v2

from app import snoo
from app.snoo import Detection, parse_frame
from app.snoo_live import LiveWriter

JPEG = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))[1].tobytes()
PI = ("192.168.0.188", 40000)


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_writes_the_frame_and_reply_once_per_interval(tmp_path):
    clock = _Clock()
    live = LiveWriter(tmp_path, interval_s=1.0, clock=clock)
    frame = parse_frame(_packet_v2(seq=5, t=10**12, jpeg=JPEG, steer=-12.0, pan=10.0))
    dets = [Detection((0.1, 0.2, 0.3, 0.4), 0.91234, 0.7)]
    clear = {"clear_cm": 42.0, "clear_conf": 0.9, "clear_seq": 4, "clear_blind_cm": 12.0}
    assert live.offer(PI, frame, dets, 4.321, clear)
    live.flush()
    assert (tmp_path / "192.168.0.188.jpg").read_bytes() == JPEG
    info = json.loads((tmp_path / "192.168.0.188.json").read_text())
    assert info["seq"] == 5 and info["t_capture_ns"] == 10**12
    assert info["image"] == "192.168.0.188.jpg" and info["size"] == [640, 480]
    assert info["infer_ms"] == 4.32
    assert info["dets"] == [{"box": [0.1, 0.2, 0.3, 0.4], "cat": 0.9123, "snoo": 0.7}]
    assert info["cmd"]["steer_deg"] == -12.0 and info["cmd"]["pan_deg"] == 10.0
    assert info["cmd"]["moving"] is True
    assert info["clear_cm"] == 42.0 and info["clear_seq"] == 4
    assert abs(info["t_recv"] - time.time()) < 5
    assert not list(tmp_path.glob(".*"))  # 書きかけのファイルを残さない

    # 1 秒経つまでは書かない。別の送り元は別に数える
    clock.t += 0.5
    assert not live.offer(PI, frame, [], 1.0)
    assert live.offer(("100.73.14.8", 1), parse_frame(_packet(seq=1, jpeg=JPEG)), [], 1.0)
    live.flush()
    sno1 = json.loads((tmp_path / "100.73.14.8.json").read_text())
    assert "cmd" not in sno1 and "clear_cm" not in sno1  # 版 1・clear の欄が無いときは載せない
    clock.t += 0.5
    assert live.offer(PI, parse_frame(_packet_v2(seq=9, jpeg=JPEG)), [], 1.0)
    live.flush()
    assert json.loads((tmp_path / "192.168.0.188.json").read_text())["seq"] == 9


class _SlowDisk(LiveWriter):
    """書き込みに 0.5 秒かかる置き場。"""

    def __init__(self, *a, **kw):
        self.started = threading.Event()
        super().__init__(*a, **kw)

    def _write(self, *job):
        self.started.set()
        time.sleep(0.5)
        super()._write(*job)


def test_frames_are_dropped_while_the_previous_write_runs(tmp_path):
    live = _SlowDisk(tmp_path, interval_s=0.0)
    frame = parse_frame(_packet(seq=1, jpeg=JPEG))
    assert live.offer(PI, frame, [], 1.0)
    assert live.started.wait(1)
    t = time.perf_counter()
    assert not live.offer(PI, frame, [], 1.0)  # 待たずに捨てる
    assert not live.offer(("100.73.14.8", 1), frame, [], 1.0)
    assert time.perf_counter() - t < 0.05
    assert live.skipped == 2
    live.flush()
    assert live.offer(PI, frame, [], 1.0)


def test_serve_does_not_wait_for_a_slow_write(tmp_path):
    import socket

    live = _SlowDisk(tmp_path, interval_s=0.0)
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()
    threading.Thread(
        target=snoo.serve,
        args=("127.0.0.1", port, _FakeDetector()),
        kwargs={"live": live},
        daemon=True,
    ).start()
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(2)
    _ask(client, port, _packet(seq=1, jpeg=JPEG))  # 0.5 秒の書き込みが始まる
    times = []
    for i in range(2, 12):
        t = time.perf_counter()
        reply = _ask(client, port, _packet(seq=i, jpeg=JPEG))
        times.append(time.perf_counter() - t)
        assert reply["seq"] == i
    assert max(times) < 0.2  # 書き込みを待たずに返す
    live.flush()
    info = json.loads((tmp_path / "127.0.0.1.json").read_text())
    assert info["seq"] >= 1 and info["dets"][0]["cat"] == 0.9
