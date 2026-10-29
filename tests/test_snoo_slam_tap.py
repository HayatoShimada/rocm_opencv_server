"""app/snoo_slam_tap.py（走っている間のフレームを SLAM に渡す）を、作り物の検出で試す
（GPU なし）。
"""

import json
import threading
import time

import cv2
import numpy as np
from test_snoo import _ask, _FakeDetector, _packet, _packet_v2

from app import snoo
from app.snoo import Detection, parse_frame
from app.snoo_slam import read_queue
from app.snoo_slam_tap import SlamTap, file_stem

JPEG = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))[1].tobytes()
PI = ("192.168.0.188", 40000)
MS = 1_000_000


def _moving(seq, t, flags=1, **kw):
    return parse_frame(_packet_v2(seq=seq, t=t, jpeg=JPEG, flags=flags, **kw))


def test_writes_moving_frames_at_the_rate_and_the_reader_gets_them_back(tmp_path):
    tap = SlamTap(tmp_path, hz=4)
    dets = [Detection((0.1, 0.2, 0.3, 0.4), 0.91234, 0.7)]
    t0 = 10**12
    wrote = []
    for i in range(30):  # 60 fps で 0.5 秒
        f = _moving(i, t0 + i * 16_667_000, steer=-12.0, pan=-18.0)
        if tap.offer(PI, f, dets if i == 0 else []):
            wrote.append(i)
        tap.flush()
    assert wrote == [0, 15]  # 250 ms ごと
    d = tmp_path / "192.168.0.188"
    info = json.loads((d / f"{file_stem(t0)}.json").read_text())
    assert info["t_capture_ns"] == t0 and info["image"] == f"{file_stem(t0)}.jpg"
    assert info["steer_deg"] == -12.0 and info["pan_deg"] == -18.0 and info["tilt_deg"] == -15.0
    assert info["moving"] is True and info["reversing"] is False and info["speed"] == 15.0
    assert info["dets"] == [{"box": [0.1, 0.2, 0.3, 0.4], "cat": 0.9123, "snoo": 0.7}]
    assert (d / f"{file_stem(t0)}.jpg").read_bytes() == JPEG
    assert not list(d.glob(".*"))  # 書きかけを残さない

    kfs = read_queue(d)
    assert [k.t_capture_ns for k in kfs] == [t0, t0 + 15 * 16_667_000]
    assert kfs[0].boxes == [[0.1, 0.2, 0.3, 0.4]] and kfs[1].boxes == []
    assert kfs[0].pan_deg == -18.0 and kfs[0].source == "tap"
    assert [k.t_capture_ns for k in read_queue(d, after_ns=t0)] == [t0 + 15 * 16_667_000]


def test_still_and_sno1_frames_are_not_written(tmp_path):
    tap = SlamTap(tmp_path, hz=0)
    assert not tap.offer(PI, _moving(1, 10**12, flags=0), [])  # 止まっている
    assert not tap.offer(PI, parse_frame(_packet(seq=2, jpeg=JPEG)), [])  # 版 1
    assert tap.offer(PI, _moving(3, 10**12 + 1, flags=3), [])  # 後ろへ走っている
    tap.flush()
    (info,) = [json.loads(p.read_text()) for p in tmp_path.glob("*/*.json")]
    assert info["reversing"] is True and info["flags"] == 3


def test_keeps_only_the_newest_files(tmp_path):
    tap = SlamTap(tmp_path, hz=0, max_files=5)
    for i in range(12):
        assert tap.offer(PI, _moving(i, 10**12 + i * 100 * MS), [])
        tap.flush()
    d = tmp_path / "192.168.0.188"
    names = sorted(p.name for p in d.iterdir())
    assert len(names) == 10  # 5 枚の jpg と json
    assert names[0] == f"{file_stem(10**12 + 7 * 100 * MS)}.jpg"


def test_a_restarted_pi_starts_counting_again(tmp_path):
    tap = SlamTap(tmp_path, hz=1)
    assert tap.offer(PI, _moving(1, 500 * 10**9), [])
    tap.flush()
    assert not tap.offer(PI, _moving(2, 500 * 10**9 + 10 * MS), [])
    assert tap.offer(PI, _moving(3, 5 * 10**9), [])  # 起動し直して時刻が小さくなった


class _SlowDisk(SlamTap):
    def __init__(self, *a, **kw):
        self.started = threading.Event()
        super().__init__(*a, **kw)

    def _write(self, *job):
        self.started.set()
        time.sleep(0.5)
        super()._write(*job)


def test_frames_are_dropped_while_the_previous_write_runs(tmp_path):
    tap = _SlowDisk(tmp_path, hz=0)
    assert tap.offer(PI, _moving(1, 10**12), [])
    assert tap.started.wait(1)
    t = time.perf_counter()
    assert not tap.offer(PI, _moving(2, 10**12 + 300 * MS), [])  # 待たずに捨てる
    assert time.perf_counter() - t < 0.05
    assert tap.skipped == 1
    tap.flush()
    assert tap.offer(PI, _moving(3, 10**12 + 600 * MS), [])


def test_serve_does_not_wait_for_a_slow_write(tmp_path):
    import socket

    tap = _SlowDisk(tmp_path, hz=0)
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()
    threading.Thread(
        target=snoo.serve,
        args=("127.0.0.1", port, _FakeDetector()),
        kwargs={"slam_tap": tap},
        daemon=True,
    ).start()
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(2)
    _ask(client, port, _packet_v2(seq=1, t=10**12, jpeg=JPEG))  # 0.5 秒の書き込みが始まる
    times = []
    for i in range(2, 12):
        t = time.perf_counter()
        reply = _ask(client, port, _packet_v2(seq=i, t=10**12 + i * 300 * MS, jpeg=JPEG))
        times.append(time.perf_counter() - t)
        assert reply["seq"] == i
    assert max(times) < 0.2  # 書き込みを待たずに返す
    tap.flush()
    info = json.loads(sorted(tmp_path.glob("*/*.json"))[0].read_text())
    assert info["seq"] == 1 and info["dets"][0]["cat"] == 0.9
