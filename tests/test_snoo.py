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


def test_parse_frame_v2_keeps_the_commands():
    from app.snoo import HEADER_V2

    data = HEADER_V2.pack(b"SNO2", 9, 123, 640, 480, -12.5, 15.0, 10.0, -15.0, 1) + b"\xff\xd8x"
    frame = parse_frame(data)
    assert (frame.seq, frame.t_capture_ns, frame.jpeg) == (9, 123, b"\xff\xd8x")
    commands = (frame.steer_deg, frame.speed, frame.pan_deg, frame.tilt_deg)
    assert commands == (-12.5, 15.0, 10.0, -15.0)
    assert frame.moving
    assert not parse_frame(_packet()).moving  # 版 1 は走っていない扱い


@pytest.mark.parametrize(
    "data", [b"SNO1", b"SNO2" + bytes(10), HEADER.pack(b"XXXX", 1, 2, 640, 480)]
)
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


# --- 走っている間の前の空き（clear_cm） ---


def _packet_v2(seq=7, t=10**12, jpeg=b"\xff\xd8jpeg", flags=1, steer=0.0, pan=0.0, tilt=-15.0):
    from app.snoo import HEADER_V2

    return HEADER_V2.pack(b"SNO2", seq, t, 640, 480, steer, 15.0, pan, tilt, flags) + jpeg


class _FakeEstimator:
    """見積もりの代わり。delay 秒かかり、同時に回った数の最大を数える。"""

    def __init__(self, cm=42.0, conf=0.8, delay=0.0):
        self.cm, self.conf, self.delay = cm, conf, delay
        self.calls: list[tuple] = []
        self.running = self.max_running = 0
        self.lock = threading.Lock()

    def estimate(self, image, tilt_deg, pan_deg, steer_deg):
        from app.clearance import Clearance

        with self.lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        time.sleep(self.delay)
        with self.lock:
            self.running -= 1
            self.calls.append((tilt_deg, pan_deg, steer_deg))
        return Clearance(self.cm, self.conf, 13.0, "both")


def _worker(estimator, **kw):
    kw.setdefault("interval_ms", 0)  # 間を空けずに回す（間のテストだけ別に渡す）
    worker = snoo.ClearanceWorker(lambda: estimator, **kw)
    assert worker.wait_ready(5)
    return worker


def _wait_idle(worker, timeout=2.0):
    end = time.monotonic() + timeout
    while worker._busy and time.monotonic() < end:
        time.sleep(0.005)
    assert not worker._busy


IMAGE = np.zeros((480, 640, 3), np.uint8)
JPEG = cv2.imencode(".jpg", IMAGE)[1].tobytes()
ADDR = ("127.0.0.1", 5000)


def test_clear_fields_are_absent_for_sno1_and_frames_not_moving():
    worker = _worker(_FakeEstimator())
    sno1, still = parse_frame(_packet()), parse_frame(_packet_v2(flags=0))
    for frame in (sno1, still):
        assert worker.fields(ADDR, frame) is None
        assert not worker.submit(ADDR, frame, IMAGE)  # 見積もりもしない
        reply = json.loads(encode_reply(frame, [], 1.0, worker.fields(ADDR, frame)))
        assert not any(k.startswith("clear") for k in reply)
    assert worker.estimator.calls == []


def test_clear_fields_are_absent_when_the_model_cannot_be_built():
    def build():
        raise RuntimeError("no model")

    worker = snoo.ClearanceWorker(build)
    assert not worker.wait_ready(5)
    assert worker.fields(ADDR, parse_frame(_packet_v2())) is None


def test_moving_frames_get_the_latest_value_from_an_earlier_frame():
    estimator = _FakeEstimator(cm=42.04, conf=0.83)
    worker = _worker(estimator)
    first = parse_frame(_packet_v2(seq=1, t=10**12, steer=-12.0, pan=-12.0, tilt=-15.0))
    # まだ見積もっていない最初のフレームは null（欄はある）
    assert worker.fields(ADDR, first) == snoo.NO_CLEAR
    assert worker.submit(ADDR, first, IMAGE)
    _wait_idle(worker)
    assert estimator.calls == [(-15.0, -12.0, -12.0)]  # tilt・pan・舵はヘッダーの値
    second = parse_frame(_packet_v2(seq=2, t=10**12 + 16_700_000))
    assert worker.fields(ADDR, second) == {
        "clear_cm": 42.0,
        "clear_conf": 0.83,
        "clear_seq": 1,
        "clear_blind_cm": 13.0,
    }
    assert worker.fields(("10.0.0.9", 5000), second) == snoo.NO_CLEAR  # 送り元ごとに分ける


def _at(t0, ms, flags=1):
    return parse_frame(_packet_v2(seq=9, t=t0 + int(ms * 1e6), flags=flags))


def test_old_low_confidence_and_reversing_values_become_null():
    worker = _worker(_FakeEstimator())
    t0 = 10**12
    worker.submit(ADDR, parse_frame(_packet_v2(seq=1, t=t0)), IMAGE)
    _wait_idle(worker)
    assert worker.fields(ADDR, _at(t0, 150))["clear_cm"] == 42.0
    assert worker.fields(ADDR, _at(t0, 151)) == snoo.NO_CLEAR  # 150 ms より古い
    assert worker.fields(ADDR, _at(t0, -1)) == snoo.NO_CLEAR  # Pi が起動し直して時刻が戻った
    assert worker.fields(ADDR, _at(t0, 10, flags=1 | 2)) == snoo.NO_CLEAR  # 下がっている

    low = _worker(_FakeEstimator(conf=0.3), min_conf=0.5)
    low.submit(ADDR, parse_frame(_packet_v2(seq=1, t=t0)), IMAGE)
    _wait_idle(low)
    fields = low.fields(ADDR, _at(t0, 10))
    assert fields["clear_cm"] is None and fields["clear_conf"] == 0.3


def test_only_one_estimate_runs_at_a_time_and_busy_frames_are_dropped():
    estimator = _FakeEstimator(delay=0.05)
    worker = _worker(estimator)
    frames = [parse_frame(_packet_v2(seq=i, t=10**12 + i)) for i in range(20)]
    accepted = sum(worker.submit(ADDR, f, IMAGE) for f in frames)
    assert accepted == 1  # 回っている間に来た 19 枚は溜めずに捨てる
    _wait_idle(worker)
    assert worker.submit(ADDR, parse_frame(_packet_v2(seq=99, t=10**12 + 99)), IMAGE)
    _wait_idle(worker)
    assert estimator.max_running == 1 and len(estimator.calls) == 2
    assert worker._results[ADDR].seq == 99  # 終わった後に来た最新のフレームを使う


def test_estimates_start_at_most_once_per_interval():
    worker = _worker(_FakeEstimator(), interval_ms=200)
    frame = parse_frame(_packet_v2(seq=1))
    assert worker.submit(ADDR, frame, IMAGE)
    _wait_idle(worker)
    assert not worker.submit(ADDR, frame, IMAGE)  # 終わっていても、200 ms 経つまでは回さない
    time.sleep(0.21)
    assert worker.submit(ADDR, frame, IMAGE)


def _serve_with(clearance, stages=None):
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()
    threading.Thread(
        target=snoo.serve,
        args=("127.0.0.1", port, _FakeDetector(), stages),
        kwargs={"clearance": clearance},
        daemon=True,
    ).start()
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(2)
    return client, port


def _ask(client, port, packet):
    for _ in range(50):  # サーバーが bind するまで送り直す
        client.sendto(packet, ("127.0.0.1", port))
        try:
            return json.loads(client.recv(65535))
        except TimeoutError:
            continue
    raise AssertionError("返事が無い")


def test_serve_puts_clear_on_moving_replies_only():
    from app.snoo import Stages

    stages = Stages()
    worker = _worker(_FakeEstimator())
    client, port = _serve_with(worker, stages)
    still = _ask(client, port, _packet(seq=1, jpeg=JPEG))
    assert "clear_cm" not in still and still["dets"][0]["cat"] == 0.9
    first = _ask(client, port, _packet_v2(seq=2, t=10**12, jpeg=JPEG))
    assert first["clear_cm"] is None  # まだ見積もっていない
    _wait_idle(worker)
    second = _ask(client, port, _packet_v2(seq=3, t=10**12 + 16_700_000, jpeg=JPEG))
    assert (second["clear_cm"], second["clear_seq"]) == (42.0, 2)
    assert second["dets"][0]["cat"] == 0.9
    assert len(stages.samples["clear"]) >= 1  # 見積もりの時間も区間に入る


def test_serve_does_not_wait_for_a_slow_estimate():
    estimator = _FakeEstimator(delay=0.5)
    worker = _worker(estimator)
    client, port = _serve_with(worker)
    _ask(client, port, _packet_v2(seq=1, t=10**12, jpeg=JPEG))  # 見積もりが 0.5 秒回り始める
    times = []
    for i in range(2, 12):
        t = time.perf_counter()
        reply = _ask(client, port, _packet_v2(seq=i, t=10**12 + i * 16_700_000, jpeg=JPEG))
        times.append(time.perf_counter() - t)
        assert reply["seq"] == i and "clear_cm" in reply
    assert max(times) < 0.2  # 0.5 秒の見積もりを待たずに返す
    assert estimator.max_running == 1
