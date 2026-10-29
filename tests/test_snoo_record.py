import json

from app.snoo import Detection, Frame
from app.snoo_record import Recorder


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _frame(seq):
    return Frame(seq, seq * 1000, 640, 480, b"\xff\xd8" + bytes([seq % 256]))


CAT = [Detection((0.4, 0.3, 0.6, 0.7), 0.9, 0.8)]
NOISE = [Detection((0.0, 0.0, 0.1, 0.1), 0.3)]  # 確からしさが低い箱では場面を始めない
SHADOW = [Detection((0.0, 0.0, 0.5, 0.5), 0.87, 0.02)]  # 猫らしいが snoo らしくない影


def _clips(root):
    return sorted(p for p in root.glob("*/*") if p.is_dir())


def test_records_from_before_the_cat_until_after_it_leaves(tmp_path):
    clock = _Clock()
    rec = Recorder(tmp_path, fps=10, pre_s=0.2, post_s=0.3, min_free_gb=0, clock=clock)
    addr = ("192.168.0.188", 5000)
    seq = 0
    for dets in [[], NOISE, SHADOW, *[CAT] * 3, *[[]] * 6, []]:  # 0.1 s ごと
        rec.offer(addr, _frame(seq), dets)
        seq += 1
        clock.t += 0.1
    rec.flush()
    (clip,) = _clips(tmp_path)
    lines = [json.loads(x) for x in (clip / "frames.jsonl").read_text().splitlines()]
    # 猫の 2 枚前（pre_s 0.2 s）から、見えなくなって 0.3 s 過ぎた所まで
    assert [x["seq"] for x in lines] == list(range(1, 10))
    assert lines[2]["dets"] == [{"box": [0.4, 0.3, 0.6, 0.7], "cat": 0.9, "snoo": 0.8}]
    assert (clip / lines[0]["file"]).read_bytes() == _frame(1).jpeg  # JPEG は作り直さない
    assert rec.senders[addr].clip is None  # 場面は閉じた


def test_thins_frames_to_the_recording_fps(tmp_path):
    clock = _Clock()
    rec = Recorder(tmp_path, fps=10, min_free_gb=0, clock=clock)
    for seq in range(30):  # 30 fps で 1 s
        rec.offer(("10.0.0.1", 1), _frame(seq), CAT)
        clock.t += 1 / 30
    rec.flush()
    (clip,) = _clips(tmp_path)
    assert len((clip / "frames.jsonl").read_text().splitlines()) == 10


def test_does_not_start_a_clip_when_the_disk_is_full(tmp_path):
    rec = Recorder(tmp_path, min_free_gb=1e9)
    rec.offer(("10.0.0.1", 1), _frame(1), CAT)
    rec.flush()
    assert _clips(tmp_path) == []
