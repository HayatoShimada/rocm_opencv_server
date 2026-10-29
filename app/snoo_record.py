"""検出サーバーに届いたフレームのうち、猫が写っている間を学習用に保存する。

snoo は臆病なので、まずは車を動かさず見守りながら（watch）、いろいろな姿をためる。
Pi はもう全フレームをサーバーに送っているので、ここで保存すれば Pi の容量も通信も増えない。

- snoo（cat ≥ cat_min かつ snoo ≥ snoo_min）が写り始める pre_s 秒前から、
  写らなくなって post_s 秒後までを 1 つの場面にする
- 保存は fps 枚/秒まで間引く。JPEG は作り直さず、届いたものをそのまま書く
- 書き込みは別のスレッドで行い、検出の返事を遅らせない。追いつかなければそのフレームは捨てる
- 空きが min_free_gb を切ったら新しい場面を始めない

場面ごとに <root>/<日付>/<時刻>_<送り元>/ を作り、<seq>.jpg と frames.jsonl（1 行 1 フレーム:
seq・t_capture_ns（Pi の CLOCK_MONOTONIC。Pi の trace のパン・チルトと突き合わせる）・
t_recv（サーバーの UNIX 時刻）・file・dets）を書く。ファイルの持ち主は root の持ち主に揃える
（コンテナは root で動くので、そのままだとホストのユーザーが消せない）。
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.snoo import Detection, Frame

log = logging.getLogger("snoo")

RECORD_DIR = os.getenv("SNOO_RECORD_DIR", "")
RECORD_FPS = float(os.getenv("SNOO_RECORD_FPS", "10"))
# 場面を始める猫の確からしさと snoo らしさの下限。返す箱の下限 0.25 より高くし、
# 誤検出で場面を作らない。暗い部屋では影や机の脚が cat 0.5〜0.87 になることがあるが、
# snoo らしさは 0.07 未満だった。後ろ姿などで snoo らしさが下がっても録れるよう、
# snoo のほうは低めにする（見本が無く null なら見ない）
RECORD_CAT_MIN = float(os.getenv("SNOO_RECORD_CAT_MIN", "0.5"))
RECORD_SNOO_MIN = float(os.getenv("SNOO_RECORD_SNOO_MIN", "0.3"))
RECORD_MIN_FREE_GB = float(os.getenv("SNOO_RECORD_MIN_FREE_GB", "50"))


@dataclass
class _Clip:
    dir: Path
    last_seen: float
    frames: int = 0


@dataclass
class _Sender:
    clip: _Clip | None = None
    next_due: float = -1e9
    # 場面が始まる前の数秒分（猫が近づいてくるところも残す）
    before: deque = field(default_factory=deque)


class Recorder:
    def __init__(
        self,
        root: str | Path,
        fps: float = RECORD_FPS,
        cat_min: float = RECORD_CAT_MIN,
        snoo_min: float = RECORD_SNOO_MIN,
        pre_s: float = 1.0,
        post_s: float = 2.0,
        min_free_gb: float = RECORD_MIN_FREE_GB,
        clock=time.monotonic,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        stat = self.root.stat()
        self.owner = (stat.st_uid, stat.st_gid)
        self.interval = 1.0 / fps
        self.cat_min = cat_min
        self.snoo_min = snoo_min
        self.pre = max(1, round(pre_s * fps))
        self.post_s = post_s
        self.min_free_gb = min_free_gb
        self.clock = clock
        self.senders: dict[tuple, _Sender] = {}
        self.dropped = 0
        self._queue: queue.Queue = queue.Queue(maxsize=256)
        threading.Thread(target=self._write_loop, daemon=True, name="snoo-record").start()

    def offer(self, addr: tuple, frame: Frame, dets: list[Detection]) -> None:
        """返事を送った後に、フレームごとに呼ぶ（軽い。書き込みは別のスレッド）。"""
        now = self.clock()
        sender = self.senders.setdefault(addr, _Sender())
        # fps まで間引く。届く間隔が揺れても平均が fps になるよう、次に取る時刻を予定で進める
        # （届いた時刻から測ると、30 fps を 10 fps にするつもりが 7.5 fps になったりする）
        if now < sender.next_due - self.interval * 0.05:
            return
        if now - sender.next_due < self.interval:
            sender.next_due += self.interval
        else:  # しばらく途切れていた（溜まった分をまとめて取らない）
            sender.next_due = now + self.interval
        item = (frame, [_det(d) for d in dets], time.time())
        seen = any(
            d.cat >= self.cat_min and (d.snoo is None or d.snoo >= self.snoo_min) for d in dets
        )
        clip = sender.clip
        if clip is None:
            if not seen:
                sender.before.append(item)
                while len(sender.before) > self.pre:
                    sender.before.popleft()
                return
            if not self._has_room():
                sender.before.clear()
                return
            clip = sender.clip = _Clip(self._clip_dir(addr), now)
            self._put(("open", clip, None))
            log.info("録画を始める: %s", clip.dir)
            while sender.before:
                self._put(("frame", clip, sender.before.popleft()))
        if seen:
            clip.last_seen = now
        self._put(("frame", clip, item))
        if now - clip.last_seen > self.post_s:
            self._put(("close", clip, None))
            sender.clip = None

    def flush(self, timeout: float = 5.0) -> None:
        """書き込みが終わるまで待つ（テスト・終了用）。"""
        end = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)

    def _has_room(self) -> bool:
        free_gb = shutil.disk_usage(self.root).free / 1e9
        if free_gb < self.min_free_gb:
            log.warning("空きが %.0f GB しかないので録画しない", free_gb)
            return False
        return True

    def _clip_dir(self, addr: tuple) -> Path:
        stamp = time.localtime()
        name = f"{time.strftime('%H%M%S', stamp)}_{addr[0].replace(':', '-')}"
        return self.root / time.strftime("%Y%m%d", stamp) / name

    def _put(self, job) -> None:
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self.dropped += 1

    def _write_loop(self) -> None:
        while True:
            kind, clip, item = self._queue.get()
            try:
                if kind == "open":
                    clip.dir.mkdir(parents=True, exist_ok=True)
                    self._chown(clip.dir.parent)
                    self._chown(clip.dir)
                elif kind == "frame":
                    self._write_frame(clip, *item)
                else:
                    log.info("録画を終える: %s（%d 枚）", clip.dir, clip.frames)
            except OSError as e:
                log.warning("録画を書けない（%s）: %s", clip.dir, e)
            finally:
                self._queue.task_done()

    def _write_frame(self, clip: _Clip, frame: Frame, dets: list[dict], t_recv: float) -> None:
        name = f"{frame.seq:010d}.jpg"
        path = clip.dir / name
        path.write_bytes(frame.jpeg)
        self._chown(path)
        index = clip.dir / "frames.jsonl"
        line = {
            "seq": frame.seq,
            "t_capture_ns": frame.t_capture_ns,
            "t_recv": round(t_recv, 3),
            "file": name,
            "size": [frame.width, frame.height],
            "dets": dets,
        }
        with index.open("a") as f:
            f.write(json.dumps(line) + "\n")
        if clip.frames == 0:
            self._chown(index)
        clip.frames += 1

    def _chown(self, path: Path) -> None:
        if os.geteuid() == 0 and self.owner != (0, 0):
            os.chown(path, *self.owner)


def _det(d: Detection) -> dict:
    return {
        "box": [round(v, 4) for v in d.box],
        "cat": round(d.cat, 4),
        "snoo": None if d.snoo is None else round(d.snoo, 4),
    }
