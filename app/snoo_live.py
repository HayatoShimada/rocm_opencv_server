"""検出サーバーに届いた最新のフレームと返事を、送り元ごとに 1 秒に 1 回ほどファイルに書く。

ブラウザで見る画面（地図のサーバーの /twin）に、今カメラに写っている物・猫の箱・
snoo らしさ・clear_cm を出すため。検出サーバーは Docker、地図のサーバーはホストのプロセスで
別々に動くので、共有のディレクトリ（compose.yaml の /run/snoo。tmpfs なので SSD を減らさない）を
間に置く。

- <root>/<送り元の IP>.jpg：届いた JPEG をそのまま（作り直さない）
- <root>/<送り元の IP>.json：seq・撮った時刻・受けた時刻・返事（dets・infer_ms・clear の欄）・
  撮った時刻の指令（版 2 のヘッダーだけ）

検出を遅らせないよう、書き込みは別のスレッドで行う。前の書き込みがまだ終わっていなければ、
そのフレームは捨てる（溜めない）。読む側が半端なファイルを読まないよう、別の名前で書いてから
名前を変える。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.snoo import Detection, Frame

from app import atomic

log = logging.getLogger("snoo")

# 書く所。空なら書かない
LIVE_DIR = os.getenv("SNOO_LIVE_DIR", "")
# 送り元ごとに書く間の最短（秒）。人が見るだけなので 1 秒で足りる
LIVE_INTERVAL_S = float(os.getenv("SNOO_LIVE_INTERVAL_S", "1.0"))


def sender_name(addr: tuple) -> str:
    """ファイルの名前にする送り元。ポートは Pi が起動し直すと変わるので入れない。"""
    return str(addr[0]).replace(":", "-")


class LiveWriter:
    def __init__(self, root: str | Path, interval_s: float = LIVE_INTERVAL_S, clock=time.monotonic):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.interval = interval_s
        self.clock = clock
        self._due: dict[str, float] = {}
        self._lock = threading.Lock()
        self._job = None
        self._wake = threading.Event()
        self.written = self.skipped = 0
        self._failed = False
        threading.Thread(target=self._run, daemon=True, name="snoo-live").start()

    def offer(
        self,
        addr: tuple,
        frame: Frame,
        dets: list[Detection],
        infer_ms: float,
        clear: dict | None = None,
    ) -> bool:
        """返事を送った後に、フレームごとに呼ぶ（軽い）。書くことにしたら True。"""
        name = sender_name(addr)
        now = self.clock()
        with self._lock:
            if now < self._due.get(name, -1e18):
                return False
            if self._job is not None:  # 前のがまだ書けていない
                self.skipped += 1
                return False
            self._due[name] = now + self.interval
            self._job = (name, frame, dets, infer_ms, clear, time.time())
        self._wake.set()
        return True

    def flush(self, timeout: float = 5.0) -> None:
        """書き込みが終わるまで待つ（テスト用）。"""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                if self._job is None:
                    return
            time.sleep(0.005)

    def _run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                job = self._job
            if job is None:
                continue
            try:
                self._write(*job)
                self.written += 1
                self._failed = False
            except OSError as e:
                if not self._failed:  # 続けて失敗しても 1 回だけ書く
                    log.warning("今のフレームを %s に書けない: %s", self.root, e)
                self._failed = True
            finally:
                with self._lock:
                    self._job = None

    def _write(self, name, frame, dets, infer_ms, clear, t_recv) -> None:
        info = {
            "sender": name,
            "seq": frame.seq,
            "t_capture_ns": frame.t_capture_ns,
            "t_recv": round(t_recv, 3),
            "size": [frame.width, frame.height],
            "image": f"{name}.jpg",
            "infer_ms": round(infer_ms, 2),
            "dets": [
                {
                    "box": [round(v, 4) for v in d.box],
                    "cat": round(d.cat, 4),
                    "snoo": None if d.snoo is None else round(d.snoo, 4),
                }
                for d in dets
            ],
        }
        if frame.steer_deg is not None:  # 版 2 のヘッダーなら、撮った時刻の指令も載せる
            info["cmd"] = {
                "steer_deg": round(frame.steer_deg, 2),
                "speed": round(frame.speed, 2),
                "pan_deg": round(frame.pan_deg, 2),
                "tilt_deg": round(frame.tilt_deg, 2),
                "moving": frame.moving,
                "flags": frame.flags,
            }
        if clear is not None:
            info.update(clear)
        # 画像を先に置く（JSON の seq を見て画像を取りに来るので）
        atomic.write_bytes(self.root / f"{name}.jpg", frame.jpeg)
        atomic.write_bytes(
            self.root / f"{name}.json", json.dumps(info, ensure_ascii=False).encode()
        )
