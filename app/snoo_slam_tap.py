"""走っている間のフレームを、SLAM（app/snoo_slam.py）に渡すためにファイルに書く。

検出サーバーは Docker、SLAM はホストのプロセス（別の venv）で動くので、共有のディレクトリ
（compose.yaml の /run/snoo。tmpfs なので SSD を減らさない）を間に置く。

- SNO2 のヘッダーの flags に MOVING があるフレームだけ書く（止まって首を振っている間の画像は、
  地図のサーバーがステーションとして残すので、SLAM はそちらを読む）
- 撮った時刻（Pi の t_capture_ns）で間引き、hz 枚/秒まで書く（既定 4）
- 書き込みは別のスレッドで行う。前の書き込みがまだ終わっていなければ、そのフレームは捨てる
  （溜めない。返事を遅らせない）
- 送り元ごとに新しい max_files 枚だけ残し、古いものは消す（SLAM が止まっていても tmpfs が
  あふれない）。SLAM は名前（撮った時刻）で読んだ所を覚えるので、こちらが消してよい

置き場の形（読む側は app/snoo_slam.read_queue）:

    <root>/<送り元の IP>/<t_capture_ns を 20 桁>.jpg   届いた JPEG をそのまま
    <root>/<送り元の IP>/<t_capture_ns を 20 桁>.json  下の欄。画像を書いてから置く

json の欄: v（1）・sender・seq・t_capture_ns・t_recv（サーバーの UNIX 時刻）・size [幅, 高さ]・
image（jpg の名前）・steer_deg・speed・pan_deg・tilt_deg・flags・moving・reversing・
dets（[{box: [x1, y1, x2, y2]（幅・高さに対する比）, cat, snoo}]。SLAM は猫の所を使わない）。
読む側が半端なファイルを読まないよう、別の名前で書いてから名前を変える。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from app.snoo_live import sender_name

if TYPE_CHECKING:
    from app.snoo import Detection, Frame

log = logging.getLogger("snoo")

# 書く所。空なら書かない
SLAM_TAP_DIR = os.getenv("SNOO_SLAM_TAP_DIR", "")
# 1 秒に何枚まで書くか（撮った時刻で数える）。20 cm/s なら 4 枚/秒で 5 cm ごと
SLAM_TAP_HZ = float(os.getenv("SNOO_SLAM_TAP_HZ", "4"))
# 送り元ごとに残す枚数。4 枚/秒で 75 秒分、1 枚 40 KB ほどなので 12 MB
SLAM_TAP_MAX_FILES = int(os.getenv("SNOO_SLAM_TAP_MAX_FILES", "300"))

MOVING, REVERSING = 1, 2  # app/snoo.py の flags のビット（読み込みを軽くするためここにも書く）


def file_stem(t_capture_ns: int) -> str:
    """撮った時刻の名前。桁をそろえて、名前の順 = 撮った順にする。"""
    return f"{int(t_capture_ns):020d}"


class SlamTap:
    def __init__(
        self,
        root: str | Path,
        hz: float = SLAM_TAP_HZ,
        max_files: int = SLAM_TAP_MAX_FILES,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # 0 以下なら間引かない（届いたものをみな書く。保存したセッションを流し直すとき）
        self.interval_ns = int(1e9 / hz) if hz > 0 else 0
        self.max_files = max_files
        self._due: dict[str, int] = {}
        self._lock = threading.Lock()
        self._job = None
        self._wake = threading.Event()
        self.written = self.skipped = 0
        self._failed = False
        threading.Thread(target=self._run, daemon=True, name="snoo-slam-tap").start()

    def offer(self, addr: tuple, frame: Frame, dets: list[Detection]) -> bool:
        """返事を送った後に、フレームごとに呼ぶ（軽い）。書くことにしたら True。"""
        if frame.steer_deg is None or not frame.flags & MOVING:  # SNO1・止まっている間は書かない
            return False
        name = sender_name(addr)
        with self._lock:
            due = self._due.get(name)
            # Pi が起動し直すと t_capture_ns は小さくなる。そのときは数え直す
            if due is not None and frame.t_capture_ns < due and due - frame.t_capture_ns < 60e9:
                return False
            if self._job is not None:  # 前のがまだ書けていない
                self.skipped += 1
                return False
            self._due[name] = frame.t_capture_ns + self.interval_ns
            self._job = (name, frame, dets, time.time())
        self._wake.set()
        return True

    def flush(self, timeout: float = 5.0) -> None:
        """書き込みが終わるまで待つ（テスト・流し直し用）。"""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                if self._job is None:
                    return
            time.sleep(0.002)

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
                    log.warning("SLAM のフレームを %s に書けない: %s", self.root, e)
                self._failed = True
            finally:
                with self._lock:
                    self._job = None

    def _write(self, name: str, frame: Frame, dets: list[Detection], t_recv: float) -> None:
        d = self.root / name
        d.mkdir(exist_ok=True)
        stem = file_stem(frame.t_capture_ns)
        info = {
            "v": 1,
            "sender": name,
            "seq": frame.seq,
            "t_capture_ns": frame.t_capture_ns,
            "t_recv": round(t_recv, 3),
            "size": [frame.width, frame.height],
            "image": f"{stem}.jpg",
            "steer_deg": round(frame.steer_deg, 2),
            "speed": round(frame.speed, 2),
            "pan_deg": round(frame.pan_deg, 2),
            "tilt_deg": round(frame.tilt_deg, 2),
            "flags": frame.flags,
            "moving": bool(frame.flags & MOVING),
            "reversing": bool(frame.flags & REVERSING),
            "dets": [
                {
                    "box": [round(v, 4) for v in det.box],
                    "cat": round(det.cat, 4),
                    "snoo": None if det.snoo is None else round(det.snoo, 4),
                }
                for det in dets
            ],
        }
        _replace(d / f"{stem}.jpg", frame.jpeg)
        _replace(d / f"{stem}.json", json.dumps(info, ensure_ascii=False).encode())
        self._trim(d)

    def _trim(self, d: Path) -> None:
        """古いものから消して max_files 枚にする。"""
        names = sorted(p.name for p in d.glob("*.json"))
        for old in names[: max(0, len(names) - self.max_files)]:
            stem = old[: -len(".json")]
            for suffix in (".json", ".jpg"):
                (d / f"{stem}{suffix}").unlink(missing_ok=True)


def _replace(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
