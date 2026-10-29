"""ファイルを書きかけで読ませないよう、別の名前に書いてから名前を変える。

検出（Docker）・地図のサーバー・SLAM（別の venv）のどれからも読めるよう、標準のものだけで書く。
"""

from pathlib import Path


def write_bytes(path: Path, data: bytes) -> None:
    """path に data を書く。読む側は、前の中身か新しい中身のどちらかだけを見る。"""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    write_bytes(path, text.encode())
