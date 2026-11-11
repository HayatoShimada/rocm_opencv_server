"""着用イメージの PoC の一覧（data/vton/index.html）。スタッフが1枚ずつ見て点を付ける用。

  uv run python -m scripts.vton_contact_sheet

1 行に、服の正面の写真・下敷き・できた画像・尺度の検査の結果（ずれ cm）と、説明文の案を並べる。
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from scripts.vton_tryon import RESULTS, ROOT

CAPTION = (
    "着用イメージ（AI で合成）。身長 {height}cm のスタッフに、この服の実寸（{garment}）どおりの"
    "大きさで合成しました。裾と袖口の位置は、実寸から計算した位置との差が {tol:g}cm 以内のもの"
    "だけを載せています。"
)


def caption(job: dict) -> str:
    c = job.get("check", {}).get("caption") or {}
    garment = "・".join(f"{k} {v:g}" for k, v in (c.get("garment") or {}).items()) + "cm"
    return CAPTION.format(
        height=c.get("person_height"), garment=garment, tol=c.get("tolerance_cm", 2)
    )


def build(jobs: list[dict]) -> str:
    rows = []
    for job in jobs:
        rel = f"{job['person']}/{job['handle']}"
        check = job.get("check") or {}
        dev = check.get("deviations_cm") or {}
        verdict = "使える" if check.get("passes") else ("使わない" if check else "未検査")
        notes = "".join(f"<li>{html.escape(n)}</li>" for n in job.get("notes", []))
        gen = job.get("generate") or {}
        imgs = "".join(
            f'<figure><img src="{rel}/{f}" loading="lazy"><figcaption>{label}</figcaption></figure>'
            for f, label in (("garment.jpg", "服"), ("guide.png", "下敷き"), ("out.jpg", "合成"))
            if (ROOT / rel / f).exists()
        )
        dev_text = html.escape(json.dumps(dev, ensure_ascii=False))
        cls = "ok" if check.get("passes") else "ng"
        rows.append(
            f"""<section>
  <h2>{html.escape(job["title"])} <small>{html.escape(job["handle"])} × {job["person"]}</small></h2>
  <div class="imgs">{imgs}</div>
  <p class="v {cls}">{verdict}　ずれ: {dev_text}
  　{gen.get("seconds", "–")} 秒・VRAM {gen.get("vram_gb", "–")} GB</p>
  <ul>{notes}</ul>
  <p class="cap">{html.escape(caption(job)) if check.get("passes") else ""}</p>
  <p>評価（1〜5）：柄・色 [ ]　形 [ ]　顔・手 [ ]　丈の見え方 [ ]</p>
</section>"""
        )
    return f"""<!doctype html><meta charset="utf-8"><title>着用イメージ PoC</title>
<style>
body {{ font-family: sans-serif; margin: 24px; background: #f5f4ef; }}
section {{ margin-bottom: 40px; }}
.imgs {{ display: flex; gap: 12px; }}
figure {{ margin: 0; }}
img {{ height: 420px; }}
.v.ok {{ color: #2e7d50; font-weight: 700; }}
.v.ng {{ color: #c0392b; font-weight: 700; }}
.cap {{ max-width: 760px; font-size: 13px; }}
</style>
<h1>着用イメージ（AI で合成）PoC</h1>
{"".join(rows)}
"""


def main() -> None:
    jobs = json.loads(Path(RESULTS).read_text())
    out = ROOT / "index.html"
    out.write_text(build(jobs))
    print(f"{out} に書いた（{len(jobs)} 件）")


if __name__ == "__main__":
    main()
