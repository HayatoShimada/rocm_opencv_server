"""細分類の判定（data/item-subtype/predictions.json）を、規則の候補と比べて集計する。

  uv run python -m scripts.item_subtype_report --subtypes <subtypes.json> --guess <rule-guess.json>

data/item-subtype/report.json（集計）と、店主が見る一覧 index.html を書く。
集計では、題名の語で高い確からしさで決めた規則の候補（flannel・S/S・fleece・coat など）を
正解の代わりにして、
写真だけの判定がどのくらい当たるか（写真を見られているかの目安）も出す。ブランドは集計で見るだけで、判定には使わない。
"""

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from app import item_subtype

DATA = Path("data/item-subtype")
MODES = item_subtype.MODES
MODE_NAMES = {"photo": "写真だけ", "text": "テキストあり"}
# 規則が「題名の語」で決めたと言える理由（rule-guess の rule_reason）
TITLE_REASON = "題名の語"


def pick(pred: dict, mode: str) -> tuple[str, float] | None:
    a = (pred.get(mode) or {}).get("answers", {}).get(item_subtype.CHOICE)
    return (a["choice"], a["confidence"]) if a else None


def rows(predictions: dict, guess: dict) -> list[dict]:
    """判定と規則の候補を、商品ごとに1行にする（どちらも揃う商品だけ）。"""
    out = []
    for gid, pred in predictions.items():
        g = guess.get(gid)
        if not g:
            continue
        out.append(
            {
                "id": gid,
                **g,
                "images": pred.get("images", []),
                "photo": pick(pred, "photo"),
                "text": pick(pred, "text"),
                "answers": (pred.get("photo") or pred.get("text") or {}).get("answers", {}),
            }
        )
    return out


def agreement(rs: list[dict], mode: str, only=lambda r: True) -> dict:
    """規則の候補と同じ細分類だった割合。品目ごとと全体。"""
    n, ok = Counter(), Counter()
    for r in rs:
        if r[mode] and only(r):
            for key in (r["product_type"], "all"):
                n[key] += 1
                ok[key] += r[mode][0] == r["rule_code"]
    return {k: {"n": n[k], "ok": ok[k], "rate": round(ok[k] / n[k], 3)} for k in n}


def confusion(rs: list[dict], mode: str, only=lambda r: True) -> dict:
    """品目 → 規則の候補 → Clef の判定 → 件数。"""
    table = defaultdict(lambda: defaultdict(Counter))
    for r in rs:
        if r[mode] and only(r):
            table[r["product_type"]][r["rule_code"]][r[mode][0]] += 1
    return {t: {a: dict(c) for a, c in rows_.items()} for t, rows_ in table.items()}


def by_code(rs: list[dict], mode: str, only) -> dict:
    """規則の細分類ごとの、Clef が同じと答えた割合（再現率）。"""
    n, ok = Counter(), Counter()
    for r in rs:
        if r[mode] and only(r):
            key = f"{r['product_type']}/{r['rule_code']}"
            n[key] += 1
            ok[key] += r[mode][0] == r["rule_code"]
    return {k: {"n": n[k], "ok": ok[k], "rate": round(ok[k] / n[k], 3)} for k in sorted(n)}


def title_based(r: dict) -> bool:
    return r.get("rule_reason") == TITLE_REASON and r.get("rule_confidence") == "high"


def is_dress(r: dict, mode: str) -> bool:
    return bool(r[mode]) and r[mode][0] == "dress_shirt"


def dress_summary(rs: list[dict], mode: str) -> dict:
    dress = [r for r in rs if is_dress(r, mode)]
    weak_rule = [r for r in rs if r["rule_code"] == "cotton_shirt"]
    brands = Counter(r["brand"] or "（不明）" for r in dress)
    all_brands = Counter(r["brand"] or "（不明）" for r in rs if r["product_type"] == "Shirts")
    return {
        "dress_count": len(dress),
        "rule_dress_count": sum(1 for r in rs if r["rule_code"] == "dress_shirt"),
        "rule_cotton_total": len(weak_rule),
        "rule_cotton_to_dress": sum(1 for r in weak_rule if is_dress(r, mode)),
        "rule_cotton_kept": sum(1 for r in weak_rule if r[mode] and not is_dress(r, mode)),
        "rule_dress_agreed": sum(
            1 for r in rs if r["rule_code"] == "dress_shirt" and is_dress(r, mode)
        ),
        "brands_dress": dict(brands.most_common()),
        "brands_shirts_total": dict(all_brands.most_common()),
    }


def confidence_bands(rs: list[dict], mode: str, only) -> list[dict]:
    """確からしさの帯ごとの、規則（題名の語）との一致。自動の候補にしてよい下限を探す材料。"""
    out = []
    for lo, hi in ((0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        sel = [r for r in rs if r[mode] and only(r) and lo <= r[mode][1] < hi]
        out.append(
            {
                "from": lo,
                "to": min(hi, 1.0),
                "n": len(sel),
                "ok": sum(1 for r in sel if r[mode][0] == r["rule_code"]),
            }
        )
    return out


def build(rs: list[dict]) -> dict:
    report = {"products": len(rs)}
    for mode in MODES:
        report[mode] = {
            "agreement": agreement(rs, mode),
            "confusion": confusion(rs, mode),
            "title_based_agreement": agreement(rs, mode, title_based),
            "title_based_by_code": by_code(rs, mode, title_based),
            "title_based_confusion": confusion(rs, mode, title_based),
            "title_based_bands": confidence_bands(rs, mode, title_based),
            "dress": dress_summary(rs, mode),
            "mean_confidence": round(
                sum(r[mode][1] for r in rs if r[mode]) / max(1, sum(1 for r in rs if r[mode])), 3
            ),
        }
    both = [r for r in rs if r["photo"] and r["text"]]
    report["photo_text_agree"] = {
        "n": len(both),
        "ok": sum(1 for r in both if r["photo"][0] == r["text"][0]),
    }
    return report


def label(code: str, vocabulary: dict, product_type: str | None = None) -> str:
    for t, entries in vocabulary.items():
        if product_type in (None, t):
            for e in entries:
                if e["code"] == code:
                    return e["label"]
    return code


def _card(r: dict, vocabulary: dict, tags: str) -> str:
    e = html.escape
    imgs = "".join(f'<img src="{e(u)}&width=240" loading=lazy>' for u in r["images"])

    def verdict(mode: str) -> str:
        if not r[mode]:
            return "—"
        code, conf = r[mode]
        return f"{e(label(code, vocabulary, r['product_type']))} <small>{conf:.2f}</small>"

    differ = r["photo"] and r["photo"][0] != r["rule_code"]
    a = r["answers"]
    aux = " ・ ".join(
        f"{k} {v if not isinstance(v, dict) else round(v['score'], 2)}"
        for k, v in a.items()
        if k != item_subtype.CHOICE
    )
    rule = label(r["rule_code"], vocabulary, r["product_type"])
    return (
        f'<article class="{"ng" if differ else ""}"><div class=imgs>{imgs}</div><div>'
        f'<h2><a href="https://shop.85-store.com/products/{e(r["handle"])}" target=_blank>'
        f"{e(r['title'])}</a></h2>"
        f"<p><small>{e(r['product_type'])}{tags}</small></p>"
        f"<table><tr><th>写真だけ</th><td>{verdict('photo')}</td></tr>"
        f"<tr><th>テキストあり</th><td>{verdict('text')}</td></tr>"
        f"<tr><th>規則</th><td>{e(rule)} "
        f"<small>{e(r['rule_reason'])}・{e(r['rule_confidence'])}</small></td></tr>"
        f"<tr><th>補助</th><td><small>{e(aux)}</small></td></tr></table></div></article>"
    )


STYLE = """
body{font-family:system-ui,sans-serif;margin:16px;color:#222;background:#fafafa}
article{display:flex;gap:16px;background:#fff;border:1px solid #ddd;border-radius:8px;
  padding:12px;margin:12px 0}
article.ng{border-color:#e0a030}
.imgs{display:flex;gap:6px;flex:none}
.imgs img{width:160px;height:200px;object-fit:cover;border-radius:4px}
h1{font-size:20px} h2{font-size:15px;margin:0 0 4px} p{margin:4px 0;font-size:13px;color:#555}
h3{margin-top:28px;border-bottom:2px solid #ccc}
table{border-collapse:collapse;font-size:13px;margin-top:6px}
th{text-align:left;padding:2px 10px 2px 0;color:#666;font-weight:normal;white-space:nowrap}
td{padding:2px 10px 2px 0} small{color:#888}
.sum td,.sum th{border:1px solid #ddd;padding:3px 8px}
@media (max-width:640px){article{flex-direction:column}}
"""


def write_html(rs: list[dict], report: dict, vocabulary: dict) -> Path:
    e = html.escape
    # 判定の根拠の既定は写真だけ。ドレスシャツと判定した物 → 境目 → 規則と食い違う物
    dress = sorted((r for r in rs if is_dress(r, "photo")), key=lambda r: -r["photo"][1])
    seen = {r["id"] for r in dress}
    border = sorted(
        (
            r
            for r in rs
            if r["photo"] and r["photo"][1] < item_subtype.BORDERLINE and r["id"] not in seen
        ),
        key=lambda r: r["photo"][1],
    )
    seen |= {r["id"] for r in border}
    differ = [
        r for r in rs if r["photo"] and r["photo"][0] != r["rule_code"] and r["id"] not in seen
    ]
    differ.sort(key=lambda r: -r["photo"][1])
    sections = [
        ("写真だけでドレスシャツと判定した物", dress),
        (f"境目の物（確からしさ {item_subtype.BORDERLINE} 未満）", border),
        ("規則の候補と食い違う物（上と重ならない分）", differ),
    ]
    head = []
    for mode in MODES:
        ag = report[mode]["agreement"].get("all")
        tb = report[mode]["title_based_agreement"].get("all")
        if ag:
            head.append(
                f"{MODE_NAMES[mode]}: 規則と一致 {ag['ok']}/{ag['n']}（{ag['rate']:.0%}）"
                + (f"・題名の語で決めた物 {tb['ok']}/{tb['n']}（{tb['rate']:.0%}）" if tb else "")
            )
    d = report["photo"]["dress"]
    brand_rows = "".join(
        f"<tr><td>{e(b)}</td><td>{n}</td><td>{d['brands_shirts_total'].get(b, 0)}</td></tr>"
        for b, n in d["brands_dress"].items()
    )
    body = [
        f"<h1>細分類の判定（Clef）</h1><p>{e(' / '.join(head))}</p>",
        f"<p>ドレスシャツ（写真だけ）{d['dress_count']} 点。"
        "ブランド別（判定にブランドは使っていない）:</p>"
        f"<table class=sum><tr><th>ブランド</th><th>ドレスと判定</th><th>シャツ全体</th></tr>"
        f"{brand_rows}</table>",
    ]
    for title, items in sections:
        body.append(f"<h3>{e(title)}（{len(items)} 点）</h3>")
        body += [_card(r, vocabulary, f" ・ {e(r['brand'] or '')}") for r in items]
    page = (
        "<!doctype html><html lang=ja><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>細分類の判定（Clef）</title><style>{STYLE}</style>{''.join(body)}</html>"
    )
    path = DATA / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtypes", type=Path, required=True)
    parser.add_argument("--guess", type=Path, required=True)
    args = parser.parse_args()
    vocabulary = item_subtype.load_vocabulary(args.subtypes)
    guess = json.loads(args.guess.read_text())
    predictions = json.loads((DATA / "predictions.json").read_text())
    rs = rows(predictions, guess)
    report = build(rs)
    (DATA / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    path = write_html(rs, report, vocabulary)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"レビュー用: {path.resolve()}")


if __name__ == "__main__":
    main()
