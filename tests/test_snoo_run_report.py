import json

from test_snoo_map_server import _client, _meta, _post

from scripts import snoo_run_report


def test_report_on_a_fake_session(tmp_path):
    c = _client(tmp_path)
    for i in range(2):
        _post(c, _meta(i))
    session = tmp_path / "explore_20261029_201500"
    lines = snoo_run_report.report(session)
    text = "\n".join(lines)
    assert "2 か所" in lines[0]
    assert "返事:" in text and "地図: 床 m²" in text
    # 作り物の部屋の VGGT の向きは指令のパンどおりなので、ずれは小さい
    errs = snoo_run_report.pan_errors(session)
    assert errs and max(errs) < 1.0


def test_clear_stats_counts_skips_nulls_and_differences(tmp_path):
    rows = [
        {"clear_cm": 27.0, "ultrasonic_cm": 70.0},  # 低い物（ラグの縁）
        {"clear_cm": 60.0, "ultrasonic_cm": 45.0},  # clear_cm が長い：見落としの疑い
        {"clear_cm": None, "clear_skip": "settling", "ultrasonic_cm": 80.0},
        {"clear_cm": None, "clear_skip": None, "ultrasonic_cm": 80.0},
    ]
    p = tmp_path / "clear.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    s = snoo_run_report.clear_stats(p)
    assert s["行"] == 4 and s["値あり"] == 2 and s["null"] == 1 and s["skip:settling"] == 1
    assert s["clear_cm が 25 cm 以上短い（低い物か見誤り）"] == 1
    assert s["clear_cm が 10 cm 以上長い（見落としの疑い）"] == 1
