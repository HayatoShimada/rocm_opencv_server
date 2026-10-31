"""テストの分け方（README の「開発の流れ」）。

- 重いテスト（作り物の部屋で間取りを作る・シミュレーターを何周も回す）に slow の印を付ける。
  `make test-fast`（pytest -m "not slow"）はそれを飛ばす。印はテストのファイルではなくここに書く
  （テストのファイルを直している人とぶつからないように）。名前を変えたテストは、ここも直す
- `make test-fast` で、slow でないのに SLOW_CANDIDATE_S 秒を超えたテストは最後に名前を出す
  （ここに足す候補）
- 環境変数 PYTEST_SHARD=i/n（0 始まり）で、テストを n 個に分けたうちの i 番目だけを走らせる。
  `make test` が n 個の pytest を同時に動かすのに使う。下の秒数で、重さが揃うように分ける
"""

import os

import pytest

# テストの名前 → 1 回の秒数のめやす（2026-10-29、ほかの pytest と同時に回して測った）
SLOW = {
    "test_segments_only_for_a_pi_that_supports_them": 34,
    "test_no_reverse_for_a_pi_that_does_not_support_it": 34,
    "test_explore_runs_a_few_cycles_without_bumping": 22,
    "test_explore_with_sonar_segments_and_retrace": 22,
    "test_next_best_view_with_segments_keeps_the_first_segment_on_top": 18,
    "test_two_stations_return_the_next_step": 17,
    "test_reverse_retraces_the_last_arc_by_the_distance_vggt_measured": 17,
    "test_reverse_without_segments_is_one_arc": 17,
    "test_floor_ignores_a_big_table_above_the_camera": 9,
    "test_first_step_runs_a_useful_distance_toward_the_path": 9,
    "test_next_best_view_heads_toward_the_unseen_side_without_leaving_the_floor": 9,
    "test_ultrasonic_marks_an_obstacle_the_camera_did_not_see": 8,
    "test_scale_and_floor": 8,
    "test_short_episode_scores_and_rewards": 4,
    "test_unknown_height_station_is_measured": 8,
    "test_decide_returns_segments_only_when_asked": 8,
}
# slow でないのにこれより長くかかったテストは、最後に名前を出す
SLOW_CANDIDATE_S = 3.0

_durations: dict[str, float] = {}


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 重い（make test-fast では飛ばす）")


def _name(item) -> str:
    return getattr(item, "originalname", None) or item.name


def pytest_collection_modifyitems(config, items):
    for item in items:
        if _name(item) in SLOW:
            item.add_marker(pytest.mark.slow)
    shard = os.getenv("PYTEST_SHARD")
    if not shard:
        return
    i, n = (int(x) for x in shard.split("/"))
    keep = set(map(id, split(items, n)[i]))
    config.hook.pytest_deselected(items=[it for it in items if id(it) not in keep])
    items[:] = [it for it in items if id(it) in keep]


def split(items, n: int) -> list[list]:
    """重いものから順に、いちばん軽い組に入れる（並びはもとのまま）。"""
    order = sorted(range(len(items)), key=lambda k: -SLOW.get(_name(items[k]), 0.05))
    loads = [0.0] * n
    groups: list[list[int]] = [[] for _ in range(n)]
    for k in order:
        g = min(range(n), key=lambda j: loads[j])
        groups[g].append(k)
        loads[g] += SLOW.get(_name(items[k]), 0.05)
    return [[items[k] for k in sorted(g)] for g in groups]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    import time

    t = time.perf_counter()
    yield
    _durations[item.nodeid] = time.perf_counter() - t


def pytest_terminal_summary(terminalreporter, config):
    slow = [
        (t, nid)
        for nid, t in _durations.items()
        if t > SLOW_CANDIDATE_S and nid.rsplit("::", 1)[-1].split("[")[0] not in SLOW
    ]
    if slow:
        terminalreporter.section(
            "slow の印が無い重いテスト（tests/conftest.py の SLOW に足す候補）"
        )
        for t, nid in sorted(slow, reverse=True):
            terminalreporter.write_line(f"{t:6.1f} 秒  {nid}")
