# 探索の 1 歩（地図のサーバー ⇄ Pi）

Pi（snoo_trucar）が止まって首を振って撮った分（station）を `POST /v1/snoo/map/station` で送り、
地図のサーバー（`app/snoo_map_server.py`）が次の 1 歩を返す。道はサーバーがすべて決め、Pi はそのとおり
走る。Pi に残すのは反射の安全だけ（clear_cm・guard・動かない・電池・USR）。

## Pi ができること（meta の `supports`）

| 書いたもの | 返す 1 歩の形 |
| --- | --- |
| 無し | 前への 1 本の弧 `{steer_deg, distance_cm}` |
| `"segments"` | `segments: [{steer_deg, distance_cm}, …]`（3 つまで。1 歩の中は同じ向き） |
| `"reverse"` | 後ろへの 1 歩も返す（返事の `"reverse": true`。後ろへは合わせて 30 cm まで） |
| `"long_step"`（と `"segments"`） | 前への合計を 60 cm まで |
| `"gear_segments"`（と `"reverse"`・`"segments"`） | 区切りごとに前・後ろを変えられる（5 つまで）。下の形 |

`gear_segments` を書かない Pi には、同じ計画器（`app/snoo_maneuver.py`）が 1 方向だけの列を返す
（今の Pi と同じ形で走れる）。

## 返事（`gear_segments` のとき）

```json
{
  "action": "move",
  "steer_deg": -28.3, "distance_cm": 20.0,
  "speed": 12, "reverse_speed": 8,
  "segments": [
    {"steer_deg": -28.3, "distance_cm": 20.0, "reverse": false,
     "end_pose": {"x": 0.31, "y": 0.12, "yaw_deg": 64.0}, "min_clear_cm": null},
    {"steer_deg": 28.3, "distance_cm": 15.0, "reverse": true,
     "end_pose": {"x": 0.28, "y": 0.02, "yaw_deg": 95.1}, "min_clear_cm": null},
    {"steer_deg": 0.0, "distance_cm": 30.0, "reverse": false,
     "end_pose": {"x": 0.25, "y": 0.32, "yaw_deg": 95.1}, "min_clear_cm": 35.0}
  ],
  "pose": {"x": 0.19, "y": 0.26, "yaw_deg": 116.8},
  "target": {"x": 0.25, "y": 0.32, "yaw_deg": 95.1},
  "planner": "mppi", "gap": false
}
```

- 向きは **区切りごとの `"reverse": bool`**（符号付きの距離にはしない。`distance_cm` はいつも正）
- 上の `steer_deg`・`distance_cm` は最初の区切り（segments を知らない Pi でも同じに動く）。返事の
  `"reverse": true` は、全部の区切りが後ろへのときだけ付く
- `speed` は前への区切り、`reverse_speed` は後ろへの区切りの速さ
- `end_pose` はその区切りを走り終えたときのカメラの真下の姿勢（セッションの座標。見込み）
- `min_clear_cm` は前への区切りで Pi が止まる clear_cm の下限。地図から見込んだ通り道の空きより 15 cm
  近くに物が見えたら（地図に無い物・猫）止まるよう、`min(見込み − 15, 40)` を渡す。Pi は
  `max(explore_clear_stop_cm, min_clear_cm)` より近く見えたら止まる（サーバーは止まる近さを縮められない）。
  25 cm 以下になるとき・ハンドルが大きい（Pi の clear_cm が画角の外）区切り・後ろへの区切りは `null`
- 合わせて：前へは `max_step_m`（40 cm、`long_step` なら 60 cm）、後ろへは 30 cm、1 つの区切りは 30 cm
  （ハンドルが 15° より大きい前への区切りは 20 cm：Pi の clear_skip の上限）まで
- `"rescan": true`（長さ 0、`segments: []`）：走れる列が無い。動かずに撮り直す。同じ所が続けば
  `done stuck` で終える
- `"retrace": true`：位置合わせが崩れ、位置のグラフでも決まらないとき、いま走った 1 歩を逆にたどる
  （区切りの順と向きを逆に）

## done の理由

`no_frontier`（境目が無く、確信度の低い縁も少ない）・`stuck`（直近 6 か所が半径 20 cm に固まり、行き先へ
つながる道が無い）・`step_limit`・`low_battery`・`bad_registration`・`too_little_floor`。

## Pi が返す走った 1 歩（次の station の `prev_step`）

```json
{"steer_deg": 0.0, "speed": 12.0, "reverse": false, "seconds": 3.1, "est_distance_cm": 33.0,
 "stop_reason": "clear",
 "segments": [{"steer_deg": -28.3, "distance_cm": 20.0, "reverse": false},
              {"steer_deg": 28.3, "distance_cm": 13.0, "reverse": true}]}
```

- `segments` は実際に走った区切りまで（止まった区切りは止まった所まで）。区切りごとに `reverse`
- サーバーは区切りごとの向きで、位置のグラフの走った辺（`snoo_posegraph.odom_edge_moves`）・走った所の床
  （`roommap.mark_driven_moves`）・VGGT の位置に合わせた走った長さ（`roommap.fit_moves`）を出す
- `stop_reason` が `clear`・`stuck` なら、止まった所の先を「たぶん障害物」にする（`roommap.mark_blocked_ahead`。
  カメラの足もとは写らないので、そこにある物が地図に入らず、同じ 1 歩を出し続けていた）。足すのは超音波と同じ
  弱い証拠（log-odds +0.3。1 回では障害物にならない）で、床と見たセルは変えない。止まった区切りは、
  `est_distance_cm` までに走った区切りの最後。直近 6 か所で止まった分だけ足す
- 区切りに `reverse` が無い（前の版の Pi）なら `prev_step.reverse`、それも無ければ、そのとき返した
  返事（`reply.json`）の `reverse` で決める

## 超音波

当てにならないので、Pi は超音波では止まらない。測れたときだけ meta に `"ultrasonic": {"cm", "t_ns"}` を付ける。
サーバーは、測れた近い値（3 cm〜1.5 m）だけを、前の扇形（±15°）のその距離に「たぶん障害物」として弱く
（log-odds +0.3。1 回では障害物にならない）足す。床にはしない。カメラの障害物と合えば確信度を上げ、
カメラの床と食い違えば、近づいて確かめる行き先にする（`snoo_belief.Confidence.add_sonar`）。

## 地図の層（room.json の layers）

- `confidence.png`：確信度（0 = 見ていない、1〜255 = 0〜1）。見た回数と近さ（遠いほど弱い）から
- `texture.png`：手ざわり（写真の勾配。0〜255 = 0〜1）。0.35 以上を目印にする
- /twin の「確信度」「目印」で重ねて見られる
