# 手で動かす（teleop）：サーバーの API と Pi 側の仕様

人がブラウザ（ゲームパッド・キーボード・スマホの指）で PiCar-X を動かす。ブラウザは地図のサーバー
（app/snoo_map_server.py、ポート 8720）の `/drive` を開き、10 回/秒で指令を送る。Pi は 10 回/秒で
いちばん新しい指令を取りに来て、**自分の安全の層を通してから**書く。

```
ブラウザ /drive ──POST /v1/snoo/teleop（10 回/秒）──▶ 地図のサーバー（指令は 1 つだけ持つ）
                                                         ▲
Pi（mode = teleop）──GET /v1/snoo/teleop/latest（10 回/秒）┘  → guard・電池・超音波・clear_cm → モーター
```

サーバー側はできている（app/snoo_teleop.py、tests/test_snoo_teleop.py）。この文書の「Pi 側」の節は、
snoo_trucar で作るものを、迷わずに作れるよう決めたもの。

## サーバーの API

秒はどれもサーバーの時計で数える（Pi とブラウザの時計は合わせなくてよい）。送り元の制限
（`SNOO_MAP_ALLOW`）は地図のサーバーと同じ。

### POST /v1/snoo/teleop（ブラウザ）

```json
{"seq": 1730277600123, "steer": 0.4, "speed": 1.0, "pan": 0.0, "tilt": -0.2, "mode": null}
```

- `steer`・`speed`・`pan`・`tilt` は -1..1（外れたら 422）。サーバーが Pi の単位に直して上限で切る：

  | 欄 | -1 | +1 | Pi の欄 |
  |---|---|---|---|
  | speed | -10（後ろ） | 15（前、約 25 cm/s） | `speed`（picarx の -100..100、1 あたり約 1.7 cm/s） |
  | steer | -28°（左） | +28°（右） | `steer_deg`（右が +） |
  | pan | -35°（左） | +35°（右） | `pan_deg`（右が +） |
  | tilt | -20°（下） | +30°（上） | `tilt_deg`（上が +、0 が水平） |

  負と正で倍率が違う（speed -0.5 は -5、+0.5 は 7.5）。値は 0.1 に丸める
- `seq` は送るたびに大きくする（画面は `Date.now()` から数える）。前に受けた seq 以下は捨てる
  （`accepted: false, reason: "old_seq"`。届く順が入れ替わっても、古い指令で上書きしない）
- `mode` は Pi に頼むモード（`"teleop"` か `"idle"`）。頼むだけで、Pi が `ui_mode` に対応していれば変わる
- 返事は `/status` と同じ形に `accepted` を足したもの

### POST /v1/snoo/teleop/stop（ブラウザ）

体は `{"reason": "button"}` か空（空でも `Content-Length: 0` を付ける。地図のサーバーは長さの無い POST を 411 で断る）。すぐ `speed` を 0 にする（ハンドル・首はそのまま）。seq を見ない。
次に新しい seq の指令が来るまで止まったまま。

### GET /v1/snoo/teleop/latest（Pi）

クエリ（どれも省いてよい。分からない値は送らない）：

| 名前 | 中身 |
|---|---|
| `after` | 前に受けた `seq`（初めは 0） |
| `mode` | Pi の今のモード（`MODES` の名前） |
| `battery_v` | 電池の電圧（V） |
| `clear_cm` | Pi が今使っている clear_cm（無い・古いなら送らない） |
| `sonar_cm` | 今の超音波（測れていない -1 なら送らない） |
| `steer_deg`・`speed`・`pan_deg`・`tilt_deg` | **Pi が今実際に書いている値**（安全の層と電池を通した後） |
| `limit` | 手の指令より弱めた理由。`,` でつなぐ（下の「limit の名前」） |
| `supports` | `teleop,ui_mode` のように、Pi ができること |

返事：

```json
{
  "seq": 42, "changed": true,
  "command": {"steer_deg": 11.2, "speed": 15.0, "pan_deg": 0.0, "tilt_deg": -4.0},
  "age_s": 0.061, "stale": false, "stop": false, "reason": "",
  "operator": {"present": true, "last_post_age_s": 0.06},
  "request_mode": null, "request_id": 0,
  "deadman_s": 0.3, "poll_hz": 10.0
}
```

- `age_s`：サーバーが指令を受けてからの秒。`None`（null）はまだ一度も受けていない
- `stale`：`age_s` が 0.3 秒（`deadman_s`）より古い。そのとき `command.speed` は 0 にしてある
  （ハンドル・首はそのまま）
- `stop`：`stale` か、stop を受けた。`command.speed` は 0
- `reason`：`deadman`（古い）・stop の理由（`button`・`release`・`hidden`・`blur` など）・空
- `request_mode`・`request_id`：画面が頼んだモードと、頼んだ回の通し番号（頼みは 5 秒だけ返す）
- `changed`：`seq > after`。同じ指令をまた受けているだけなら false（それでも `age_s` は進む）

### GET /v1/snoo/teleop/status（画面）

いちばん新しい指令・操縦する人がいるか（最後の POST から 2 秒以内）・Pi の様子（最後の問い合わせの
クエリ、`connected`＝1 秒以内に来た、`accepted`＝`connected` かつ mode が teleop）・上限。

## Pi 側の仕様（snoo_trucar で作る）

### 1. モード teleop

- `controller.MODES` の**後ろに** `"teleop"` を足す（共有メモリではモードは番号なので、前に入れない。
  番号は 6）。`STATE_NAMES` に `TELEOP = 6` を足す
- 入り方：
  - USR を 6 回押す（`buttons.CLICK_MODES[6] = "teleop"`、`LED_BLINKS["teleop"] = 6`）
  - `snoo_trucar run --mode teleop`（cli の `choices=MODES` で自然に入る）
  - 画面から（`ui_mode` に対応したとき）：問い合わせの返事の `request_id` が前に受けた番号より大きく、
    `request_mode` が `teleop` か `idle` なら、`runner.set_mode(request_mode)`。番号を覚えるのは
    プロセスの中だけ（起動したときは、最初の返事の `request_id` を「受けた」ことにして、前の頼みで
    モードを変えない）
- 出方：USR の長押し（今の `event_to_mode("long") == "idle"` のまま）、画面の「idle に戻す」、
  ほかのクリック（ほかのモードに替わる）
- teleop に入ったとき・出たときは、速さ 0・ハンドル 0 から始める（前のモードの指令を持ち越さない）

### 2. 問い合わせ（メインプロセスのスレッド）

- 宛先：`config.server.map_url`（`http://192.168.0.248:8720/v1/snoo/map/station`）の scheme・host・port に
  `/v1/snoo/teleop/latest` を付けたもの
- 間隔：mode が teleop の間は **0.1 秒**（10 Hz、`poll_hz`）。teleop でない間は、`ui_mode` に
  対応するなら 1 秒（画面にモードを出し、頼みを受けるため）、しないなら問い合わせない
- 1 回の待ちは 0.25 秒まで（`timeout=0.25`）。遅れても次の問い合わせを重ねない（1 本のスレッドで順に）
- `after` には最後に受けた `seq` を渡す。クエリには上の表の値を入れる（`steer_deg` などは
  `runner.last_cmd` の実際に書いた値。電池で走らないときは `speed=0`）
- 受けた指令を、共有メモリの新しい `SeqBuffer`（`teleop`）に書く：
  `(t_recv_ns, seq, steer_deg, speed, pan_deg, tilt_deg, ok)`。`t_recv_ns` は**Pi が返事を受けた
  monotonic の時刻**、`ok` は `1.0 if (not stop and not stale and age_s is not None and age_s <= 0.3) else 0.0`
- 失敗（つながらない・タイムアウト・200 でない・JSON でない）のときは書かない（古くなって止まる）

### 3. deadman（制御プロセス）

teleop の間、`Controller.step` は `teleop` の共有メモリの最新の値を見て、次の**どれか**なら
`speed = 0`（ハンドル・首は最後の値のまま）にする：

1. 最後に受けた返事が 0.3 秒より古い（`now_ns - t_recv_ns > 300_000_000`。問い合わせが失敗し続けている・
   スレッドが止まった）
2. `ok == 0`（サーバーが stale・stop と言った）
3. まだ一度も受けていない

いちばん長く走り続けるのは、サーバーの 0.3 秒＋Pi の 0.3 秒で約 0.6 秒（15 の速さで約 15 cm）。

### 4. いつも効く安全（teleop でも外さない）

`Controller.step` の teleop の枝で、次の順に当てる。どれかが弱めたら、その名前を `limit` に入れる。

1. **上限**（サーバーと同じものをもう一度）：`speed` は -10..15、`steer` は ±28°、`pan` は
   `pan_min..pan_max`（±35°）、`tilt` は -20..30°。（`ControlConfig.max_speed` が小さければそちら）
2. **guard**（snoo を怖がらせない層。今の `self.guard.apply(...)` をそのまま通す）：
   - 渡す snoo の距離は、ほかのモードと同じく見えている間（`since < coast_s`）だけ
   - `playing` は explore と同じく `est.valid and since < lost_s`（snoo が近くにいる間だけ遊びの時間を数える）
   - 30 cm より自分から近づかない・速く離れたら下がる（BACKOFF は**後ろへ動いてよい**。人の指令より
     guard を先にする）・HOLD・REST は走らない。弱めたら `guard`
3. **電池**：`drive_ok` が False（`battery_low_v` より低い・測れない）なら `speed = 0`（首は動かす）。`battery`
4. **超音波**：測れた近い値（`stop_distance_cm` = 20 cm より近い）が `US_NEAR_STOP_NS`（0.1 秒、
   15 Hz で 2 回）続いたときだけ、前へ（`speed > 0`）を 0 にする。**-1・None・古い値（`us_stale`）は
   「分からない」で、止めない**（この超音波は -1 が多く、外れ値も出る）。後ろへはそのまま。`sonar`
   - teleop の間は、`step` の最後の「1 回の近い値で `speed = min(speed, 0)`」を当てない
     （確かめた近い値の決まりに置き換える）
5. **clear_cm**：`clear_state == CLEAR_VALUE and clear_cm < 20` なら前へを 0。null・古い（STALE）・
   skip（分からない）ときは、前へを `CLEAR_SKIP_SPEED`（10）までに落とす。後ろへはそのまま。`clear`
6. 首はこれまでどおり `PAN_RATE`（120°/秒）で寄せる

### 5. 返す（実際に書いた指令）

次の問い合わせのクエリに、実際にモーターに書いた値（`runner._actuate_loop` が `car.apply` に渡した
値。電池で 0 にしたなら 0）と `limit` を入れる。`limit` の名前：

| 名前 | 意味 |
|---|---|
| `deadman` | サーバーの指令が古い（stale）・まだ無い |
| `poll` | 問い合わせが 0.3 秒届いていない |
| `stopped` | サーバーが stop と言った |
| `guard` | guard が弱めた・下がっている・待っている |
| `battery` | 電池で走らない |
| `sonar` | 超音波の近い値で前へを止めた |
| `clear` | clear_cm で前へを止めた・落とした |
| `mode` | teleop でない（問い合わせだけしている） |

### 6. Pi で足すテスト（案）

- 新しい指令が来なくなって 0.3 秒で speed が 0 になる。首は動かない
- `ok == 0`（stale・stop）なら走らない
- clear_cm 15 cm で前へは 0、後ろへは走る。clear が null なら前へは 10 まで
- 超音波 -1 が続いても止まらない。15 cm が 1 回だけなら止まらず、2 回続けば止まる
- snoo が 35 cm に見えていて前へ 15 を頼むと、guard が 30 cm の手前で止める
- 電池が少ないと speed 0 で、首は動く
- `request_id` が増えたときだけモードを変える。長押しで idle にした後、同じ頼みで teleop に戻らない

## 画面（/drive）

- 地図のサーバーの `/drive`。このマシンなら `http://127.0.0.1:8720/drive`
- 今のカメラ（`/v1/snoo/agent/frame.jpg` を 7 回/秒まで）・小さな地図（`map.png`、2 秒ごと）・
  モード・電池・clear_cm・snoo の距離・Pi が teleop で受けているか
- 入力：ゲームパッド（左のスティック＝前後とハンドル、右のスティック＝首、A＝鳴らす、B＝止める、
  Y＝首を正面に）、キーボード（WASD・矢印・Space で止める・H で鳴らす）、スマホの 2 本の指のスティック
- 送り方：入力がある間は 10 回/秒、離してから 0.5 秒は 0 を送り、そのあと stop を 1 回。大きな「止める」
  のボタン、画面が隠れた・窓から外れた・閉じるときも stop
- 「首を保つ」を入れると、右のスティックは首を回す速さになり、離してもその向きのまま
- **スマホ・別の PC から開く**：認証が無いので、その機械の IP を `~/.config/snoo-map.env` の
  `SNOO_MAP_ALLOW` に足して `make deploy-map`（入っていない機械のブラウザには、足し方を書いた 403 の
  画面が出る）
