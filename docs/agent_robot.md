# エージェントが動かすロボット（snoo の PiCar-X）

AI のエージェント（Claude Code など）が PiCar-X を自分で見て・動かして、部屋の意味のあるもの
（家具・ケーブル・snoo の好きな所）と困りごと（検出の見落とし・地図の誤り）を見つける。
人が手で何度も走らせて確かめる回数を減らすため。

この文書は、サーバー側（このリポジトリ）が何を用意したかと、Pi 側（snoo_trucar）が何を作れば
よいかを決める。Pi 側はここの形どおりに作る。

- サーバー: `app/snoo_agent_api.py`（地図のサーバー `app/snoo_map_server.py` の `/v1/snoo/agent/`、ポート 8720）
- MCP のサーバー: `scripts/snoo_robot_mcp.py`（stdio。エージェントはこれを通して上の API を使う）
- 第 1 段（今）: Pi を変えずにできることだけ。**指令は並べるが、Pi はまだ実行しない**（`pi_support: false`）

## 全体の形

```
Claude Code ──stdio(JSON-RPC)── scripts/snoo_robot_mcp.py ──HTTP──┐
                                                                   ▼
                                  地図のサーバー :8720（snoo-map、ホストの systemd）
                                  ├ /v1/snoo/map/station   Pi のステーション（今まで通り）
                                  ├ /twin                   人が見る画面
                                  └ /v1/snoo/agent/...      ここで足したもの
                                       │ 読むだけ
              ┌────────────────────────┼─────────────────────────────┐
     /run/snoo/live/<IP>.json・.jpg   data/snoo_map/sessions/<s>/    <s>/slam/status.json
     （検出の Docker が 1 秒ごと）     meta.json・reply.json・         （snoo-slam）
                                      grid.png・room.json・poses.json
                                       │ 書く
              data/snoo_agent/（commands.jsonl・command_results.jsonl・say.jsonl・tts/・
                                issues/・snapshots/）、data/snoo_map/annotations.jsonl
                                       ▲
     85pi4（snoo_trucar） ── GET commands/next（0.5 秒ごと）・POST command/<id>/result・
                             GET say/next・GET say/<名前>.wav（第 2 段で Pi に作る）
```

送り元の制限は地図のサーバーと同じ（`SNOO_MAP_ALLOW`。既定は Pi の LAN・Pi の tailscale・127.0.0.1）。
認証は無いので、このマシンの外に開かない。

## MCP の道具

| 道具 | すること |
| --- | --- |
| `state` | 今の様子（下の「state の欄」） |
| `look` | 今のカメラの画像（JPEG を画像のまま返す）と、猫の箱・距離・向き |
| `map` | 上から見た地図（PNG を画像のまま）と、画素と座標の対応・印の一覧 |
| `annotate` | 部屋に印を付ける（x・y・label・note・image_ref） |
| `annotations` | 付けた印の一覧 |
| `report_issue` | 困りごとの下書きを `data/snoo_agent/issues/<時刻>_<題>/` に書く（GitHub には出さない） |
| `say` | 日本語の文を声にして、Pi が鳴らす順番待ちに入れる |
| `command_look` | 首を向ける指令を並べる |
| `command_move` | 少し走る指令を並べる（サーバーで確かめてから） |
| `command_scan` | その場で首を振って撮り、ステーションとして送る指令を並べる |
| `command_stop` | 止める指令を並べる（いつでも） |
| `command_status` | 並べた指令の様子と Pi の結果 |

`command_*` は返事に `pi_support` を付ける。Pi が `commands/next` を 10 秒のうちに問い合わせていなければ
`false` で、「並べたが何も動かない」と書く。

### Claude Code で使う

まず地図のサーバーを、この版で起動し直しておく（`make deploy-map`）。探っている途中で起動し直さない。

プロジェクトの `.mcp.json`（例。どこにも入れていない。使うときにリポジトリの根に置く）:

```json
{
  "mcpServers": {
    "snoo-robot": {
      "type": "stdio",
      "command": "python3",
      "args": ["scripts/snoo_robot_mcp.py"],
      "env": { "SNOO_AGENT_URL": "http://127.0.0.1:8720" }
    }
  }
}
```

または、コマンドで足す（自分のユーザーだけ）:

```sh
claude mcp add snoo-robot -e SNOO_AGENT_URL=http://127.0.0.1:8720 \
    -- python3 /home/hayatoshimada/orca/rocm_opencv_server/scripts/snoo_robot_mcp.py
```

MCP のサーバーは標準のものだけで書いてあるので、どの Python 3.10 以上でも動く（venv は要らない）。

## HTTP の API（/v1/snoo/agent/）

| メソッドと道 | 中身 |
| --- | --- |
| `GET state?session=` | 今の様子（JSON） |
| `GET frame.jpg?sender=` | 今のフレーム（JPEG）。ヘッダー `X-Frame-Seq`・`X-Frame-Age-S` |
| `GET frame.json?sender=` | 今のフレームの欄（検出サーバーの live の JSON に `age_s`・`cats`・`snoo` を足したもの） |
| `GET map.png?session=` | 上から見た地図 |
| `GET map.json?session=` | 地図の画素と座標の対応・車の位置・印・凡例 |
| `POST annotate` | `{session?, x, y, label, note?, image_ref?}` → 書いた行 |
| `GET annotations?session=&label=` | 印の一覧 |
| `GET snapshots/<名前>.jpg` | 印・困りごとのために写しておいた画像 |
| `POST issue` | `{title, body, images[], session?}` → `{path, files, posted: false}` |
| `POST say` | `{text, queue?: true}` → `audio/wav`（16 kHz・モノラル・16 ビット）。ヘッダー `X-Say-Id`・`X-Say-Wav` |
| `GET say/next?since=<id>` | 鳴らす順番待ちの次の 1 件 `{say: {...} \| null, now}` |
| `GET say/<hash>.wav` | 声の wav |
| `POST command` | `{kind, ...}` → `{queued, id, kind, params, checks, pi_support, note}` |
| `GET commands/next?after=<id>&mode=&battery_v=&sonar_cm=&running=&supports=` | Pi が問い合わせる（下） |
| `POST command/<id>/result` | Pi が結果を書く（下） |
| `GET command/<id>` | 指令・今の様子・最後の結果 |
| `GET commands?limit=` | 最近の指令 |

### state の欄

分からない欄は `null`。`sources` に、どこから取ったか（`pi`・`station_meta`・`live`）を書く。

| 欄 | どこから |
| --- | --- |
| `mode`・`battery_v`・`sonar_cm` | Pi の問い合わせ（10 秒以内）、無ければ最後のステーションの meta.json（`mode`・`battery_v`・`ultrasonic_cm`）。今の Pi は超音波が壊れていて `ultrasonic_cm` は null |
| `clear_cm`・`clear_conf` | 検出サーバーの live の JSON（走っている間だけ出る） |
| `pan_deg`・`tilt_deg`・`steer_deg`・`speed`・`moving` | live の JSON の `cmd`（SNO2 のヘッダー。撮った時刻の指令） |
| `cats`・`snoo` | live の JSON の `dets`。`cat ≥ 0.35` の箱ごとに `distance_cm`・`bearing_deg`・`clipped`。`snoo` は Pi と同じ選び方（`snoo ≥ 0.6` の中で `cat × snoo` が最大） |
| `frame`・`frame_age_s`・`sender` | live の JSON（いちばん新しい送り元） |
| `session`・`station`・`station_age_s` | `data/snoo_map/sessions/` でいちばん新しく書いたセッションと、その最後のステーション |
| `last_reply`・`last_reply_station` | 最後の `reply.json`（地図のサーバーが Pi に返した 1 歩） |
| `slam` | `<s>/slam/status.json`（`x`・`y`・`yaw_deg`・`age_s`・`rmse_m`） |
| `pose` | 車の今の位置。SLAM が `ok` で status.json が 30 秒以内ならそれ、無ければ最後のステーションの位置（`source` に書く） |
| `pi_support`・`pi_status` | Pi の最後の問い合わせ |

猫の距離は、箱の下の端の真ん中（足もと）が床にあるとして、カメラの高さ 8 cm・首の向きから床の位置を出す
（`app/clearance.floor_point`）。箱が画面の下で切れていれば（`clipped`）、足もとはもっと手前なので、距離は
長めに出る。水平より上なら（家具の上・遠い）距離は null。`bearing_deg` は車の前から右が +（Pi の pan と同じ）。

### 座標

印・地図・`pose` は **セッションの座標**（m）。原点はそのセッションの最初のステーションの床、x はそのときの
車の向き、y は左、yaw は左回りが +。セッションごとに原点が違うので、印は `session` と一緒に残す
（`frame: "session"`）。部屋の座標に直すには、いずれセッションどうしの位置合わせが要る。

`map.json` の対応: 画素 (u, v) の座標は `x = x_at_u0 + u / px_per_m`、`y = y_at_v0 - v / px_per_m`。

## 指令（Pi への wire format）

### 指令の形

`POST command` で並んだ指令は、`commands/next` で Pi に次の形で渡る。

```json
{"id": 12, "t": 1793351403.9, "expires_at": 1793351418.9, "kind": "move",
 "steer_deg": -10.0, "distance_cm": 20.0, "speed": 8.0, "reverse": false,
 "session": "explore_20261030_175740",
 "from_pose": {"x": 0.18, "y": 0.07, "yaw_deg": -133.6, "source": "slam"},
 "note": "ソファの右の端を見に行く"}
```

| kind | 欄 | Pi がすること |
| --- | --- | --- |
| `look` | `pan_deg?`（右が +）・`tilt_deg?`（上が +） | 首を向け、止まるのを待つ（0.5 s ほど）。無い欄は今のまま。Pi の限り（±35°）に収める |
| `move` | `steer_deg`（右が +）・`distance_cm`（≤ 30）・`speed`（≤ 10）・`reverse` | 走る。下の「move の止め方」 |
| `scan` | なし | explore と同じ止まって首を振る撮影をし、ステーションとして `POST /v1/snoo/map/station` に送る。返ってきた 1 歩は**走らない** |
| `stop` | なし | すぐ止める。動いている指令を `cancelled`（reason `stop`）にする |

`id` は 1 から増える整数。`t`・`expires_at` はサーバーの UNIX 時刻（秒）。`from_pose`・`session` は、サーバーが
確かめたときの位置（記録用。Pi は使わなくてよい）。知らない欄は読み捨てる。

### 指令の受け方：HTTP で問い合わせる（勧める）

Pi は **`GET /v1/snoo/agent/commands/next?after=<最後に受けた id>` を 0.5 秒ごとに** 問い合わせる。

```
GET /v1/snoo/agent/commands/next?after=11&mode=agent&battery_v=7.62&sonar_cm=&running=11&supports=look,move,scan,stop,say
→ 200 {"now": 1793351404.4, "heartbeat_s": 3.0, "command": {...} | null}
```

- 問い合わせの欄は Pi の今の様子で、`state` に出る（`mode`: idle・watch・keep・follow・map・explore・agent、
  `battery_v`、`sonar_cm`（測れなければ空）、`running`: 実行中の指令の id（無ければ省く）、`supports`: 受けられる kind）
- `command` は `after` より大きい id の、まだ誰も受けていない（結果の無い）・期限の切れていない指令のうち、
  いちばん古いもの。ただし `stop` があれば、それより前のものを飛ばして最新の `stop` を返す
- 期限（`expires_at`、並べてから 15 秒。stop は 60 秒）を過ぎたものはサーバーが渡さない。サーバーが確かめた
  ときの様子から変わっているかもしれないので、遅れた move は走らない
- Pi は受けたらすぐ `accepted` を書き（下）、`after` をその id にする。Pi を起動し直したら `after=0` でよい
  （終わったもの・期限の切れたものは渡らない）
- この問い合わせが通ること自体を heartbeat にする（下）

UDP の検出の返事に載せない理由:

- 検出の返事は Docker のコンテナが返し、指令は地図のサーバー（ホストのプロセス）が持つ。載せるには
  コンテナとホストの間にもう 1 つ受け渡しが要る
- UDP の返事は落ちる・500 ms より古ければ捨てる決まりなので、1 回だけ届けばよい指令と受け取りの確認を
  載せると、送り直しと重なりの扱いが要る
- 検出の返事は、Pi が画像を送っている間（カメラが動いている間）しか来ない。idle で画像を止めると届かない
- Pi は station で既に `requests` を使って同じサーバーに HTTP で送っている。0.5 秒ごとの GET は LAN で
  数 ms・1 KB ほどで、軽い

遅れは最大で 0.5 秒（問い合わせの間）になるが、指令は 30 cm・低速の move と首の向きだけで、急ぎの止めは
Pi 側の安全（下）が受け持つので足りる。

### 結果の書き方

```
POST /v1/snoo/agent/command/<id>/result
{"status": "done", "reason": null, "detail": {...}}
```

| status | いつ |
| --- | --- |
| `accepted` | 受けた（すぐ書く） |
| `running` | 動き始めた（look・move・scan） |
| `done` | 終えた |
| `refused` | 実行しなかった（`reason`: `not_agent_mode`・`battery_low`・`guard`・`busy`・`bad_params` など） |
| `failed` | 途中で止めた・失敗した（`reason` は下の止めた理由） |
| `cancelled` | stop・ボタンで取り消した |
| `expired` | Pi が受けたときに期限が過ぎていた（サーバーの `now` で比べる。Pi の時計は使わない） |

`detail` の例:

- move: `{"moved_cm": 18.4, "stop_reason": "done", "seconds": 1.9, "steer_deg": -10, "speed": 8, "reverse": false, "t_start_ns": ..., "t_end_ns": ...}`
  （`moved_cm` は explore の `est_distance_cm` と同じ数え方。guard・電池を通した後の速さで）
- look: `{"pan_deg": 20, "tilt_deg": -10}`（実際に向けた値）
- scan: `{"session_id": "agent_20261031_100000", "station_id": 3, "reply_action": "move"}`

同じ id に何回書いてもよい（最後のものが今の様子）。サーバーは、結果の無い move がある間は次の move を並べない。

### heartbeat（3 秒）

Pi は、最後に `commands/next` が 200 で返ってから **3 秒**（返事の `heartbeat_s`）を過ぎたら、動いている
agent の指令（move・look・scan）を止め、止めた所で待つ。繋がったら、その指令に `failed`（reason `heartbeat`）
を書く。エージェントが考えている間（数十秒）に Pi を走らせ続けないため、move は 1 回 30 cm・数秒で終わる
大きさに限ってある。

### move の止め方（Pi 側）

agent の move は explore の 1 歩と同じ止め方をすべて当て、さらに次で止める。止めた理由を `stop_reason` に書く。

- 走った長さが `distance_cm` に届いた（`done`）
- 前へ: 超音波が `explore_step_stop_cm`（30 cm）より近い（`ultrasonic`）・`clear_cm` が `explore_clear_stop_cm`
  （25 cm）より近い（`clear`）・null が 0.3 s 続いた（`clear_unknown`）
- 画像が動かない（`stuck`）
- guard が OK でない（`guard`）・snoo が近い（`snoo_near`）
- 電池が低い（`battery`）
- heartbeat が切れた（`heartbeat`）・stop が来た（`stop`）・USR を押した（`button`）
- 時間切れ: `distance_cm / (speed × 1.7 cm/s) × 1.5 + 1` 秒を過ぎた（`timeout`）

速さは `min(speed, 10)`、ハンドルは ±30° に収める。後ろへは、Pi の後ろにセンサーが無いので、
画像が動かないときの止め方（下がるときの短いほう）を使う。

## Pi 側でいつも当てる安全

サーバーの確かめ（下の「サーバーが move を確かめること」）は、並べたときの地図と画像でしか見ていない。
走っている間の安全は、いつも Pi 側が受け持つ。agent の指令は、どれもこの層を通らないと車体を動かさない。

- **agent のモード**: 指令を実行するのは `mode == "agent"` のときだけ。ほかのモード（explore など）では
  問い合わせは続けるが、受けた指令には `refused`（`not_agent_mode`）を書く。agent に入るのは人が決めたときだけ
  （USR を 5 回押す・CLI の `--mode agent` など。Pi 側で決める）。agent では Pi は自分では走らない
- **guard**（`snoo_trucar/guard.py`）: agent の move にも最後にかぶせる。snoo に自分から 30 cm より近づかない、
  ストレスで止まって下がる、遊びの時間の上限・休み、をそのまま当てる
- **電池**: `battery_low_v`（6.6 V）より低ければ走らない（`refused`・`battery_low`）。首は動かしてよい。
  6.4 V より低いのが 10 s 続いたら止めて shutdown する、は今まで通り
- **USR の長押し**（1.5 s）: すぐ止めて idle にし、そのとき受けている・並んでいる agent の指令をすべて
  `cancelled`（reason `button`）にする。長押しの後に問い合わせで受けた指令のうち、長押しより前に並んだもの
  （`t` が長押しのときのサーバーの `now` より前）も実行しない
- 速さ・長さの上限は Pi 側でももう一度収める（サーバーを信じきらない）

## サーバーが move を確かめること

`POST command` の move は、次のどれかに当たれば並べない（`queued: false`、`checks` に理由）。

| check | 内容 |
| --- | --- |
| `distance`・`speed`・`steer` | 0 < 長さ ≤ 30 cm、0 < 速さ ≤ 10、ハンドル ±30° |
| `one_move_at_a_time` | 前の move に結果（done など）が無い |
| `pi_mode` | Pi が問い合わせで idle・agent 以外のモードを書いてきた |
| `fresh_frame` | 今のフレームが 3 秒より古い（猫がいないと言えない） |
| `robot_still` | 今のフレームの指令が走っている |
| `cat_far` | `cat ≥ 0.35` の箱で、距離が 150 cm より近い・距離が分からないものがある（snoo でなくても） |
| `grid` | セッションの間取り（grid.png・room.json、セッションの座標）が無い |
| `pose` | 今の位置が分からない |
| `clip_pieces` | 前へ: 自転車モデルで走らせ、車の真ん中かカメラの真下が、車の大きさ分広げた障害物・見ていない所に入る手前で縮めると、頼んだ長さに届かない（`app/roommap.clip_pieces`） |
| `moves_are_clear` | 車の真ん中が障害物・見ていない所に入る。後ろへの頭 15 cm だけは見ていない所も通す（`app/roommap.moves_are_clear`） |

間取りは地図のサーバーが最後に書いた grid.png（次の 1 歩を決めた格子。ステーションで見た所・走った所・SLAM の
分を足した後）。位置は SLAM が新しければその位置、無ければ最後のステーションの位置。look・scan・stop は
範囲だけを確かめる。

## 声（スピーカー）

`say` は pyopenjtalk で読み上げ、16 kHz・モノラル・16 ビットの wav にする。同じ文は
`data/snoo_agent/tts/<sha256 の頭 16 桁>.wav` を使い回す。pyopenjtalk は初めて使うときに辞書（23 MB）を
GitHub から取る。地図のサーバーの venv（main の `.venv`。uv.lock に無い VGGT が入っているので `uv sync`
しない）には `uv pip install pyopenjtalk` で足す（`pyproject.toml` の `agent` の組にある）。
無ければ `say` は 503 を返す。

### Pi で鳴らす

1. 1〜2 秒ごとに `GET /v1/snoo/agent/say/next?since=<最後に鳴らした id>` を問い合わせる（`commands/next`
   と同じループでよい）。`say` が null なら何もしない
2. `say.url`（`/v1/snoo/agent/say/<hash>.wav`）を取り、Pi の tmpfs（`/run/user/...` か `/dev/shm`）に置く
3. スピーカーを有効にしてから `aplay -q <ファイル>` で鳴らす。鳴らし終えたら `since` をその id にする
4. 期限（`expires_at`、2 分）はサーバーが見る。古い知らせは渡らない

```json
{"say": {"id": 4, "t": 1793351410.2, "text": "ソファの下を見に行くよ", "wav": "1f0c...e2.wav",
         "url": "/v1/snoo/agent/say/1f0c...e2.wav", "expires_at": 1793351530.2}, "now": 1793351411.0}
```

- Robot HAT のスピーカーのアンプは **GPIO 20** で有効にする（High で鳴る）。`robot_hat` の
  `utils.enable_speaker()`、または `pinctrl set 20 op dh`（古い OS は `raspi-gpio set 20 op dh`）。
  鳴らさない間は Low に戻すと、雑音と電池の減りが少ない
- snoo を驚かせないよう、音量は小さめにし、snoo が近い（guard が OK でない）ときは鳴らさない
- 鳴らしている間も走る・止まるの安全は止めない（aplay は別のプロセスで）

### SunFounder のスピーカーの準備（人が手で 1 回）

Robot HAT のスピーカーは I2S で鳴らす。SunFounder の robot-hat のリポジトリにある `i2samp.sh` で、
I2S のオーバーレイと ALSA の設定を入れる。**sudo が要り、最後に再起動する**ので、ユーザーが手で行う
（エージェントや Pi のプログラムからは行わない）。

```sh
cd ~ && git clone https://github.com/sunfounder/robot-hat.git   # 既にあれば git pull
cd robot-hat
sudo bash i2samp.sh        # 聞かれたら y。最後に再起動する
# 再起動の後
aplay -l                   # sndrpihifiberry などの I2S のカードが出る
```

## 第 1 段でできること・できないこと

- できる: 今の様子・カメラ・地図を見る、印を付ける、困りごとの下書きを書く、声を作って並べる、指令を確かめて並べる
- できない（Pi 側を作るまで）: 指令の実行、声を鳴らすこと。`pi_support` は false のまま

## Pi 側で作ること（第 2 段）

1. `agent` のモードと、入り方（USR 5 回・CLI）。LED の点滅
2. `commands/next` の 0.5 秒ごとの問い合わせ（全モード。様子の欄を付ける）と heartbeat（3 秒）
3. look・move・scan・stop の実行と、結果（accepted → running → done など）の書き込み。move は上の止め方、
   guard・電池・USR の長押しを必ず通す
4. scan: agent のセッション（`agent_<日時>`）でステーションを送る。返ってきた 1 歩は走らない。
   `prev_step` には前のステーションからの agent の move を書く（同じ向きの move だけなら `segments` に並べ、
   向きが混じったら `{"stop_reason": "agent"}` だけにする。地図のサーバーは長さの分からない歩を地図に足さない）
5. `say/next` を取って `aplay` で鳴らす（GPIO 20）。i2samp.sh はユーザーが先に入れておく
6. docs/PROTOCOL.md にこの文書への参照を足す
