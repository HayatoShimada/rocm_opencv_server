# rocm_opencv_server

AMD GPU（ROCm）で OpenCV の画像処理を実行する HTTP API サーバー。

OpenCV の CUDA モジュールは AMD GPU では使えないため、**OpenCL（T-API / `cv2.UMat`）経由で ROCm の OpenCL ランタイムに処理を流す**構成にしている。OpenCL が使えない環境では同じコードが CPU で動く。

- FastAPI + `opencv-python-headless`
- ベースイメージ: `rocm/dev-ubuntu-24.04:10.0.0-full`（ROCm 10.0.0 の SDK 一式。HIP / PyTorch への拡張を見込んでいる）
- 動作確認: Radeon RX 7900 XTX（gfx1100）/ ROCm 10.0.0 / ホストカーネル 7.0

## 起動（Docker・GPU あり）

```bash
docker compose up -d --build
curl localhost:8000/device   # opencl_enabled: true と GPU 名が出れば OK
```

コンテナに `/dev/kfd` と `/dev/dri` を渡し、`video` / `render` グループの GID を付けている（`compose.yaml`）。ホスト側のユーザーもこの2グループに入っている必要がある。

ポートはホストの `127.0.0.1:8000` だけで受ける（画像処理のエンドポイントは認証なしのため、LAN には出さない）。tailnet の別マシン（85pi の 85store-cms など）から使うときは、Tailscale で tailnet にだけ転送する。

```bash
tailscale serve --bg --tcp 8000 tcp://127.0.0.1:8000
# 呼び出し側: http://home-linux.taila713c8.ts.net:8000
```

GID の既定値は `video=44` / `render=992`。ホストで `getent group video render` の値が違う場合は `.env` で上書きする。

```bash
echo "RENDER_GID=$(getent group render | cut -d: -f3)" > .env
```

### ROCm 10 での注意点

ROCm 7.x から構成が変わっているため、`Dockerfile` で以下を吸収している。

- `-full` イメージはダウンロード約 8GB、展開後約 29GB ある。ROCm 10 には軽量版のタグが無い。OpenCL だけなら `ubuntu:24.04` に `amdrocm-opencl10.0` を入れる構成で約 2.4GB まで減らせる（パッケージ名は `rocm-opencl-runtime` から変わった）
- ROCm の実体は `/opt/rocm/core-10.0/` にあり、OpenCL の ICD が `/etc/OpenCL/vendors` に登録されない。そのままだと `clGetPlatformIDs(-1001)` になるため、ICD ファイルを自前で置き、`OPENCV_OPENCL_RUNTIME` で ROCm 付属の `libOpenCL.so.1` を指定している
- ROCm 10 のイメージには `render` グループが無いので、`group_add` は名前ではなく GID で指定する

ROCm のバージョンはビルド引数 `ROCM_IMAGE_TAG`（`compose.yaml`）で切り替えられる。コンテナ内では `clinfo` / `amd-smi` / `hipcc` が使える。

## ローカル開発（CPU）

```bash
uv sync
uv run uvicorn app.main:app --reload
uv run pytest
uv run ruff check . && uv run ruff format .
```

ホストで GPU を使いたい場合は、AMD の apt リポジトリを登録して `sudo apt install amdrocm-opencl10.0` を入れ、上記と同じく ICD の登録と `OPENCV_OPENCL_RUNTIME` の設定を行う（[インストール手順](https://rocm.docs.amd.com/en/latest/install/rocm.html)）。

## API

| メソッド | パス | パラメータ（クエリ） |
|---|---|---|
| GET | `/health` | |
| GET | `/device` | OpenCL / GPU の情報 |
| POST | `/v1/grayscale` | |
| POST | `/v1/resize` | `width`, `height`（必須） |
| POST | `/v1/fit` | `max_side`（必須）, `quality`（既定 90）。縦横比を保って長辺を収める（拡大しない）。既定の出力は `jpeg` |
| POST | `/v1/blur` | `ksize`（奇数・既定 5）, `sigma`（既定 0） |
| POST | `/v1/canny` | `threshold1`（既定 100）, `threshold2`（既定 200） |

POST は `multipart/form-data` の `file` で画像を受け取り、画像を返す。出力形式は `format=png|jpeg|webp`（既定 `png`）。Swagger UI は `http://localhost:8000/docs`。

```bash
curl -F file=@input.jpg "localhost:8000/v1/canny?threshold1=50" -o edges.png
```

## Shopify の商品画像を縮める

長辺が `FIT_MAX_SIDE`（既定 2048）を超える商品画像だけを縮め、Shopify で差し替える（`app/shopify.py`）。差し替えには `fileUpdate` の `originalSource` を使うので、画像の ID・並び順・alt は変わらない。透過のある PNG は PNG のまま、それ以外は JPEG（`FIT_QUALITY`）にする。

- 色: 書き出す画像に ICC は埋め込まないので、sRGB 以外の RGB（Display P3 など）は sRGB に変換する（判定は原色の色度で行う）。2026-10 の時点では、ストアの画像はすべて sRGB だった。
- 速さ: 時間のほとんどは Shopify とのやりとりと、差し替えた画像の処理待ち（1枚 約4秒）。画像処理は 1枚 0.1秒ほどで、縮小は CPU のほうが速い（GPU との転送が重い）。そのため画像を `--workers` 枚（既定 6）ずつ並列に処理する。
- alt: `--fill-alt`（API では `fillAlt`）で、空の alt を「商品名（n枚目）」で埋める（85store-cms の保存時と同じ形）。

`.env` に Shopify の認証情報を入れる（`.env.example`）。85store-cms と同じアプリを使う。

**手動で一括実行する**（`--apply` を付けないときは、対象を表示するだけ）:

```bash
docker compose run --rm server python -m scripts.shopify_fit_images            # 全商品を確認
docker compose run --rm server python -m scripts.shopify_fit_images --product <ハンドル> --apply
docker compose run --rm server python -m scripts.shopify_fit_images --fill-alt --apply   # 全商品
```

**CMS から呼ぶ**: `POST /v1/shopify/products/fit-images`（JSON `{"productId": "gid://shopify/Product/...", "maxSide": 2048, "fillAlt": true}`、ヘッダ `X-Internal-Token: <API_TOKEN>`）。85store-cms が商品の同期・取り込みのあとに呼ぶ。結果は `{checked, resized, alt_filled, errors}`。

## 単品の写真のホワイトバランスを揃える

単品の商品写真（白い壁にハンガーで掛けたもの）の色かぶりと明るさを、基準の画像の壁の色に揃える（`app/whitebalance.py`）。

- 基準: `amazon-hooded-sweat-parkaused` の1枚目（`0E2A5894.jpg`）の右上の壁（幅・高さの x 75.1〜83.9%・y 4.7〜10.1%）の平均 RGB(205.1, 206.7, 206.9)。それぞれの写真の同じ範囲を測り、リニア RGB でチャンネルごとのゲインを掛ける。範囲に服やハンガーが入っていたら、ほかの候補の位置を使う。
- ゲインが 0.7〜1.4 を超えるものと、壁が見つからないものは「要確認」にして補正しない。目標との色差（ΔE）が 2 未満なら「補正不要」。
- 写真の種類（単品・着用・ディテール・その他）は CLIP（`app/classify.py`、ViT-L/14）で判定し、単品だけを補正する。横長の画像と、確からしさが 0.6 未満のものは補正しない。判定は `data/classify-cache.json` に残す（モデル・説明文・画像が変わったら判定し直す）。
- CLIP は重いので依存グループ `clip` に分けてある。PyTorch は AMD 配布の ROCm 10.0 版（`torch 2.13.0+rocm10.0.0`、gfx1100）。

```bash
uv sync --group clip
uv run --env-file .env --group clip python -m scripts.shopify_white_balance            # 確認だけ（data/wb-report/index.html）
uv run --env-file .env --group clip python -m scripts.shopify_white_balance --product <ハンドル> --apply
```

`--apply` では、差し替える前に元の画像を `~/85store-shopify-originals/<日付>/` に保存する。

## 商品写真を「着用・全体・アップ」に分ける（Clef）

Cloudflare の Clef（判定用のモデル）に、写真ごとに質問をする。既定は、オープンウェイトの Clef-flash（9B）を手元の GPU で動かす（`CLEF_BACKEND=workers-ai` で Workers AI の `@cf/cloudflare/clef-flash`）。どちらも同じ判定になる（32枚で区分・向き・部分が一致、確率の差は 0.012 以内）。
- 人が服を着ている写真か（手や指だけが写っているものは含めない）
- 商品の全体が1枚に収まっているか

人が写っていれば「着用」、いなければ「全体」か「アップ」に分ける。画像は Shopify の CDN で幅 512px に縮めて渡す（1枚あたり約450トークン）。

手元の GPU で動かすには、重み（約19GB）を一度ダウンロードしておく。

```bash
uvx --from 'huggingface_hub[hf_xet]' hf download Cloudflare/clef-flash --local-dir ~/models/clef-flash
```

| | 手元の GPU（既定） | Workers AI |
|---|---|---|
| 速さ（RX 7900 XTX） | 2.5〜2.8 枚/秒（1555枚で約9分） | 12 枚/秒（6並列。約2分） |
| 読み込み | 約9秒（VRAM 約19〜21GB） | なし |
| 費用 | 電気代 | 1枚 約0.013円（入力 約1000トークン） |

```bash
# 全商品の画像を分類する（data/photo-labels/predictions.json。差し替えていない画像は取り直さない）
uv run --env-file .env --group clip --group clef-local python -m scripts.shopify_classify_photos
# 分類が正しいかを確かめる画面（http://127.0.0.1:8010）
uv run python -m scripts.photo_review
```

確かめる画面の使い方:
- 写真ごとに、正しい区分（着用・全体・アップ）を選ぶ。写真にカーソルを置いて 1・2・3 のキーでも選べる。
- 判定が合っていれば、下の「判定どおり確認済みにする」（Enter）で、そのページをまとめて確認済みにする。
- 初めは「迷っている順」（確率が 0.5 に近い順）に並ぶ。
- 上の欄に、確かめた写真での判定の正しさが出る。
- 確かめた結果は `data/photo-labels/labels.json` に保存する。

### 写真の区分を alt に入れる

Clef には、向き（正面・背面・横）と、アップで写している部分（タグ・ロゴやプリント・生地・ボタンやジッパー・襟元・袖口や裾・ポケット・傷や汚れ）も聞いている。それを使って、自動の alt を「商品名（正面）」「商品名（タグのアップ）」の形にする。
- 確からしさが低い向き・部分は使わず、「全体」「着用」「ディテール」とだけ書く。
- 区分の語は 85store-cms の `AUTO_ALT_LABELS` と同じ（`app/shopify.py` の `ALT_LABELS`）。CMS は、この形の alt を自動の alt として扱い、商品名を変えたら付け直す。

```bash
uv run --env-file .env python -m scripts.shopify_photo_alts          # 書き換える数と例を出す
uv run --env-file .env python -m scripts.shopify_photo_alts --apply  # Shopify の alt を書き換える
```

- 書き換えるのは、alt が空か、自動の alt（n枚目・区分）の画像だけ。
- 人が入れた alt と、ほかの商品と共有している画像には触らない。
- `--apply` では、前の alt を `~/85store-shopify-originals/alts-<日時>.json` に保存する。

## 商品の属性を判定する（Clef・AI 検索の下ごしらえ）

ショップの AI 検索（85store の `cloudflare/product-search`）が使う、商品の属性を Clef で判定する（`app/product_attributes.py`）。商品名・説明・ブランド・素材・実寸と、写真の分類から選んだ「全体・正面」「着用」の2枚を渡す。

- 聞くこと: 種類・色・柄・テイスト（ワーク・ミリタリー・アウトドア・スポーツ・ストリート・トラッド・ウエスタン・きれいめ・シンプル）・季節・厚さ・シルエット・対象・系統（アメリカ・ヨーロッパ・日本）
- 色・柄と種類は、入力済みの値と比べて正しさを測るため Clef に渡さない。販売中の293点（2026-10）で、種類 95%・色 85%・柄 86% が入力済みの値と一致した。色の外れには、入力済みの値のほうが違うもの（ブラックで登録されたベージュのキャップなど）が含まれる
- 1点 約2.8秒（手元の GPU）。Shopify には書き込まない

```bash
# 判定する（data/product-attributes/predictions.json と、確かめるための index.html）
uv run --env-file .env --group clip --group clef-local python -m scripts.shopify_product_attributes --sample 30
uv run --env-file .env --group clip --group clef-local python -m scripts.shopify_product_attributes --all
# AI 検索の Worker 用に書き出す（85store の cloudflare/product-search の README を参照）
uv run --env-file .env python -m scripts.shopify_product_attributes --all --export data/product-attributes/search-attributes.json
```

## 設定（環境変数）

| 変数 | 既定値 | 説明 |
|---|---|---|
| `USE_OPENCL` | `true` | `false` で常に CPU 処理 |
| `MAX_UPLOAD_BYTES` | `20971520` | アップロード上限（20MB） |
| `API_TOKEN` | なし | `/v1/shopify/...` の認証（`X-Internal-Token`）。空なら使えない |
| `SHOPIFY_STORE` / `SHOPIFY_CLIENT_ID` / `SHOPIFY_CLIENT_SECRET` | なし | Shopify Admin API（Client Credentials） |
| `FIT_MAX_SIDE` | `2048` | 商品画像の長辺の上限 |
| `FIT_QUALITY` | `90` | 縮めた画像の JPEG の品質 |
| `CLEF_BACKEND` | `local` | 写真の分類を動かす場所（`local` = 手元の GPU、`workers-ai`） |
| `CLEF_MODEL_PATH` | `~/models/clef-flash` | 手元で動かすときの Clef-flash の重み |
| `CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_API_TOKEN` | なし | Workers AI（`CLEF_BACKEND=workers-ai`）。トークンは「Workers AI」の権限で作る |
| `CLEF_MODEL` | `@cf/cloudflare/clef-flash` | Workers AI で使うモデル（`@cf/cloudflare/clef` で 27B） |

## ベンチマーク

```bash
docker compose run --rm server python scripts/bench.py --size 3840x2160
```

4K 画像に対して ブラー → グレースケール → Canny を実行し、CPU と OpenCL の時間を比べる。初回呼び出しは OpenCL カーネルのコンパイルが入るので遅い（スクリプトではウォームアップで除外している）。

## 構成

```
app/
  main.py        # FastAPI のエンドポイント
  processing.py  # 画像処理（numpy ⇄ UMat の変換を含む）
  gpu.py         # OpenCL の初期化とデバイス情報
  shopify.py     # Shopify の商品画像を縮めて差し替える
  whitebalance.py # 単品の写真のホワイトバランスを基準の壁の色に揃える
  classify.py    # 写真の種類を CLIP で判定する
  clef.py        # 写真の区分を Clef（手元の GPU か Workers AI）で判定する
  product_attributes.py # 商品の属性（AI 検索用）を Clef で判定する質問
  config.py      # 環境変数
tests/           # API テスト（CPU で動く）
scripts/bench.py # CPU / OpenCL 比較
scripts/shopify_fit_images.py # 商品画像を手動で一括で縮める
scripts/shopify_white_balance.py # 単品の写真のホワイトバランスを揃える
scripts/shopify_classify_photos.py # 全商品の画像を Clef で分類する
scripts/photo_review.py # 分類が正しいかを確かめる画面
scripts/shopify_photo_alts.py # 写真の区分を付けた alt にする
scripts/shopify_product_attributes.py # 商品の属性を判定し、AI 検索用に書き出す
```

新しい処理を足すときは `processing.py` に `_to_device` → OpenCV 関数 → `_to_host` の形で関数を書き、`main.py` にエンドポイントを追加する。

## PiCar-X の猫の検出（UDP）

85pi4（PiCar-X、snoo_trucar）から UDP で届くカメラ画像から猫を検出し、箱を返す（`app/snoo.py`）。やりとりの仕様は snoo_trucar の `docs/PROTOCOL.md`（20 バイトのヘッダー + JPEG を受け、JSON を返す）。

- 検出: RT-DETR v2（`PekingU/rtdetr_v2_r18vd`、COCO の cat、Apache-2.0）を GPU で fp16 で動かす（`--compile` で torch.compile）。60 fps で受けてから返すまで 8 ms ほど、85pi4 からの往復で 15 ms ほど
- snoo らしさ: 猫の箱を CLIP で埋め込み、snoo の写真から作った見本（`scripts/snoo_gallery.py`）との近さを返す。見本が無ければ `null`
- 家の中では、Pi は tailscale を通さず LAN の IP に送る（tailscale 経由より往復が約 3 ms 短い）。`0.0.0.0` で受け、認証の無い UDP なので `--allow`（`SNOO_ALLOW`）に書いた送り元（85pi4 の LAN と tailscale の IP）以外は捨てる。tailscale serve は UDP を通せないので、tailscale からも直接受ける
- 10 秒ごとのログに、区間（受信・デコード・検出・CLIP・送信）ごとの p50/p95 を出す
- compose では `snoo` サービスとして常に動かす（`docker compose up -d snoo`）
- GPU の時計: 検出はフレームの合間に GPU が時計を下げるため、60 fps の間隔だと 7.8 ms かかる（休まず回すと 4.3 ms）。上げたままだと消費電力が増えるので、ホストの `deploy/snoo-gpu-perf.sh` が、検出がフレームを処理している間（`/run/snoo/active` が 3 秒以内に触られている間）だけ `high` にし、それ以外は `auto` に戻す。スクリプトが止まったときは trap と systemd の `ExecStopPost` で、起動し直したときは amdgpu の既定で `auto` になる

  ```bash
  sudo install -m 755 deploy/snoo-gpu-perf.sh /usr/local/bin/
  sudo install -m 644 deploy/snoo-gpu-perf.service /etc/systemd/system/
  sudo systemctl daemon-reload && sudo systemctl enable --now snoo-gpu-perf
  journalctl -u snoo-gpu-perf -f   # 「GPU の時計: high」「auto」が出る
  ```

```bash
uv sync --group clip --group clef-local   # torch（ROCm）と transformers
uv run python -m app.snoo --host 0.0.0.0 --allow 192.168.0.188,100.73.14.8   # 既定のポートは 8710
```

## 開発の流れ（snoo のサーバー）

main の置き場（`~/orca/rocm_opencv_server`）で `make` を使う（`make help` で一覧）。待つものは、いまどの準備をしているかと経った秒数を出し、上限の秒数を超えたらやめる。

| やること | コマンド | かかる時間（2026-10-29） |
|---|---|---|
| 重いテストを飛ばして試す | `make test-fast` | 約 20 秒（200 個） |
| すべてのテスト | `make test` | 約 50 秒（6 個の pytest に分けて同時に。1 つだと約 4 分） |
| ruff | `make lint`（直すときは `make fmt`） | 1 秒 |
| 検出（app/snoo.py など）を直して試す | `make deploy-detector` | 約 15〜20 秒（待ち受けまで）＋ p50/p95 が出るまで最大 10 秒 |
| 地図（app/snoo_map_server.py など）を直して試す | `make deploy-map` | 約 7 秒 |
| 両方の様子を見る | `make status` | 0.1 秒 |
| 保存したセッションの最後の 1 歩を GPU なしで決め直す | `make replay SESSION=explore_20261029_223944` | 1 秒 |

- **検出**: `compose.dev.yaml` が `./app` をコンテナに読み取り専用で重ねるので、app/ を直したら作り直さずに起動し直すだけで新しいコードで動く。`make deploy-detector` は、上書きの無いコンテナなら一度だけ作り直し（`docker compose -f compose.yaml -f compose.dev.yaml up -d snoo`）、あとは `restart` して「待ち受け」まで待ち、最初の p50/p95 の行を出す。起動に失敗したら Traceback を出してやめる。torch.compile の準備は torch-compile ボリュームのキャッシュが効く（検出 約 11 秒・clear_cm の深度 約 2.5 秒）。もっと速く試すときは `SNOO_CLEARANCE=0`（約 16 秒）・`SNOO_COMPILE=0`（clear も切って約 9 秒）を付けて `make deploy-detector`（作り直しになる）。pyproject.toml・uv.lock・Dockerfile を変えたときは `docker compose build snoo` が要る（`make deploy-detector` が知らせる）。本番の形（上書きなし）に戻すのは `docker compose up -d snoo`、その後に待つだけなら `.venv/bin/python -m scripts.snoo_dev wait-detector`
- **地図**: systemd のユーザーのサービス（`deploy/snoo-map.service`）で動かす。落ちたら 3 秒後に起動し直し、ホストを起動し直しても動く。入れ方（sudo は要らない。手で起動したものを先に止める）:

  ```bash
  pkill -f 'python -m app.snoo_map_server'          # 手で起動したもの（8720）を止める
  mkdir -p ~/.config/systemd/user
  ln -sf ~/orca/rocm_opencv_server/deploy/snoo-map.service ~/.config/systemd/user/
  systemctl --user daemon-reload && systemctl --user enable --now snoo-map
  loginctl show-user "$USER" -p Linger              # Linger=yes なら、ログインしていなくても動く
  journalctl --user -u snoo-map -f                  # ログ
  ```

  送り元（`SNOO_MAP_ALLOW`）などを変えるときは `~/.config/snoo-map.env` に書いて `make deploy-map`
- **走っている間の SLAM**: 検出が走っている間のフレームを 4 枚/秒ほど `/run/snoo/slam` に書き（`SNOO_SLAM_TAP_DIR`、app/snoo_slam_tap.py）、ホストの app/snoo_slam.py（VGGT-SLAM 2.0）がステーションと合わせて読み、`<セッション>/slam/` に trajectory.jsonl・points.npz・status.json を書く（形は app/snoo_slam.py の先頭）。torch・gtsam は別の venv にあるので、`deploy/snoo-slam-venv.sh` で作ってから `deploy/snoo-slam.service` を地図と同じやり方で入れる（`systemctl --user enable --now snoo-slam`）。保存したセッションを流し直して poses.json と比べるのは `HSA_CU_MASK=0:0-71 ~/orca/vggt-slam-venv/bin/python -m scripts.snoo_slam_replay <セッションの置き場> --out /tmp/slam`
- **テスト**: 重いテストの印（slow）は `tests/conftest.py` の `SLOW` に名前で書く（テストのファイルは直さない）。`make test-fast` で、印が無いのに 3 秒を超えたテストは最後に名前が出る。`make test ARGS='-x'` のように pytest に引数を渡せる
- **replay**: 地図のサーバーはステーションごとに VGGT の出力をセッションの `vggt.npz` に上書きするので、決め直せるのはそれを作ったステーション（ふつうは最後）の 1 歩。VGGT の代わりに `vggt.npz` を読み、サーバーと同じ `plan_station` を通す。もとのセッションには書かず、`/tmp/snoo-replay/<セッション>` に room.json・topdown.png・plan.json を作る。前に Pi に返したものと違えば終了コード 1（`ARGS=--allow-diff` で 0）

## 実機で探索する日の手順（snoo）

0. **Pi 側の安全の直しを入れる**（まだなら）：snoo_trucar#21 の `review/safety-fixes`
   （bundle は `~/orca/backups/snoo_trucar_safety_fixes.bundle`）。入れたら、走らせている間に
   `systemctl --user stop snoo-trucar` で車が止まることを一度確かめる
1. **実機のターンにする**：学習していれば `make phase-robot`（学習を止めてから見張りを起こす）。学習に戻るときは `make phase-train`、今どちらかは `make phase`
   **サーバーを起こす**：`systemctl --user start snoo-ondemand`（Pi が見えたら、検出・地図・SLAM を起こす）。
   すぐ全部起こすなら `make deploy-detector && make deploy-map && systemctl --user start snoo-slam`。
   `make status` で、検出の p50/p95 と「今のフレーム」が新しいことを見る
1.5 **縮尺を確かめる（一度だけ）**：白い A4 を車の 30〜60 cm 前の床に、縁が全部写るように置き、チルト -15° で 1 枚撮る。
   `make scale-check IMAGE=<その画像> TILT=-15` でカメラの高さが出る（今は SD カードで測った 8 cm）。
   5 % 以上違えば、`SNOO_CAMERA_HEIGHT_CM`（compose）と Pi の `camera_height_cm` を直す。
   長い辺と短い辺で合わなければ、チルトの較正がずれている
2. **部屋の明かりをつける**（暗いと VGGT の手がかりが取れず、誤検出も増える）。snoo が近くにいないことを確かめる
   （後ろにはセンサーが無く、下がるときに猫に当たりうる）
3. **電池**：満充電（8.4 V）にしてスイッチを入れる。6.8 V を切ると地図のサーバーが low_battery で止める
4. **探索を始める**：USR を 4 回。ブラウザで `http://127.0.0.1:8720/twin` を開いて様子を見る
5. **終わったら**：`make run-report SESSION=<explore_...> BASE=explore_20261029_223944`。Pi の clear.jsonl を
   取ってきて `CLEAR=...` も付けると、clear_cm と超音波の差も出る。見るところ：
   - 止まったコマのくっきりさ（BASE の 0.7 倍を切れば、撮る前の待ちが短すぎる）と、向きのずれ（4° 超え）
   - 止まった理由と返事の内訳（random・turn・reverse・done の理由）、散らばりの広さ
   - clear_cm の値が出た割合（settling・out_of_view が多すぎないか）
6. **決め直す**：直したら `make replay SESSION=<explore_...> STATION=<N>` で、そのステーションの 1 歩を GPU なしで確かめる
7. **ディスク**：古いセッションの重いファイル（recon.npz・vggt.npz・SLAM のフレーム）は、地図のサーバーを
   起こしたときに新しい 5 セッションを残して消える。手でするなら `make prune`（`ARGS=--dry-run` で数だけ）
8. **片付け**：`systemctl --user stop snoo-ondemand snoo-slam snoo-map` と `docker compose -f compose.yaml -f compose.dev.yaml stop snoo`
   （GPU を握ったプロセスが 2 つ以上あると、何もしていなくても 120 W 食う）
