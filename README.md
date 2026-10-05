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

## 設定（環境変数）

| 変数 | 既定値 | 説明 |
|---|---|---|
| `USE_OPENCL` | `true` | `false` で常に CPU 処理 |
| `MAX_UPLOAD_BYTES` | `20971520` | アップロード上限（20MB） |
| `API_TOKEN` | なし | `/v1/shopify/...` の認証（`X-Internal-Token`）。空なら使えない |
| `SHOPIFY_STORE` / `SHOPIFY_CLIENT_ID` / `SHOPIFY_CLIENT_SECRET` | なし | Shopify Admin API（Client Credentials） |
| `FIT_MAX_SIDE` | `2048` | 商品画像の長辺の上限 |
| `FIT_QUALITY` | `90` | 縮めた画像の JPEG の品質 |

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
  config.py      # 環境変数
tests/           # API テスト（CPU で動く）
scripts/bench.py # CPU / OpenCL 比較
scripts/shopify_fit_images.py # 商品画像を手動で一括で縮める
```

新しい処理を足すときは `processing.py` に `_to_device` → OpenCV 関数 → `_to_host` の形で関数を書き、`main.py` にエンドポイントを追加する。
