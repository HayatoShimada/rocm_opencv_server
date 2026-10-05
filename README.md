# rocm_opencv_server

AMD GPU（ROCm）で OpenCV の画像処理を実行する HTTP API サーバー。

OpenCV の CUDA モジュールは AMD GPU では使えないため、**OpenCL（T-API / `cv2.UMat`）経由で ROCm の OpenCL ランタイムに処理を流す**構成にしている。OpenCL が使えない環境では同じコードが CPU で動く。

- FastAPI + `opencv-python-headless`
- ROCm 10.0（`ubuntu:24.04` に `amdrocm-opencl10.0` だけを入れる）
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

- 公式イメージ `rocm/dev-ubuntu-24.04:10.0.0-full` は約 8GB あり、軽量版のタグが無い。OpenCL しか使わないので `ubuntu:24.04` に AMD の apt リポジトリを足し、`amdrocm-opencl10.0` だけを入れている（パッケージ名が `rocm-opencl-runtime` から変わった）
- ROCm は `/opt/rocm/core-10.0/` に入り、OpenCL の ICD が `/etc/OpenCL/vendors` に登録されない。そのままだと `clGetPlatformIDs(-1001)` になるため、ICD ファイルを自前で置き、`OPENCV_OPENCL_RUNTIME` で ROCm 付属の `libOpenCL.so.1` を指定している
- ROCm 10 のイメージには `render` グループが無いので、`group_add` は名前ではなく GID で指定する

ROCm のバージョンはビルド引数 `ROCM_VERSION`（`compose.yaml`）で切り替えられる。

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
| POST | `/v1/blur` | `ksize`（奇数・既定 5）, `sigma`（既定 0） |
| POST | `/v1/canny` | `threshold1`（既定 100）, `threshold2`（既定 200） |

POST は `multipart/form-data` の `file` で画像を受け取り、画像を返す。出力形式は `format=png|jpeg|webp`（既定 `png`）。Swagger UI は `http://localhost:8000/docs`。

```bash
curl -F file=@input.jpg "localhost:8000/v1/canny?threshold1=50" -o edges.png
```

## 設定（環境変数）

| 変数 | 既定値 | 説明 |
|---|---|---|
| `USE_OPENCL` | `true` | `false` で常に CPU 処理 |
| `MAX_UPLOAD_BYTES` | `20971520` | アップロード上限（20MB） |

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
  config.py      # 環境変数
tests/           # API テスト（CPU で動く）
scripts/bench.py # CPU / OpenCL 比較
```

新しい処理を足すときは `processing.py` に `_to_device` → OpenCV 関数 → `_to_host` の形で関数を書き、`main.py` にエンドポイントを追加する。
