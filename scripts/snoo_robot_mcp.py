"""PiCar-X（snoo_trucar）を AI のエージェント（Claude Code など）から使う MCP のサーバー（stdio）。

地図のサーバーのエージェントの API（app/snoo_agent_api.py、/v1/snoo/agent/）を道具にして渡す。
エージェントは、今の様子・カメラの画像・上から見た地図を見て、部屋の印を付け、困りごとの下書きを
書き、声で知らせ、首・走る・撮る・止めるの指令を並べる。仕様は docs/agent_robot.md。

    python3 scripts/snoo_robot_mcp.py            # Claude Code が stdio で起動する（.mcp.json）
    SNOO_AGENT_URL=http://127.0.0.1:8720 ...     # 地図のサーバーの場所（既定）

MCP の SDK は使わず、標準のものだけで JSON-RPC（1 行 1 件）を話す。どの Python でも（main の
.venv を作り直さなくても）動かせるように。道具の呼び出しと tools/list・initialize・ping だけ。
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = os.getenv("SNOO_AGENT_URL", "http://127.0.0.1:8720").rstrip("/")
PREFIX = "/v1/snoo/agent"
TIMEOUT_S = float(os.getenv("SNOO_AGENT_TIMEOUT_S", "30"))
PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "snoo-robot", "version": "0.1.0"}

SAFETY = (
    "安全の決まり: move は 1 回 30 cm まで・速さ 10 まで・ハンドル ±30° まで。"
    "サーバーが今の間取りでぶつからないか（clip_pieces・moves_are_clear）を確かめ、"
    "猫が 150 cm より近い・距離が分からない・"
    "今のフレームが 3 秒より古い・間取りが無い・前の move が終わっていないときは並べない。"
    "Pi 側でも超音波・clear_cm・guard（snoo を怖がらせない）・電池・USR の長押しで必ず止まる。"
    "指令は並べて 15 秒のうちに Pi が受けなければ捨てる。"
)
NOT_YET = "今は Pi がまだ指令を受けない（pi_support: false）。並べても何も動かない。"

TOOLS = [
    {
        "name": "state",
        "description": "ロボットの今の様子: Pi の mode・電池・超音波・clear_cm・首の向き・"
        "見えている猫（snoo）と距離・向き、フレームの古さ、セッション・ステーション・地図のサーバーの最後の"
        "返事、SLAM の位置、pi_support。分からない欄は null。",
        "inputSchema": {
            "type": "object",
            "properties": {"session": {"type": "string", "description": "省くと最新"}},
        },
    },
    {
        "name": "look",
        "description": "今のカメラの画像（640x480）と、その欄（猫の箱・距離・向き）を返す。"
        "首の向きは変えない（変えるのは command_look）。",
        "inputSchema": {
            "type": "object",
            "properties": {"sender": {"type": "string", "description": "Pi の IP。省くと最新"}},
        },
    },
    {
        "name": "map",
        "description": "セッションの間取りを上から見た絵と、画素と座標（m）の対応。"
        "白 = 床、黒 = 障害物、灰 = 見ていない、橙 = 境目、青い丸 = ステーション、赤 = 車、"
        "緑 = 次に撮る位置、紫 = 印（A 番号）。"
        "座標はセッションの座標（原点は最初のステーション、x は最初の車の向き、y は左）。",
        "inputSchema": {
            "type": "object",
            "properties": {"session": {"type": "string", "description": "省くと最新"}},
        },
    },
    {
        "name": "annotate",
        "description": "部屋の意味のある所（家具・ケーブル・snoo の好きな所・地図の誤りなど）に印を"
        '付ける。x・y はセッションの座標（m。map の対応で画素から出す）。image_ref に "live" を'
        "渡すと今のフレームを写して残す。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "label": {
                    "type": "string",
                    "description": "短い名前（sofa・cable・snoo_spot など）",
                },
                "note": {"type": "string"},
                "image_ref": {
                    "type": "string",
                    "description": "live・station:<s>/<id>/<file> など",
                },
                "session": {"type": "string"},
            },
            "required": ["x", "y", "label"],
        },
    },
    {
        "name": "annotations",
        "description": "付けた印の一覧（session で絞れる）。",
        "inputSchema": {"type": "object", "properties": {"session": {"type": "string"}}},
    },
    {
        "name": "report_issue",
        "description": "困りごと（検出の見落とし・地図の誤り・動きのおかしさ）の下書きを "
        "data/snoo_agent/issues/ に書く。GitHub には出さない（人が読んでから出す）。images は "
        '"live"・"map"・"snapshot:<名前>.jpg"・"station:<session>/<id>/<file>" の参照。',
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string", "description": "Markdown。何を見て、何が違うと思うか"},
                "images": {"type": "array", "items": {"type": "string"}},
                "session": {"type": "string"},
            },
            "required": ["title", "body"],
        },
    },
    {
        "name": "say",
        "description": "日本語の文を声にして（16 kHz の wav）、Pi が鳴らす順番待ちに入れる。"
        "Pi のスピーカーが使えるまでは鳴らない。snoo を驚かせないよう、短く・ときどきにする。",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "200 文字まで"}},
            "required": ["text"],
        },
    },
    {
        "name": "command_look",
        "description": "首を pan_deg（右が +、±35°）・tilt_deg（上が +、±35°）に向ける"
        "指令を並べる。" + NOT_YET,
        "inputSchema": {
            "type": "object",
            "properties": {"pan_deg": {"type": "number"}, "tilt_deg": {"type": "number"}},
        },
    },
    {
        "name": "command_move",
        "description": "少し走る指令を並べる。distance_cm（30 まで）・speed（10 まで）・steer_deg"
        "（右が +）・reverse。" + SAFETY + NOT_YET,
        "inputSchema": {
            "type": "object",
            "properties": {
                "distance_cm": {"type": "number"},
                "speed": {"type": "number"},
                "steer_deg": {"type": "number"},
                "reverse": {"type": "boolean"},
                "session": {"type": "string"},
                "note": {"type": "string", "description": "なぜ動くか（記録に残る）"},
            },
            "required": ["distance_cm", "speed"],
        },
    },
    {
        "name": "command_scan",
        "description": "その場で止まって首を振って撮り、地図のサーバーに送る（station）"
        "指令を並べる。走らない。" + NOT_YET,
        "inputSchema": {"type": "object", "properties": {"note": {"type": "string"}}},
    },
    {
        "name": "command_stop",
        "description": "すぐ止める指令を並べる（並んでいる前の指令より先に渡る）。"
        "いつでも使ってよい。" + NOT_YET,
        "inputSchema": {"type": "object", "properties": {"note": {"type": "string"}}},
    },
    {
        "name": "command_status",
        "description": "並べた指令の様子（queued・accepted・running・done・refused・"
        "failed・expired・cancelled）と Pi の結果。id を省くと最近の一覧。",
        "inputSchema": {"type": "object", "properties": {"id": {"type": "integer"}}},
    },
]


class HttpError(Exception):
    pass


def request(method: str, path: str, body: dict | None = None, query: dict | None = None):
    """(Content-Type, 中身の bytes)。HTTP のエラーは HttpError（サーバーの detail を付ける）。"""
    url = BASE + PREFIX + path
    q = {k: v for k, v in (query or {}).items() if v is not None}
    if q:
        url += "?" + urllib.parse.urlencode(q)
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except (ValueError, AttributeError):
            pass
        raise HttpError(f"{method} {path}: HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, OSError) as e:
        raise HttpError(f"{BASE} に繋がらない（地図のサーバーは動いているか）: {e}") from e


def get_json(path: str, query: dict | None = None) -> dict:
    return json.loads(request("GET", path, query=query)[1])


def post_json(path: str, body: dict) -> dict:
    return json.loads(request("POST", path, body)[1])


def text(obj) -> dict:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=1)
    return {"type": "text", "text": s}


def image(data: bytes, mime: str) -> dict:
    return {"type": "image", "data": base64.b64encode(data).decode(), "mimeType": mime}


def _pick(args: dict, *keys) -> dict:
    return {k: args[k] for k in keys if args.get(k) is not None}


def queue(kind: str, args: dict, *keys) -> list[dict]:
    r = post_json("/command", {"kind": kind, **_pick(args, "note", "session", *keys)})
    head = f"queued: {r['queued']}  id: {r['id']}  pi_support: {r['pi_support']}\n{r['note']}" + (
        "" if r["pi_support"] else "\n※ Pi が指令に対応するまで、ロボットは実際には動かない。"
    )
    return [text(head), text(r)]


def call_tool(name: str, args: dict) -> list[dict]:
    if name == "state":
        return [text(get_json("/state", _pick(args, "session")))]
    if name == "look":
        meta = get_json("/frame.json", _pick(args, "sender"))
        _, jpg = request("GET", "/frame.jpg", query={"sender": meta.get("sender")})
        return [image(jpg, "image/jpeg"), text(meta)]
    if name == "map":
        q = _pick(args, "session")
        _, png = request("GET", "/map.png", query=q)
        return [image(png, "image/png"), text(get_json("/map.json", q))]
    if name == "annotate":
        body = _pick(args, "x", "y", "label", "note", "image_ref", "session")
        return [text(post_json("/annotate", body))]
    if name == "annotations":
        return [text(get_json("/annotations", _pick(args, "session")))]
    if name == "report_issue":
        body = _pick(args, "title", "body", "images", "session")
        r = post_json("/issue", body)
        return [text(f"下書きを書いた（GitHub には出していない）: {r['path']}"), text(r)]
    if name == "say":
        ctype, wav = request("POST", "/say", {"text": args.get("text", ""), "queue": True})
        seconds = max(0, len(wav) - 44) / 2 / 16000
        return [
            text(
                f"声を作って Pi の順番待ちに入れた（{seconds:.1f} 秒、{ctype}）。"
                "Pi が say/next に対応するまでは鳴らない。"
            )
        ]
    if name == "command_look":
        return queue("look", args, "pan_deg", "tilt_deg")
    if name == "command_move":
        return queue("move", args, "distance_cm", "speed", "steer_deg", "reverse")
    if name == "command_scan":
        return queue("scan", args)
    if name == "command_stop":
        return queue("stop", args)
    if name == "command_status":
        if args.get("id") is not None:
            return [text(get_json(f"/command/{int(args['id'])}"))]
        return [text(get_json("/commands"))]
    raise KeyError(name)


def handle(msg: dict) -> dict | None:
    """JSON-RPC の 1 件を処理し、返事（通知なら None）を返す。"""
    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}
    if mid is None:  # 通知（notifications/initialized など）には返さない
        return None

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code, message):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

    if method == "initialize":
        return ok(
            {
                "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": "PiCar-X（snoo の遊び相手のロボット）を見て・印を付けて・動かす。"
                + SAFETY
                + NOT_YET,
            }
        )
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        if name not in {t["name"] for t in TOOLS}:
            return err(-32602, f"道具 {name!r} は無い")
        try:
            content = call_tool(name, params.get("arguments") or {})
            return ok({"content": content, "isError": False})
        except (HttpError, ValueError, KeyError, TypeError) as e:
            return ok({"content": [text(f"失敗: {e}")], "isError": True})
    return err(-32601, f"{method} は無い")


def serve(stdin=sys.stdin, stdout=sys.stdout) -> None:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            reply = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "JSON でない"},
            }
        else:
            reply = handle(msg) if isinstance(msg, dict) else None
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
