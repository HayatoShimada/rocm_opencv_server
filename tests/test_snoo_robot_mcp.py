"""scripts/snoo_robot_mcp.py（MCP のサーバー）を、作り物の HTTP のサーバーに向けて stdio で試す。"""

import base64
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "snoo_robot_mcp.py"
JPEG = b"\xff\xd8fake-jpeg\xff\xd9"
PNG = b"\x89PNG\r\n\x1a\nfake"
WAV = b"RIFF" + b"\0" * 40 + b"\0" * 32000  # 1 秒（16 kHz・16 ビット）


class Fake(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        Fake.seen.append(("GET", u.path, parse_qs(u.query)))
        p = u.path.removeprefix("/v1/snoo/agent")
        if p == "/state":
            return self._send(200, {"battery_v": 7.8, "pi_support": False})
        if p == "/frame.json":
            return self._send(200, {"sender": "192.168.0.188", "seq": 3, "cats": []})
        if p == "/frame.jpg":
            return self._send(200, JPEG, "image/jpeg")
        if p == "/map.png":
            return self._send(200, PNG, "image/png")
        if p == "/map.json":
            return self._send(200, {"px_per_m": 120})
        if p == "/command/5":
            return self._send(200, {"id": 5, "status": "queued"})
        return self._send(404, {"detail": "無い"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        Fake.seen.append(("POST", u.path, body))
        p = u.path.removeprefix("/v1/snoo/agent")
        if p == "/command":
            ok = body.get("kind") != "move" or body.get("distance_cm", 0) <= 30
            return self._send(
                200,
                {
                    "queued": ok,
                    "id": 5 if ok else None,
                    "checks": [],
                    "pi_support": False,
                    "note": "並べた" if ok else "断った",
                },
            )
        if p == "/say":
            return self._send(200, WAV, "audio/wav")
        if p == "/annotate":
            return self._send(200, {"id": 1, **body})
        if p == "/issue":
            return self._send(200, {"path": "/tmp/x", "posted": False})
        return self._send(404, {"detail": "無い"})


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    Fake.seen = []
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def rpc(url: str, messages: list[dict]) -> list[dict]:
    """MCP のサーバーを起動し、1 行ずつ送って返事を読む。"""
    stdin = "".join(json.dumps(m) + "\n" for m in messages)
    out = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
        env={"SNOO_AGENT_URL": url, "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr
    return [json.loads(x) for x in out.stdout.splitlines()]


def call(i: int, name: str, args: dict | None = None) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": i,
        "method": "tools/call",
        "params": {"name": name, "arguments": args or {}},
    }


def test_initialize_and_list_tools(server):
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
    }
    note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    lst = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    r = rpc(server, [init, note, lst])
    assert [x["id"] for x in r] == [1, 2]  # 通知には返さない
    assert r[0]["result"]["serverInfo"]["name"] == "snoo-robot"
    assert "tools" in r[0]["result"]["capabilities"]
    names = {t["name"] for t in r[1]["result"]["tools"]}
    assert names >= {
        "state",
        "look",
        "map",
        "annotate",
        "report_issue",
        "say",
        "command_look",
        "command_move",
        "command_scan",
        "command_stop",
    }
    move = next(t for t in r[1]["result"]["tools"] if t["name"] == "command_move")
    assert "30 cm" in move["description"] and "150 cm" in move["description"]
    assert "pi_support: false" in move["description"]


def test_tools_call_the_http_api(server):
    r = rpc(
        server,
        [
            call(1, "state"),
            call(2, "look"),
            call(3, "map"),
            call(4, "command_move", {"distance_cm": 20, "speed": 8, "steer_deg": -10}),
            call(5, "command_move", {"distance_cm": 50, "speed": 8}),
            call(6, "say", {"text": "こんにちは"}),
            call(7, "annotate", {"x": 1.0, "y": 0.5, "label": "sofa"}),
            call(8, "report_issue", {"title": "t", "body": "b", "images": ["live"]}),
            call(9, "command_stop"),
            call(10, "command_status", {"id": 5}),
            call(11, "nope"),
        ],
    )
    by = {x["id"]: x for x in r}
    assert json.loads(by[1]["result"]["content"][0]["text"])["battery_v"] == 7.8
    img = by[2]["result"]["content"][0]
    assert img["type"] == "image" and img["mimeType"] == "image/jpeg"
    assert base64.b64decode(img["data"]) == JPEG
    assert by[3]["result"]["content"][0]["mimeType"] == "image/png"
    moved = by[4]["result"]["content"][0]["text"]
    assert "queued: True" in moved and "pi_support: False" in moved and "実際には動かない" in moved
    assert "queued: False" in by[5]["result"]["content"][0]["text"]
    assert "1.0 秒" in by[6]["result"]["content"][0]["text"]
    assert "GitHub には出していない" in by[8]["result"]["content"][0]["text"]
    assert json.loads(by[10]["result"]["content"][0]["text"])["status"] == "queued"
    assert by[11]["error"]["code"] == -32602
    posts = [b for m, p, b in Fake.seen if m == "POST" and p.endswith("/command")]
    assert posts[0] == {"kind": "move", "distance_cm": 20, "speed": 8, "steer_deg": -10}
    assert posts[-1] == {"kind": "stop"}


def test_http_errors_come_back_as_tool_errors(server):
    r = rpc(server, [call(1, "annotations")])  # 作り物のサーバーには無い
    assert r[0]["result"]["isError"] and "HTTP 404" in r[0]["result"]["content"][0]["text"]
    r = rpc("http://127.0.0.1:1", [call(1, "state")])
    assert r[0]["result"]["isError"] and "繋がらない" in r[0]["result"]["content"][0]["text"]
