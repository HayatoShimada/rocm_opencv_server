"""snoo の 2 つのサーバー（検出の Docker の snoo と、地図のホストのサービス）を直して確かめる。
ふつうは Makefile から呼ぶ（README の「開発の流れ」）。

  python -m scripts.snoo_dev status            # 動いているか・p50/p95・セッション・今のフレーム
  python -m scripts.snoo_dev deploy-detector   # app/ を重ねた snoo を起動し直し、待ち受けまで待つ
  python -m scripts.snoo_dev deploy-map        # 地図のサービスを起動し直し、/twin を確かめる

待つ間は、いまどの準備をしているかと経った秒数を 1 秒ごとに出す。どれも上限の秒数を超えたらやめる。
標準のライブラリだけで書く（どの python でも動くように）。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTAINER = os.getenv("SNOO_CONTAINER", "rocm_opencv_server-snoo-1")
COMPOSE = ["docker", "compose", "-f", "compose.yaml", "-f", "compose.dev.yaml"]
MAP_UNIT = os.getenv("SNOO_MAP_UNIT", "snoo-map")
MAP_URL = os.getenv("SNOO_MAP_URL", "http://127.0.0.1:8720")
# 変えたらイメージを作り直す（docker compose build snoo）ファイル。app/ は重ねるので要らない
IMAGE_INPUTS = ("pyproject.toml", "uv.lock", "Dockerfile")

# --- ログを読む（テストする所） ---

# app/snoo.py が 10 秒ごとに書く行
_LATENCY = re.compile(
    r"^(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO (?P<fps>[\d.]+) fps、受けてから返すまで "
    r"中央値 (?P<median>[\d.]+) ms・最大 (?P<max>[\d.]+) ms｜p50/p95 ms: (?P<stages>[^｜]*)"
    r"(?:｜(?P<rest>.*))?$"
)
_STAGE = re.compile(r"(\S+) ([\d.]+)/([\d.]+)")


def parse_latency(line: str) -> dict | None:
    """p50/p95 の行を dict にする。違う行なら None。"""
    m = _LATENCY.match(line.strip())
    if m is None:
        return None
    return {
        "time": datetime.strptime(m["time"], "%Y-%m-%d %H:%M:%S"),
        "fps": float(m["fps"]),
        "median_ms": float(m["median"]),
        "max_ms": float(m["max"]),
        "stages": {n: (float(a), float(b)) for n, a, b in _STAGE.findall(m["stages"])},
        "rest": (m["rest"] or "").strip(),
        "line": line.strip(),
    }


def last_latency(text: str) -> dict | None:
    for line in reversed(text.splitlines()):
        r = parse_latency(line)
        if r is not None:
            return r
    return None


# ログに出てくるもの → いまの準備（後ろのものほど先へ進んでいる）
DETECTOR_STAGES = [
    ("rtdetr", "検出のモデル（RT-DETR）を読んでいる"),
    ("ViT-B-16", "CLIP を読んでいる"),
    ("torch/_dynamo", "検出を torch.compile している（キャッシュがあれば 10 秒ほど）"),
    ("clear_cm を見積もる（", "深度のモデルを読んで torch.compile している"),
    ("clear_cm を見積もる準備ができた", "受け始めるところ"),
    ("待ち受け", "待ち受け"),
]
MAP_STAGES = [
    ("で待ち受け（", "VGGT を読んでいる"),
    ("VGGT を読んだ", "HTTP を開くところ"),
    ("Uvicorn running", "待ち受け"),
]


def stage(text: str, stages: list[tuple[str, str]], start: str = "python を起動している") -> str:
    """ログから、いまどの準備をしているか。"""
    now = start
    for key, name in stages:
        if key in text:
            now = name
    return now


def find_traceback(text: str) -> str | None:
    """起動に失敗したときの Traceback（最後のもの）。無ければ None。"""
    i = text.rfind("Traceback (most recent call last)")
    return text[i:] if i >= 0 else None


def image_stale(image_created: float, inputs: dict[str, float]) -> list[str]:
    """イメージを作った後に変わった、作り直しの要るファイル。"""
    return sorted(name for name, t in inputs.items() if t > image_created)


def live_lines(live: dict) -> list[str]:
    """/twin/api/live の、送り元ごとの 1 行。"""
    if not live.get("enabled"):
        return ["今のフレーム: 書いていない（SNOO_MAP_LIVE_DIR が空）"]
    out = []
    for s in live.get("senders") or []:
        age = s.get("age_s")
        cmd = s.get("cmd") or {}
        out.append(
            f"今のフレーム: {s.get('sender')} {age if age is not None else '?'} 秒前 "
            f"seq {s.get('seq')} 検出 {s.get('infer_ms')} ms 猫 {len(s.get('dets') or [])} "
            f"{'走っている' if cmd.get('moving') else '止まっている'}"
            + (f" clear {s['clear_cm']} cm" if s.get("clear_cm") is not None else "")
        )
    return out or ["今のフレーム: まだ届いていない"]


def session_lines(sessions: list[dict], state: dict | None) -> list[str]:
    """/twin/api/sessions と、いちばん新しいセッションの /twin/api/<id>/state の行。"""
    out = [f"セッション: {len(sessions)} 個"]
    if not sessions:
        return out
    s = sessions[0]
    when = datetime.fromtimestamp(s["mtime"]).strftime("%m-%d %H:%M:%S") if s.get("mtime") else "?"
    out.append(f"いちばん新しい: {s['id']}（{s.get('stations')} か所、{when}）")
    if state:
        reply = state.get("reply")
        timing = state.get("timing_s") or {}
        out.append(
            f"最後の返事（ステーション {state.get('reply_station')}）: "
            + (json.dumps(_short_reply(reply), ensure_ascii=False) if reply else "まだ無い")
        )
        if timing:
            out.append("かかった秒: " + " ".join(f"{k} {v}" for k, v in timing.items()))
    return out


def _short_reply(reply: dict) -> dict:
    """返事のうち、見て分かる所だけ（segments・map_url などは省く）。"""
    keep = ("action", "reason", "reverse", "steer_deg", "speed", "distance_cm", "target")
    return {k: reply[k] for k in keep if k in reply}


def latency_text(r: dict | None, now: datetime | None = None) -> str:
    if r is None:
        return "p50/p95: まだ無い（フレームが届いていない）"
    age = ((now or datetime.now()) - r["time"]).total_seconds()
    stages = "  ".join(f"{n} {a:.2f}/{b:.2f}" for n, (a, b) in r["stages"].items())
    return (
        f"p50/p95（{age:.0f} 秒前）: {r['fps']:.1f} fps 中央値 {r['median_ms']:.1f} ms "
        f"最大 {r['max_ms']:.1f} ms｜{stages}" + (f"｜{r['rest']}" if r["rest"] else "")
    )


# --- 外を呼ぶ ---


def run(cmd: list[str], check: bool = False, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=check, **kw)


def user_env() -> dict:
    """systemctl --user を、ログインの外（cron・ssh のコマンド）からでも呼べるように。"""
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return env


def docker_logs(since: str | None = None, tail: int | None = None) -> str:
    cmd = ["docker", "logs", CONTAINER]
    if since:
        cmd += ["--since", since]
    if tail:
        cmd += ["--tail", str(tail)]
    p = run(cmd)
    return p.stdout + p.stderr


def container_info() -> dict | None:
    p = run(["docker", "inspect", CONTAINER])
    if p.returncode != 0:
        return None
    return json.loads(p.stdout)[0]


def http_json(path: str, timeout: float = 3.0):
    with urllib.request.urlopen(MAP_URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def http_status(path: str, timeout: float = 3.0) -> int | None:
    try:
        with urllib.request.urlopen(MAP_URL + path, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError):
        return None


def wait_until(check, describe, timeout: float, interval: float = 1.0) -> float:
    """check() が真になるまで、describe() と経った秒数を出しながら待つ。経った秒数を返す。
    check が例外を投げたら（起動に失敗したなど）そのまま上に投げる。上限を超えたら TimeoutError。
    """
    t0 = time.monotonic()
    last = ""
    while not check():
        elapsed = time.monotonic() - t0
        if elapsed >= timeout:
            print()
            raise TimeoutError(f"{timeout:.0f} 秒待っても終わらなかった（{describe()}）")
        line = f"  [{elapsed:5.1f} 秒] {describe()}"
        print("\r" + line.ljust(len(last)), end="", flush=True)
        last = line
        time.sleep(interval)
    elapsed = time.monotonic() - t0
    print("\r" + f"  [{elapsed:5.1f} 秒] {describe()}".ljust(len(last)), flush=True)
    return elapsed


# --- deploy-detector ---


def is_main_worktree() -> bool:
    """git worktree の中から compose を動かすと、別のプロジェクト名でコンテナを作ってしまう。"""
    d = run(["git", "rev-parse", "--git-dir"]).stdout.strip()
    c = run(["git", "rev-parse", "--git-common-dir"]).stdout.strip()
    return bool(d) and Path(ROOT, d).resolve() == Path(ROOT, c).resolve()


def deploy_detector(timeout: float, latency_timeout: float) -> int:
    t0 = time.monotonic()
    if not is_main_worktree():
        print(
            "git worktree の中では動かさない（別のコンテナを作ってしまう）。main の置き場で動かす"
        )
        return 2
    image = run(["docker", "image", "inspect", "-f", "{{.Created}}", "rocm-opencv-server-snoo"])
    if image.returncode == 0:
        created = _docker_time(image.stdout.strip())
        stale = image_stale(created, {n: (ROOT / n).stat().st_mtime for n in IMAGE_INPUTS})
        if stale:
            print(
                f"注意: {', '.join(stale)} がイメージより新しい。`docker compose build snoo` が要る"
            )

    before = container_info()
    before_start = before["State"]["StartedAt"] if before else None
    # 上書き（app/ を重ねる・init）が無いコンテナや、環境変数が変わったときだけ作り直される
    p = run([*COMPOSE, "up", "-d", "--no-build", "snoo"])
    if p.returncode != 0:
        print(p.stdout + p.stderr)
        return 1
    info = container_info()
    if info is None:
        print(f"{CONTAINER} が無い")
        return 1
    if info["State"]["StartedAt"] == before_start:
        print("snoo を起動し直す（app/ を読み直す）")
        p = run([*COMPOSE, "restart", "-t", "5", "snoo"])
        if p.returncode != 0:
            print(p.stdout + p.stderr)
            return 1
        info = container_info()
    else:
        print("snoo を作り直した（compose.dev.yaml の上書き・環境変数が変わった）")
    if not any(m.get("Destination") == "/app/app" for m in info.get("Mounts", [])):
        print("注意: app/ が重なっていない（compose.dev.yaml が効いていない）")
    rc = wait_detector(info["State"]["StartedAt"], timeout, latency_timeout)
    print(f"合わせて {time.monotonic() - t0:.1f} 秒")
    return rc


def wait_detector(since: str, timeout: float, latency_timeout: float) -> int:
    """since（コンテナを起動した時刻）からのログで、待ち受けと最初の p50/p95 を待つ。"""
    logs = {"text": ""}

    def ready() -> bool:
        logs["text"] = docker_logs(since=since)
        tb = find_traceback(logs["text"])
        if tb is not None:
            raise RuntimeError(tb)
        state = (container_info() or {}).get("State", {})
        if not state.get("Running"):
            raise RuntimeError(
                f"コンテナが止まった（{state.get('Status')}）\n{logs['text'][-3000:]}"
            )
        return "待ち受け" in logs["text"]

    try:
        waited = wait_until(ready, lambda: stage(logs["text"], DETECTOR_STAGES), timeout)
    except RuntimeError as e:
        print(f"\n起動に失敗した:\n{e}")
        return 1
    except TimeoutError as e:
        print(e)
        return 1
    print(f"待ち受けまで {waited:.1f} 秒（起動し直してから）")

    # p50/p95 は 10 秒ごと、フレームが届いているときだけ出る
    got = {"r": None}

    def latency() -> bool:
        got["r"] = last_latency(docker_logs(since=since))
        return got["r"] is not None

    try:
        wait_until(
            latency, lambda: "最初の p50/p95 を待っている（10 秒ごとに出る）", latency_timeout
        )
        print(latency_text(got["r"]))
    except TimeoutError:
        print("p50/p95: まだ無い（Pi がフレームを送っていない）")
    return 0


def _docker_time(s: str) -> float:
    """docker の時刻（2026-10-29T13:21:03.732464743Z）を UNIX 時刻にする。"""
    s = re.sub(r"(\.\d{6})\d*", r"\1", s.strip()).replace("Z", "+00:00")
    return datetime.fromisoformat(s).timestamp()


# --- deploy-map ---


def user(*cmd: str) -> subprocess.CompletedProcess:
    """systemctl --user・journalctl --user を呼ぶ。"""
    return subprocess.run(cmd, capture_output=True, text=True, env=user_env())


def journal(since: float) -> str:
    since_arg = f"@{since:.0f}"
    return user("journalctl", "--user", "-u", MAP_UNIT, "--since", since_arg, "-o", "cat").stdout


# 探索中に起動し直すなら、返事を返してからこの秒までに始める（Pi は 1 歩走って首を振るので、
# 次のステーションを送るまで 10 秒ほどある。起動し直しは 1 秒ほど）
LIVE_WINDOW_S = 3.0
# これより長く返事が無ければ、探索していないとみなしてすぐ起動し直す
LIVE_IDLE_S = 60.0


def busy_state() -> dict | None:
    try:
        with urllib.request.urlopen(f"{MAP_URL}/v1/snoo/map/busy", timeout=2) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def wait_between_stations(timeout: float) -> bool:
    """Pi が 1 歩走っている間（move を返した直後）か、探索していないときまで待つ。"""
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        b = busy_state()
        if b is None:  # 古いサーバー（busy が無い）・止まっている
            return True
        age = b.get("since_reply_s")
        if b.get("in_flight", 0) == 0 and (age is None or age >= LIVE_IDLE_S):
            return True
        if b.get("in_flight", 0) == 0 and age is not None and age <= LIVE_WINDOW_S:
            print(f"返事を返して {age:.1f} 秒（Pi は走っている）。いま起動し直す")
            return True
        time.sleep(0.2)
    return False


def deploy_map(timeout: float, wait_s: float = 90.0) -> int:
    t0 = time.monotonic()
    if user("systemctl", "--user", "cat", MAP_UNIT).returncode != 0:
        print(f"{MAP_UNIT} のサービスが無い。README の「開発の流れ」の入れ方で入れる")
        return 2
    # 探索を止めずに入れ替える：ステーションを受けている間は起動し直さない
    if not wait_between_stations(wait_s):
        print(f"{wait_s:.0f} 秒待っても、ステーションの間が来なかった。起動し直さない")
        return 1
    since = time.time() - 1
    p = user("systemctl", "--user", "restart", MAP_UNIT)
    if p.returncode != 0:
        print(p.stderr)
        return 1
    logs = {"text": ""}

    def ready() -> bool:
        logs["text"] = journal(since)
        active = user("systemctl", "--user", "is-active", MAP_UNIT).stdout.strip()
        if active == "failed" or find_traceback(logs["text"]):
            raise RuntimeError(logs["text"][-3000:])
        return "Uvicorn running" in logs["text"] and http_status("/twin") == 200

    try:
        waited = wait_until(ready, lambda: stage(logs["text"], MAP_STAGES), timeout, 0.5)
    except RuntimeError as e:
        print(f"\n起動に失敗した:\n{e}")
        return 1
    except TimeoutError as e:
        print(e)
        return 1
    print(f"待ち受けまで {waited:.1f} 秒。/twin は 200")
    try:
        for line in map_lines():
            print(line)
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"/twin/api を読めない: {e}")
        return 1
    print(f"合わせて {time.monotonic() - t0:.1f} 秒")
    return 0


# --- status ---


def map_lines() -> list[str]:
    sessions = http_json("/twin/api/sessions")["sessions"]
    state = http_json(f"/twin/api/{sessions[0]['id']}/state") if sessions else None
    return session_lines(sessions, state) + live_lines(http_json("/twin/api/live"))


def status() -> int:
    ok = True
    print("== 検出（Docker の snoo）")
    info = container_info()
    if info is None:
        print(f"{CONTAINER} が無い")
        ok = False
    else:
        st = info["State"]
        dev = any(m.get("Destination") == "/app/app" for m in info.get("Mounts", []))
        started = datetime.fromtimestamp(_docker_time(st["StartedAt"])).strftime("%m-%d %H:%M:%S")
        kind = "app/ を重ねた開発用" if dev else "イメージのまま"
        print(f"{st['Status']}（{started} から、{kind}）")
        ok &= bool(st.get("Running"))
        text = docker_logs(since=st["StartedAt"])
        if "待ち受け" not in text:
            print(f"準備中: {stage(text, DETECTOR_STAGES)}")
        print(latency_text(last_latency(text)))

    print("== 地図（ホストの snoo_map_server）")
    active = user("systemctl", "--user", "is-active", MAP_UNIT).stdout.strip()
    if user("systemctl", "--user", "cat", MAP_UNIT).returncode != 0:
        active = "入っていない（手で起動したもの？）"
    pids = run(["pgrep", "-f", "python -m app.snoo_map_server"]).stdout.split()
    print(f"サービス {MAP_UNIT}: {active or '無い'}、プロセス: {' '.join(pids) or '無い'}")
    code = http_status("/twin")
    print(f"{MAP_URL}/twin: {code or '応えない'}")
    if code == 200:
        try:
            for line in map_lines():
                print(line)
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"/twin/api を読めない: {e}")
            ok = False
    else:
        ok = False
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    d = sub.add_parser("deploy-detector")
    d.add_argument("--timeout", type=float, default=180)
    d.add_argument("--latency-timeout", type=float, default=15)
    w = sub.add_parser(
        "wait-detector", help="起動し直した snoo の待ち受けを待つだけ（本番の up の後にも）"
    )
    w.add_argument("--timeout", type=float, default=180)
    w.add_argument("--latency-timeout", type=float, default=15)
    m = sub.add_parser("deploy-map")
    m.add_argument("--timeout", type=float, default=60)
    m.add_argument("--wait", type=float, default=90, help="ステーションの間を待つ秒")
    args = parser.parse_args()
    if args.cmd == "status":
        sys.exit(status())
    if args.cmd == "deploy-detector":
        sys.exit(deploy_detector(args.timeout, args.latency_timeout))
    if args.cmd == "wait-detector":
        info = container_info()
        if info is None:
            print(f"{CONTAINER} が無い")
            sys.exit(1)
        sys.exit(wait_detector(info["State"]["StartedAt"], args.timeout, args.latency_timeout))
    sys.exit(deploy_map(args.timeout, args.wait))


if __name__ == "__main__":
    main()
