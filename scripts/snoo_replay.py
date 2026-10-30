"""保存したセッションの 1 歩を、GPU を使わずに決め直す（地図のサーバーを直したときに確かめる）。

  python -m scripts.snoo_replay <セッションの名前か置き場> [--out <出力先>]
  make replay SESSION=explore_20261029_223944

--station N を渡すと、そのステーションの recon.npz（地図のサーバーがステーションごとに残す
VGGT の点）を使って、どのステーションの 1 歩でも決め直す。前の返事はそのステーションの reply.json。
渡さなければ、セッションの vggt.npz（毎回上書きされるので、作ったステーション＝ふつうは最後
だけ）を使う。VGGT の代わりにそれを読み、サーバーと同じ plan_station を通す。

もとのセッションには書かない。ステーションの画像と meta.json を出力先（既定は
/tmp/snoo-replay/<セッション>）に写し、poses.json・reverse.json（と位置のグラフの
graph_edges.jsonl・graph_poses.json）はそのステーションより前の分だけに戻してから決め直す。
出力先には room.json・topdown.png・plan.json などができる。
前の返事（plan.json の reply）と同じかを出す。違えば終了コード 1（--allow-diff で 0）。
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from app.snoo_map_server import (
    DATA,
    RECON_FILE,
    Recon,
    load_recon,
    plan_station,
    station_dir,
    station_ids,
)
from app.snoo_mapper import MapParams, VggtOut, frame_colors, load_vggt, unproject


class ReplayError(ValueError):
    pass


def frame_key(path) -> tuple[str, str]:
    """（ステーションの置き場の名前, 画像の名前）。写した先でも同じになる。"""
    p = Path(path)
    return p.parent.name, p.name


def npz_station(v: VggtOut) -> int:
    """vggt.npz を作ったステーション（入っているコマのうち、いちばん新しいステーション）。"""
    return max(int(frame_key(f)[0]) for f in v.files)


def pick(v: VggtOut, keys: list[tuple[str, str]]) -> VggtOut:
    """vggt.npz のコマを、サーバーが選んだコマの並びにする。選んだコマが vggt.npz に無ければ
    （コマの選び方を変えて、前に VGGT に入れなかったコマを選んだ）決め直せない。
    """
    index = {frame_key(f): i for i, f in enumerate(v.files)}
    missing = [k for k in keys if k not in index]
    if missing:
        names = ", ".join("/".join(k) for k in missing[:5])
        raise ReplayError(f"vggt.npz に無いコマを選んだ（{len(missing)} コマ: {names} …）")
    ix = [index[k] for k in keys]
    return VggtOut(
        v.extrinsic[ix], v.intrinsic[ix], v.depth[ix], v.depth_conf[ix], [v.files[i] for i in ix]
    )


def npz_reconstruct(v: VggtOut):
    """VGGT の代わり（app/snoo_map_server.py の VggtReconstruct と同じ点を作る）。"""

    def reconstruct(refs, npz, marks):
        t = time.perf_counter()
        w = pick(v, [frame_key(r.path) for r in refs])
        params = MapParams()
        points, conf = unproject(w, params.stride, params.border_px)
        colors = frame_colors([r.path for r in refs], w.depth.shape[1:], params.stride)
        marks["vggt"] = 0.0
        marks["points"] = time.perf_counter() - t
        return Recon(w.extrinsic, points, conf, colors)

    return reconstruct


def recon_reconstruct(keys: list[str], recon: Recon):
    """ステーションの recon.npz を返す VGGT の代わり。

    選んだコマが残したときと違えば決め直せない。
    """

    def reconstruct(refs, npz, marks):
        now = [f"{Path(r.path).parent.name}/{Path(r.path).name}" for r in refs]
        if now != keys:
            raise ReplayError("コマの選び方が recon.npz を残したときと違う（決め直せない）")
        marks["vggt"] = 0.0
        return recon

    return reconstruct


def _cut(src: Path, dst: Path, before: int) -> None:
    """station_id の書いてある JSON（poses.json・reverse.json）を、before より前の分だけ写す。"""
    if not src.exists():
        return
    data = {k: v for k, v in json.loads(src.read_text()).items() if int(k) < before}
    dst.write_text(json.dumps(data, indent=1))


def prepare(session: Path, out: Path, station: int) -> None:
    """station を受けた直後（決める前）のセッションを out に作る。"""
    if out.resolve() == session.resolve():
        raise ReplayError("出力先がもとのセッションと同じ")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for i in station_ids(session):
        if i <= station:
            shutil.copytree(station_dir(session, i), station_dir(out, i))
    _cut(session / "poses.json", out / "poses.json", station)
    _cut(session / "reverse.json", out / "reverse.json", station)
    _cut_graph(session, out, station)


def _cut_graph(session: Path, out: Path, before: int) -> None:
    """位置のグラフ（graph_edges.jsonl・graph_poses.json）を、before より前の分だけ写す。
    graph_poses.json の位置は最後に解いたもので、受けたときのものではない（始めの位置に使うだけ）。
    """
    from app.snoo_map_server import GRAPH_EDGES, GRAPH_POSES

    edges = session / GRAPH_EDGES
    if edges.exists():
        rows = [x for x in edges.read_text().splitlines() if x.strip()]
        keep = [x for x in rows if json.loads(x)["station"] < before]
        (out / GRAPH_EDGES).write_text("".join(x + "\n" for x in keep))
    state = session / GRAPH_POSES
    if state.exists():
        d = json.loads(state.read_text())
        d["poses"] = {k: v for k, v in d.get("poses", {}).items() if int(k) < before}
        d["localized"] = [i for i in d.get("localized", []) if i < before]
        d["bad"] = [i for i in d.get("bad", []) if i < before]
        (out / GRAPH_POSES).write_text(json.dumps(d, indent=1))


def previous_reply(session: Path, station: int) -> dict | None:
    """前にそのステーションで Pi に返したもの（plan.json がそのステーションのものなら）。"""
    p = session / "plan.json"
    if not p.exists():
        return None
    plan = json.loads(p.read_text())
    if ((plan.get("window") or {}).get("stations") or [None])[-1] != station:
        return None
    return plan.get("reply") or plan.get("next")


def same_reply(a: dict, b: dict) -> bool:
    """map_url（出力先の名前が入る）のほかが同じか。"""
    return {k: v for k, v in a.items() if k != "map_url"} == {
        k: v for k, v in b.items() if k != "map_url"
    }


def replay(
    session: Path, out: Path, station: int | None = None
) -> tuple[int, dict, dict | None, dict]:
    """(ステーション, 決め直した返事, 前の返事, 時間)。"""
    if station is not None:
        path = station_dir(session, station) / RECON_FILE
        if not path.exists():
            raise ReplayError(f"{path} が無い（recon.npz を残す前の版で受けたステーション）")
        keys, recon = load_recon(path)
        prepare(session, out, station)
        marks: dict = {}
        t = time.perf_counter()
        reply = plan_station(out, station, recon_reconstruct(keys, recon), marks)
        marks["total"] = time.perf_counter() - t
        before_path = station_dir(session, station) / "reply.json"
        before = json.loads(before_path.read_text()) if before_path.exists() else None
        return station, reply, before, marks
    npz = session / "vggt.npz"
    if not npz.exists():
        raise ReplayError(f"{npz} が無い（VGGT を回したステーションが無い）")
    v = load_vggt(npz)
    station = npz_station(v)
    prepare(session, out, station)
    marks: dict = {}
    t = time.perf_counter()
    reply = plan_station(out, station, npz_reconstruct(v), marks)
    marks["total"] = time.perf_counter() - t
    return station, reply, previous_reply(session, station), marks


def resolve(name: str) -> Path:
    p = Path(name)
    if p.is_dir():
        return p
    p = Path(DATA) / name
    if p.is_dir():
        return p
    raise ReplayError(f"{name} というセッションは無い（{DATA} の下か、置き場を渡す）")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", help="セッションの名前（SNOO_MAP_DATA の下）か置き場")
    parser.add_argument("--out", type=Path, help="既定は /tmp/snoo-replay/<セッション>")
    parser.add_argument("--station", type=int, help="このステーションの recon.npz で決め直す")
    parser.add_argument("--allow-diff", action="store_true", help="前の返事と違っても 0 で終わる")
    args = parser.parse_args()
    try:
        session = resolve(args.session)
        out = args.out or Path("/tmp/snoo-replay") / session.name
        station, reply, before, marks = replay(session, out, args.station)
    except ReplayError as e:
        print(e)
        sys.exit(2)
    print(f"{session.name}/{station:03d} を決め直した（出力: {out}）")
    print("秒: " + " ".join(f"{k} {v:.2f}" for k, v in marks.items()))
    print("今の返事: " + json.dumps(reply, ensure_ascii=False))
    if before is None:
        print("前の返事: 残っていない（比べない）")
        return
    print("前の返事: " + json.dumps(before, ensure_ascii=False))
    same = same_reply(reply, before)
    print("前と同じ" if same else "前と違う")
    if not same and not args.allow_diff:
        sys.exit(1)


if __name__ == "__main__":
    main()
