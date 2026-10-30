/* /drive：実機の PiCar-X を手で動かす（app/snoo_teleop.py）。入力と送る周りは controller.js。
   左のスティック＝速さとハンドル、右のスティック＝首。サーバーが -1..1 を Pi の単位に直して上限で切る。 */
(function () {
  "use strict";
  const C = window.SnooController;
  const T = "/v1/snoo/teleop";
  const A = "/v1/snoo/agent";
  const $ = (id) => document.getElementById(id);

  // 首を保つとき：右のスティックは向きを変える速さ（1 秒で端から端の半分）
  const HEAD_RATE = 1.0;
  let hold = false;
  let head = { pan: 0, tilt: 0 };
  let lastSendT = performance.now();
  let teleop = null; // /status の最後の返事
  let agent = null; // /v1/snoo/agent/state の最後の返事

  function toast(text) {
    const t = $("toast");
    t.textContent = text;
    t.classList.add("show");
    clearTimeout(t._h);
    t._h = setTimeout(() => t.classList.remove("show"), 2200);
  }

  function headFrom(s) {
    const now = performance.now();
    const dt = Math.min(0.3, (now - lastSendT) / 1000);
    lastSendT = now;
    if (!hold) return { pan: s.rx, tilt: s.ry };
    head.pan = Math.max(-1, Math.min(1, head.pan + s.rx * HEAD_RATE * dt));
    head.tilt = Math.max(-1, Math.min(1, head.tilt + s.ry * HEAD_RATE * dt));
    return head;
  }

  async function send(s, mode) {
    const h = headFrom(s);
    const body = { seq: C.nextSeq(), steer: s.lx, speed: s.ly, pan: h.pan, tilt: h.tilt };
    if (mode) body.mode = mode;
    const r = await C.postJSON(T, body);
    if (r && !r.error) showTeleop(r);
    else setPill("pill-op", "bad", r && r.error === 403 ? "この機械は入れない（403）" : "送れない");
    return r;
  }

  async function stop(reason) {
    const r = await C.postJSON(T + "/stop", { reason: String(reason || "stop").slice(0, 60) });
    if (r && !r.error) showTeleop(r);
  }

  function setPill(id, cls, text) {
    const p = $(id);
    p.className = "pill " + (cls || "");
    p.querySelector("span").textContent = text;
  }

  function fmt(v, unit, digits) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    return (typeof v === "number" ? v.toFixed(digits === undefined ? 1 : digits) : v) + (unit || "");
  }

  function showTeleop(st) {
    teleop = Object.assign(teleop || {}, st);
    const op = st.operator || {};
    if (op.present) setPill("pill-op", st.stopped ? "warn" : "ok", st.stopped ? "止めている" : "送っている");
    else setPill("pill-op", "", "送っていない");
    const pi = st.pi;
    if (!pi || !pi.connected) setPill("pill-pi", "bad", pi ? `Pi：${fmt(pi.age_s, " 秒", 0)}前から来ない` : "Pi：まだ見えない");
    else if (pi.accepted) setPill("pill-pi", "ok", "Pi：teleop で受けている");
    else setPill("pill-pi", "warn", `Pi：${pi.mode || "?"}（teleop ではない）`);
    $("not-accepted").hidden = !(pi && pi.connected && !pi.accepted);
    render();
  }

  function render() {
    const st = teleop || {};
    const pi = st.pi || {};
    const ag = agent || {};
    const cmd = st.command || {};
    const ap = pi.applied || {};
    const snoo = ag.snoo;
    const rows = [
      ["モード", pi.mode || ag.mode || "—"],
      ["電池", fmt(pi.battery_v ?? ag.battery_v, " V", 2)],
      ["clear_cm", fmt(pi.clear_cm ?? ag.clear_cm, " cm", 0)],
      ["超音波", fmt(pi.sonar_cm ?? ag.sonar_cm, " cm", 0)],
      ["snoo", snoo ? (snoo.distance_cm != null ? `${snoo.distance_cm.toFixed(0)} cm` : "見えている（距離不明）")
        + (snoo.bearing_deg != null ? ` ${snoo.bearing_deg > 0 ? "右" : "左"} ${Math.abs(snoo.bearing_deg).toFixed(0)}°` : "")
        : "見えない"],
      ["送った指令", `速さ ${fmt(cmd.speed, "", 1)}  ハンドル ${fmt(cmd.steer_deg, "°", 0)}  首 ${fmt(cmd.pan_deg, "°", 0)} / ${fmt(cmd.tilt_deg, "°", 0)}`],
      ["Pi が書いた", ap.speed == null ? "—" : `速さ ${fmt(ap.speed)}  ハンドル ${fmt(ap.steer_deg, "°", 0)}  首 ${fmt(ap.pan_deg, "°", 0)} / ${fmt(ap.tilt_deg, "°", 0)}`],
      ["弱めた理由", (pi.limit && pi.limit.length) ? pi.limit.join("・") : "—"],
      ["カメラ", ag.frame_age_s != null ? `${ag.frame_age_s.toFixed(1)} 秒前` : "—"],
    ];
    $("kv").innerHTML = "";
    for (const [k, v] of rows) {
      const dt = document.createElement("dt");
      dt.textContent = k;
      const dd = document.createElement("dd");
      dd.textContent = v;
      $("kv").append(dt, dd);
    }
  }

  async function poll(url, every, fn) {
    for (;;) {
      if (!document.hidden) {
        try {
          const r = await fetch(url, { cache: "no-store" });
          if (r.ok) fn(await r.json());
          else if (r.status === 403) setPill("pill-op", "bad", "この機械は入れない（403）");
        } catch (e) { /* つながらない間は待つ */ }
      }
      await new Promise((res) => setTimeout(res, every));
    }
  }

  const ctl = C.create({
    sticks: $("sticks"),
    labels: ["走る（前後・ハンドル）", "首"],
    send: (s) => send(s),
    stop,
    onButton: async (name) => {
      if (name === "a") horn();
      if (name === "y") center();
    },
  });
  ctl.onInput((s, o) => {
    $("inputs").textContent = `入力 ${o.source || "—"}  左 ${s.lx.toFixed(2)}, ${s.ly.toFixed(2)}  右 ${s.rx.toFixed(2)}, ${s.ry.toFixed(2)}`;
    if (o.pad) setPill("pill-pad", "ok", "ゲームパッド");
    else setPill("pill-pad", "", "ゲームパッド無し");
  });

  async function horn() {
    const r = await C.postJSON(A + "/say", { text: "プップー", queue: true });
    if (r === null) toast("鳴らせない（つながらない）");
    else if (r.error === 503) toast("声が使えない（pyopenjtalk が無い）");
    else if (r.error) toast(`鳴らせない（${r.error}）`);
    else toast("Pi に鳴らすよう頼んだ");
  }
  async function center() {
    head = { pan: 0, tilt: 0 };
    await send({ lx: 0, ly: 0, rx: 0, ry: 0 });
  }

  $("stop").addEventListener("click", () => ctl.stopNow());
  $("horn").addEventListener("click", horn);
  $("center").addEventListener("click", center);
  $("hold").addEventListener("click", () => {
    hold = !hold;
    $("hold").textContent = "首を保つ：" + (hold ? "入" : "切");
    $("hold").classList.toggle("on", hold);
  });
  $("mode-teleop").addEventListener("click", async () => {
    await send({ lx: 0, ly: 0, rx: 0, ry: 0 }, "teleop");
    toast("teleop に入るよう頼んだ（Pi が対応していれば入る）");
  });
  $("mode-idle").addEventListener("click", async () => {
    await stop("idle");
    await send({ lx: 0, ly: 0, rx: 0, ry: 0 }, "idle");
    await stop("idle");
    toast("idle に戻るよう頼んだ");
  });

  C.pollImage($("cam"), A + "/frame.jpg", 7, (ok) => {
    $("cam-empty").hidden = ok;
    $("cam").style.visibility = ok ? "visible" : "hidden";
  });
  const mapImg = $("map");
  function loadMap() {
    if (!document.hidden) mapImg.src = A + "/map.png?t=" + Date.now();
  }
  mapImg.addEventListener("error", () => { mapImg.alt = "地図はまだ無い"; });
  loadMap();
  setInterval(loadMap, 2000);
  poll(T + "/status", 500, showTeleop);
  poll(A + "/state", 1000, (s) => { agent = s; render(); });
})();
