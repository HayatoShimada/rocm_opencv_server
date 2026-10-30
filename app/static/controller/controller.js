/* 手の入力（ゲームパッド・キーボード・スマホの指の 2 本のスティック）をまとめて、決まった間隔で送る。
   実機を動かす /drive（app/snoo_teleop.py）と、シミュレーターの猫を動かす /sim
   （scripts/snoo_sim_live.py）が共に使う。外の CDN もビルドも使わない（素の JS）。

   使い方:
     const ctl = SnooController.create({
       sticks: document.getElementById("sticks"),   // 指のスティックを置く所（無くてもよい）
       send: (s) => fetch(...),   // s = {lx, ly, rx, ry}。-1..1。ly・ry は上が +
       stop: () => fetch(...),    // 離してから linger_s 経ったら 1 回だけ呼ぶ
       onButton: (name) => ...,   // "a" "b" "x" "y" "lb" "rb" "start" "1".."5"
     });

   送り方（deadman）:
   - 入力が 0 でない間は hz 回/秒で send。離してから linger_s 秒は 0 を送り続け、そのあと stop を 1 回
   - 画面が隠れた・窓から外れた・閉じるときは、すぐ stop（keepalive で、閉じる途中でも届くように）
   - keepalive_s を決めれば、入力が無くてもその間隔で 0 を送る（シミュレーターの猫を手で持ち続ける）
   受け手は「新しい指令が 0.3 秒来なければ止まる」ので、ここが止まっても（タブを落としても）車は止まる。
*/
(function () {
  "use strict";

  const DEAD = 0.12; // ゲームパッドのスティックの遊び
  const KEYS = {
    // 左のスティック（速さとハンドル。猫なら前と回る）
    KeyW: ["ly", 1], KeyS: ["ly", -1], KeyA: ["lx", -1], KeyD: ["lx", 1],
    // 右のスティック（首）
    ArrowUp: ["ry", 1], ArrowDown: ["ry", -1], ArrowLeft: ["rx", -1], ArrowRight: ["rx", 1],
  };
  const KEY_BUTTONS = {
    Space: "b", Escape: "b", KeyH: "a", KeyX: "x", KeyY: "y",
    Digit1: "1", Digit2: "2", Digit3: "3", Digit4: "4", Digit5: "5",
  };
  // 標準の並び（https://w3c.github.io/gamepad/#remapping）
  const PAD_BUTTONS = { 0: "a", 1: "b", 2: "x", 3: "y", 4: "lb", 5: "rb", 9: "start" };

  function clamp(v) { return Math.max(-1, Math.min(1, v)); }
  function dead(v) { return Math.abs(v) < DEAD ? 0 : (v - Math.sign(v) * DEAD) / (1 - DEAD); }
  function pick(...vs) {
    // 同じ軸に入力がいくつかあれば、いちばん大きいもの
    let best = 0;
    for (const v of vs) if (Math.abs(v) > Math.abs(best)) best = v;
    return best;
  }

  /* 指のスティック。el の中に、左右 2 つの丸を作る。指を置いた所が真ん中 */
  function makeStick(parent, side, label) {
    const zone = document.createElement("div");
    zone.className = "sc-stick sc-" + side;
    zone.innerHTML = '<div class="sc-base"><div class="sc-knob"></div></div><span class="sc-label"></span>';
    zone.querySelector(".sc-label").textContent = label;
    parent.appendChild(zone);
    const base = zone.querySelector(".sc-base");
    const knob = zone.querySelector(".sc-knob");
    const st = { x: 0, y: 0, id: null };
    function at(e) {
      const r = base.getBoundingClientRect();
      const R = r.width / 2;
      let dx = (e.clientX - (r.left + R)) / R;
      let dy = (e.clientY - (r.top + R)) / R;
      const n = Math.hypot(dx, dy);
      if (n > 1) { dx /= n; dy /= n; }
      st.x = dx; st.y = -dy;
      knob.style.transform = `translate(${dx * R * 0.6}px, ${dy * R * 0.6}px)`;
    }
    function end(e) {
      if (st.id !== e.pointerId) return;
      st.id = null; st.x = 0; st.y = 0;
      knob.style.transform = "";
      zone.classList.remove("on");
    }
    zone.addEventListener("pointerdown", (e) => {
      if (st.id !== null) return;
      st.id = e.pointerId;
      zone.setPointerCapture(e.pointerId);
      zone.classList.add("on");
      at(e);
      e.preventDefault();
    });
    zone.addEventListener("pointermove", (e) => { if (st.id === e.pointerId) at(e); });
    zone.addEventListener("pointerup", end);
    zone.addEventListener("pointercancel", end);
    zone.addEventListener("lostpointercapture", end);
    return st;
  }

  function create(opts) {
    const o = Object.assign({ hz: 10, linger_s: 0.5, keepalive_s: 0, labels: ["走る", "首"] }, opts);
    const keys = new Set();
    let touchL = { x: 0, y: 0 }, touchR = { x: 0, y: 0 };
    if (o.sticks) {
      o.sticks.classList.add("sc-sticks");
      touchL = makeStick(o.sticks, "left", o.labels[0]);
      touchR = makeStick(o.sticks, "right", o.labels[1]);
    }
    let padPrev = {};
    let lastActive = -1e9;
    let lastSend = -1e9;
    let stopped = true;
    let enabled = true;
    const listeners = [];
    const out = {
      state: { lx: 0, ly: 0, rx: 0, ry: 0 },
      source: "",
      pad: null,
      keepalive: !!o.keepalive_s,
      setEnabled(v) { enabled = v; if (!v) hardStop("disabled"); },
      stopNow: () => hardStop("button"),
      onInput(fn) { listeners.push(fn); },
    };

    function button(name) {
      if (name === "b") hardStop("button");
      if (o.onButton) o.onButton(name);
    }

    function isTyping(e) {
      const t = e.target;
      return t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT");
    }
    window.addEventListener("keydown", (e) => {
      if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return;
      if (KEYS[e.code]) { keys.add(e.code); e.preventDefault(); }
      else if (KEY_BUTTONS[e.code] && !e.repeat) { button(KEY_BUTTONS[e.code]); e.preventDefault(); }
    });
    window.addEventListener("keyup", (e) => keys.delete(e.code));

    function readPad() {
      const pads = navigator.getGamepads ? navigator.getGamepads() : [];
      for (const p of pads) {
        if (!p || !p.connected) continue;
        out.pad = p.id;
        const b = {};
        p.buttons.forEach((x, i) => { b[i] = x.pressed; });
        for (const [i, name] of Object.entries(PAD_BUTTONS)) {
          if (b[i] && !padPrev[i]) button(name);
        }
        padPrev = b;
        const ax = p.axes;
        // 十字キーも左のスティックとして使う（12 上・13 下・14 左・15 右）
        const dy = (b[12] ? 1 : 0) - (b[13] ? 1 : 0);
        const dx = (b[15] ? 1 : 0) - (b[14] ? 1 : 0);
        return {
          lx: pick(dead(ax[0] || 0), dx), ly: pick(-dead(ax[1] || 0), dy),
          rx: dead(ax[2] || 0), ry: -dead(ax[3] || 0),
        };
      }
      out.pad = null;
      return { lx: 0, ly: 0, rx: 0, ry: 0 };
    }

    function readKeys() {
      const s = { lx: 0, ly: 0, rx: 0, ry: 0 };
      for (const k of keys) { const [ax, v] = KEYS[k]; s[ax] += v; }
      for (const ax in s) s[ax] = clamp(s[ax]);
      return s;
    }

    function hardStop(reason) {
      stopped = true;
      lastActive = -1e9;
      if (o.stop) Promise.resolve(o.stop(reason)).catch(() => {});
    }

    function tick() {
      const now = performance.now() / 1000;
      const pad = readPad();
      const kb = readKeys();
      const s = {};
      for (const ax of ["lx", "ly", "rx", "ry"]) {
        const t = ax[0] === "l" ? touchL : touchR;
        s[ax] = clamp(pick(pad[ax], kb[ax], ax[1] === "x" ? t.x : t.y));
        s[ax] = Math.round(s[ax] * 1000) / 1000;
      }
      out.state = s;
      out.source = Math.abs(pad.lx) + Math.abs(pad.ly) + Math.abs(pad.rx) + Math.abs(pad.ry) > 0
        ? "pad" : keys.size ? "key" : (touchL.id != null || touchR.id != null) ? "touch" : "";
      for (const fn of listeners) fn(s, out);
      if (!enabled) return;
      const active = Math.abs(s.lx) + Math.abs(s.ly) + Math.abs(s.rx) + Math.abs(s.ry) > 0;
      if (active) lastActive = now;
      const linger = now - lastActive < o.linger_s;
      const alive = out.keepalive && o.keepalive_s && now - lastSend >= o.keepalive_s;
      if (active || linger || alive) {
        stopped = false;
        lastSend = now;
        Promise.resolve(o.send(s)).catch(() => {});
      } else if (!stopped) {
        hardStop("release");
      }
    }
    setInterval(tick, 1000 / o.hz);

    // 隠れた・外れた・閉じるときは止める
    document.addEventListener("visibilitychange", () => { if (document.hidden) hardStop("hidden"); });
    window.addEventListener("blur", () => hardStop("blur"));
    window.addEventListener("pagehide", () => hardStop("pagehide"));
    window.addEventListener("gamepadconnected", (e) => { out.pad = e.gamepad.id; });
    window.addEventListener("gamepaddisconnected", () => { out.pad = null; hardStop("pad"); });
    return out;
  }

  /* 送る番号。読み込み直しても前より大きくなるよう、ミリ秒の時刻から始める */
  let seq = Date.now();
  function nextSeq() { seq += 1; return seq; }

  /* JSON を POST する（閉じる途中でも届くよう keepalive）。返事の JSON か、失敗なら null */
  async function postJSON(url, body) {
    try {
      const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
        keepalive: true,
        cache: "no-store",
      });
      if (!r.ok) return { error: r.status, detail: await r.text() };
      return await r.json();
    } catch (e) {
      return null;
    }
  }

  /* 画像を読み終えてから次を読む（重なって遅れが積もらない）。fps は上限 */
  function pollImage(img, url, fps, onState) {
    let stop = false;
    let last = 0;
    function next() {
      if (stop) return;
      const wait = Math.max(0, 1000 / fps - (performance.now() - last));
      setTimeout(() => {
        if (stop) return;
        if (document.hidden) { setTimeout(next, 500); return; }
        last = performance.now();
        const u = (typeof url === "function" ? url() : url);
        img.src = u + (u.includes("?") ? "&" : "?") + "t=" + Date.now();
      }, wait);
    }
    img.addEventListener("load", () => { if (onState) onState(true); next(); });
    img.addEventListener("error", () => { if (onState) onState(false); setTimeout(next, 1000); });
    next();
    return () => { stop = true; };
  }

  window.SnooController = { create, nextSeq, postJSON, pollImage };
})();
