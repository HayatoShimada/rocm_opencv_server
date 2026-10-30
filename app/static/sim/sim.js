/* /sim：シミュレーターの猫を手で動かし、車（地図のサーバーと同じ決め方）の反応を見る
   （scripts/snoo_sim_live.py）。入力と送る周りは controller.js（/drive と同じ）。 */
(function () {
  "use strict";
  const C = window.SnooController;
  const $ = (id) => document.getElementById(id);
  // A：車のマッピングを始める、B：車を止める。猫の姿勢は X・Y・LB・RB・START と数字のキー
  const PAD_POSTURE = { x: "sit", rb: "lie", lb: "sleep", y: "alert", start: "walk",
    1: "sit", 2: "lie", 3: "sleep", 4: "alert", 5: "walk" };
  const NAMES = { walk: "歩く", sit: "座る", lie: "伏せる", sleep: "寝る", alert: "見る",
    run: "走る", stand: "立つ", groom: "毛づくろい", sniff: "嗅ぐ" };

  function setPill(id, cls, text) {
    const p = $(id);
    p.className = "pill " + (cls || "");
    p.querySelector("span").textContent = text;
  }

  // 左のスティック：上が前、右へ倒すと右回り（turn は左回りが +）。右のスティックの左右でも回れる
  function cat(s, posture) {
    const body = { forward: s.ly, turn: -(Math.abs(s.lx) >= Math.abs(s.rx) ? s.lx : s.rx) };
    if (posture) body.posture = posture;
    return C.postJSON("/sim/cat", body);
  }

  const ctl = C.create({
    sticks: $("sticks"),
    labels: ["猫を歩かせる", "回る"],
    keepalive_s: 1.0,
    send: (s) => cat(s),
    stop: () => cat({ lx: 0, ly: 0, rx: 0, ry: 0 }),
    onButton: (name) => {
      if (name === "a") robot(true);
      else if (name === "b") robot(false);
      else if (PAD_POSTURE[name]) setPosture(PAD_POSTURE[name]);
    },
  });
  ctl.keepalive = false;
  ctl.onInput((s, o) => {
    $("inputs").textContent = `入力 ${o.source || "—"}  前 ${s.ly.toFixed(2)}  回る ${(-s.lx).toFixed(2)}`
      + (o.pad ? "  ゲームパッド" : "");
  });

  async function robot(mapping) {
    const r = await C.postJSON("/sim/robot", { mapping });
    const m = $("mapping");
    if (m && r && !r.error) m.textContent = r.mapping ? "マッピング中（B で止める）" : "待機（A でマッピング）";
  }
  const mb = $("mapping");
  if (mb) mb.addEventListener("click", () => robot(mb.textContent.startsWith("待機")));

  function setPosture(p) { cat({ lx: 0, ly: 0, rx: 0, ry: 0 }, p); }
  document.querySelectorAll(".postures button").forEach((b) => {
    b.addEventListener("click", () => setPosture(b.dataset.p));
  });
  $("hold").addEventListener("click", () => {
    ctl.keepalive = !ctl.keepalive;
    $("hold").textContent = "猫を持ち続ける：" + (ctl.keepalive ? "入" : "切");
    $("hold").classList.toggle("on", ctl.keepalive);
  });

  function fmt(v, unit, d) {
    return v === null || v === undefined ? "—" : Number(v).toFixed(d === undefined ? 1 : d) + (unit || "");
  }

  function show(st) {
    const c = st.cat;
    if (c.controlled) setPill("pill-cat", "ok", `猫：手で（${NAMES[c.posture] || c.posture}）`);
    else setPill("pill-cat", "", `猫：自分で（${NAMES[c.posture] || c.posture}）`);
    if (st.error) setPill("pill-run", "bad", "止まった：" + st.error);
    else if (!st.running) setPill("pill-run", "warn", "回っていない");
    else setPill("pill-run", "ok", `${st.room_name}  ${st.t.toFixed(0)} 秒`);
    $("phase").textContent = st.phase_text;
    $("action").textContent = "次の 1 歩：" + st.action_text;
    $("reaction").textContent = st.reaction.text;
    document.querySelectorAll(".postures button").forEach((b) => {
      b.classList.toggle("on", b.dataset.p === c.posture);
    });
    const rows = [
      ["ステーション", st.stations],
      ["走った", fmt(st.distance_m, " m", 2)],
      ["床と分かった", `${fmt(st.covered_m2, " m²", 1)}（${(st.coverage * 100).toFixed(0)} %）`],
      ["ぶつかった", `${st.bumps}（猫に ${st.bumps_snoo}）`],
      ["止まった・戻った", `${st.stops} / ${st.reverses}`],
      ["見守った", st.reaction.watches],
      ["snoo まで", st.reaction.snoo_dist_cm == null ? "見えない" : fmt(st.reaction.snoo_dist_cm, " cm", 0)],
      ["clear・超音波", `${fmt(st.clear_cm, " cm", 0)} / ${fmt(st.sonar_cm, " cm", 0)}`],
      ["首", `${fmt(st.robot.pan_deg, "°", 0)} / ${fmt(st.robot.tilt_deg, "°", 0)}`],
      ["猫の指令", c.since_command_s == null ? "まだ" : `${fmt(c.since_command_s, " 秒前", 1)}${c.blocked ? "（ぶつかる）" : ""}`],
    ];
    $("kv").innerHTML = "";
    for (const [k, v] of rows) {
      const dt = document.createElement("dt");
      dt.textContent = k;
      const dd = document.createElement("dd");
      dd.textContent = String(v);
      $("kv").append(dt, dd);
    }
  }

  async function pollState() {
    for (;;) {
      if (!document.hidden) {
        try {
          const r = await fetch("/sim/state", { cache: "no-store" });
          if (r.ok) show(await r.json());
        } catch (e) { setPill("pill-run", "bad", "つながらない"); }
      }
      await new Promise((res) => setTimeout(res, 500));
    }
  }

  async function loadRooms() {
    const r = await fetch("/sim/rooms", { cache: "no-store" });
    if (!r.ok) return;
    const j = await r.json();
    const sel = $("room");
    sel.innerHTML = "";
    for (const x of j.rooms) {
      const o = document.createElement("option");
      o.value = x.room;
      o.textContent = x.label;
      sel.appendChild(o);
    }
    if (![...sel.options].some((o) => o.value === j.now)) {
      const o = document.createElement("option");
      o.value = j.now;
      o.textContent = j.now;
      sel.appendChild(o);
    }
    sel.value = j.now;
    $("seed").value = j.seed;
  }

  $("reset").addEventListener("click", async () => {
    $("reset").disabled = true;
    const r = await C.postJSON("/sim/reset", { room: $("room").value, seed: Number($("seed").value) || 0 });
    $("reset").disabled = false;
    if (r && r.error) alert("作り直せなかった：" + r.detail);
  });

  C.pollImage($("fpv"), "/sim/frame.jpg?view=fpv", 8, (ok) => { $("fpv-empty").hidden = ok; });
  C.pollImage($("top"), "/sim/frame.jpg?view=top", 4);
  C.pollImage($("v3d"), "/sim/frame.jpg?view=3d", 5);
  loadRooms();
  pollState();
})();
