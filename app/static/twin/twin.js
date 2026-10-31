// snoo ツインの画面。地図のサーバーの JSON（app/snoo_twin.py）を読み直して描く。
// - セッションの様子：1.5 秒ごと（ステーションは 20 秒ほどに 1 つなので、これで遅れない）
// - 検出サーバーの今のフレーム：1 秒ごと（向こうも 1 秒に 1 回ほどしか書かない）
// - セッションの一覧：5 秒ごと（「新しいセッションを追う」なら、増えたら移る）
"use strict";

const $ = (id) => document.getElementById(id);
const COLORS = {
  free: [201, 209, 217], occ: [229, 83, 75], high: [83, 155, 245], front: [242, 184, 75],
  unk: [29, 34, 40], st: "#b392f0", fwd: "#3fb950", rev: "#f0883e", nbv: "#e3e7eb",
  cam: "57,197,207", sonar: "247,120,186", car: "#3fb950", slam: "#58a6ff",
};
// 今のフレームがこれより古ければ、車は止まっている（検出サーバーに届いていない）とする
const LIVE_STALE_S = 5;

const S = {
  sessions: [], session: null, follow: true,
  state: null, stateKey: "", gridVer: null, grid: null,
  live: null, sender: null, liveSeq: null,
  view: { cx: 0, cy: 0, scale: 100, auto: true },
  layers: { frontier: true, high: true, conf: false, landmark: false, fov: true, sonar: true, plan: true, step: true, slam: true },
  thumbStation: null, thumbAuto: true, lastOk: 0,
};

// --- 読む ---

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

function setStatus(ok, text) {
  const el = $("status");
  el.className = "status " + (ok ? "ok" : "err");
  el.querySelector("span").textContent = text;
}

async function pollSessions() {
  try {
    const { sessions } = await getJSON("/twin/api/sessions");
    S.sessions = sessions;
    const sel = $("session");
    const keep = S.session;
    sel.innerHTML = "";
    for (const s of sessions) {
      const o = document.createElement("option");
      o.value = s.id;
      o.textContent = `${s.id}（${s.stations} か所）`;
      sel.appendChild(o);
    }
    let next = keep;
    if (sessions.length && (S.follow || !sessions.some((s) => s.id === keep))) next = sessions[0].id;
    if (next) sel.value = next;
    if (next !== keep) selectSession(next);
    if (!sessions.length) setStatus(false, "セッションがまだ無い");
  } catch (e) {
    setStatus(false, "サーバーに届かない");
  }
}

function selectSession(id) {
  S.session = id;
  $("session").value = id;
  S.state = null; S.stateKey = ""; S.gridVer = null; S.grid = null;
  S.view.auto = true; S.thumbAuto = true; S.thumbStation = null;
  pollState();
}

let stateBusy = false;
async function pollState() {
  const id = S.session;
  if (!id || stateBusy) return; // 前の読み込み（地図の画像）が終わっていなければ待たない
  stateBusy = true;
  try {
    const st = await getJSON(`/twin/api/${encodeURIComponent(id)}/state`);
    if (id !== S.session) return;
    S.lastOk = Date.now();
    const key = JSON.stringify([st.updated, st.grid && st.grid.version, st.stations.length]);
    const ver = st.grid ? `${st.grid.version}` : null;
    if (ver !== S.gridVer) {
      S.gridVer = ver;
      S.grid = st.grid ? await loadGrid(id, st.grid) : null;
    }
    S.state = st;
    if (key !== S.stateKey) {
      S.stateKey = key;
      renderPanel();
      renderThumbs();
      update3D();
    }
    drawMap();
    const ago = Math.max(0, st.now - st.updated);
    setStatus(true, `最後の更新 ${clock(st.updated)}（${ago < 90 ? ago.toFixed(0) + " 秒" : (ago / 60).toFixed(0) + " 分"}前）`);
  } catch (e) {
    setStatus(false, "読めない: " + e.message);
  } finally {
    stateBusy = false;
  }
}

async function pollLive() {
  try {
    S.live = await getJSON("/twin/api/live");
  } catch (e) {
    S.live = null;
  }
  renderLive();
  drawMap();
  update3D(true);
}

function loadImage(url) {
  return new Promise((ok, ng) => {
    const im = new Image();
    im.onload = () => ok(im);
    im.onerror = () => ng(new Error(url));
    im.src = url;
  });
}

// grid.png・high.png を読んで、色を塗った絵と、3D に使うセルの一覧を作る
async function loadGrid(id, info) {
  const v = encodeURIComponent(info.version);
  const base = `/twin/api/${encodeURIComponent(id)}/map`;
  const [gim, him, cim, tim] = await Promise.all([
    loadImage(`${base}/grid.png?v=${v}`),
    info.high ? loadImage(`${base}/high.png?v=${v}`).catch(() => null) : null,
    info.confidence ? loadImage(`${base}/confidence.png?v=${v}`).catch(() => null) : null,
    info.texture ? loadImage(`${base}/texture.png?v=${v}`).catch(() => null) : null,
  ]);
  const w = gim.width, h = gim.height;
  const read = (im) => {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const x = c.getContext("2d");
    x.drawImage(im, 0, 0);
    return x.getImageData(0, 0, w, h).data;
  };
  const g = read(gim), hi = him ? read(him) : null;
  // grid.png は 255 = 床・0 = 障害物・128 = 見ていない（app/snoo_mapper.py の PNG_VALUE）
  const state = new Uint8Array(w * h); // 0 見ていない・1 床・2 障害物
  for (let i = 0; i < w * h; i++) {
    const p = g[i * 4];
    state[i] = p >= 192 ? 1 : p <= 64 ? 2 : 0;
  }
  const base_ = document.createElement("canvas"); // 床・障害物・見ていない
  const over = document.createElement("canvas"); // 境目と上の層（切り替えられるよう分ける）
  for (const c of [base_, over]) { c.width = w; c.height = h; }
  const bd = base_.getContext("2d").createImageData(w, h);
  const fr = new Uint8Array(w * h), hiMask = new Uint8Array(w * h);
  const cells = { free: [], occ: [], frontier: [], high: [] };
  const res = info.resolution_m, [ox, oy] = info.origin;
  const a = info.origin_pose, ay = (a.yaw_deg * Math.PI) / 180, ca = Math.cos(ay), sa = Math.sin(ay);
  const toSession = (col, row) => {
    const xw = ox + (col + 0.5) * res, yw = oy + (h - row - 0.5) * res;
    return [a.x + ca * xw - sa * yw, a.y + sa * xw + ca * yw];
  };
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const i = r * w + c, s = state[i];
      const col = s === 1 ? COLORS.free : s === 2 ? COLORS.occ : COLORS.unk;
      bd.data.set([col[0], col[1], col[2], s === 0 ? 0 : 255], i * 4);
      const isHigh = hi && hi[i * 4] > 127;
      if (isHigh) hiMask[i] = 1;
      if (s === 1) {
        const unk = (rr, cc) => rr >= 0 && rr < h && cc >= 0 && cc < w && state[rr * w + cc] === 0;
        if (unk(r - 1, c) || unk(r + 1, c) || unk(r, c - 1) || unk(r, c + 1)) fr[i] = 1;
      }
      if (s !== 0 || isHigh) {
        x0 = Math.min(x0, c); x1 = Math.max(x1, c); y0 = Math.min(y0, r); y1 = Math.max(y1, r);
        const p = toSession(c, r);
        if (fr[i]) cells.frontier.push(p);
        else if (s === 1) cells.free.push(p);
        else if (s === 2) cells.occ.push(p);
        if (isHigh) cells.high.push(p);
      }
    }
  }
  base_.getContext("2d").putImageData(bd, 0, 0);
  const frc = document.createElement("canvas"), hic = document.createElement("canvas");
  for (const [cv, mask, col, alpha] of [[frc, fr, COLORS.front, 255], [hic, hiMask, COLORS.high, 170]]) {
    cv.width = w; cv.height = h;
    const x = cv.getContext("2d"), d = x.createImageData(w, h);
    for (let i = 0; i < w * h; i++) if (mask[i]) d.data.set([col[0], col[1], col[2], alpha], i * 4);
    x.putImageData(d, 0, 0);
  }
  // 確信度（低いほど赤く濃く。見ていない所は塗らない）と目印（手ざわりが 0.35 以上のセル）
  const confc = document.createElement("canvas"), lmc = document.createElement("canvas");
  for (const cv of [confc, lmc]) { cv.width = w; cv.height = h; }
  if (cim) {
    const cd = read(cim), x = confc.getContext("2d"), d = x.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const p = cd[i * 4];
      if (p > 0) d.data.set([230, 60, 60, Math.round(200 * (1 - (p - 1) / 254))], i * 4);
    }
    x.putImageData(d, 0, 0);
  }
  if (tim) {
    const td = read(tim), x = lmc.getContext("2d"), d = x.createImageData(w, h);
    for (let i = 0; i < w * h; i++) if (td[i * 4] >= 90) d.data.set([200, 60, 220, 230], i * 4);
    x.putImageData(d, 0, 0);
  }
  // 見えている所の外枠（セッションの座標）。全体を入れる大きさを決めるのに使う
  const corners = x1 < 0 ? [] : [[x0, y0], [x1, y0], [x0, y1], [x1, y1]].map(([c, r]) => toSession(c, r));
  return { info, w, h, base: base_, frontier: frc, high: hic, conf: confc, landmark: lmc, cells, corners, frontierCount: cells.frontier.length };
}

// --- 座標 ---

const rad = (d) => (d * Math.PI) / 180;
function poseOf(p) { return p ? { x: p.x, y: p.y, yaw: rad(p.yaw_deg) } : null; }

// いまの車の位置：位置の分かっている最後のステーション（走っている途中の位置は分からない）
function currentPose(st) {
  for (let i = st.stations.length - 1; i >= 0; i--) if (st.stations[i].pose) return { pose: poseOf(st.stations[i].pose), id: st.stations[i].id };
  if (st.reply && st.reply.pose) return { pose: poseOf(st.reply.pose), id: null };
  return null;
}

function freshLive() {
  const L = S.live && S.live.senders;
  if (!L || !L.length) return null;
  const s = (S.sender && L.find((x) => x.sender === S.sender)) || L[0];
  return s;
}

function panDeg() {
  const l = freshLive();
  if (l && l.age_s != null && l.age_s < LIVE_STALE_S && l.cmd) return l.cmd.pan_deg || 0;
  return 0;
}

function bounds(st) {
  const pts = [];
  if (S.grid) pts.push(...S.grid.corners);
  for (const s of st.stations) if (s.pose) pts.push([s.pose.x, s.pose.y]);
  if (st.plan && st.plan.target) pts.push([st.plan.target.x, st.plan.target.y]);
  if (st.step) for (const g of st.step.segments) pts.push(...g.path);
  if (!pts.length) pts.push([0, 0]);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const m = 0.35;
  return [Math.min(...xs) - m, Math.min(...ys) - m, Math.max(...xs) + m, Math.max(...ys) + m];
}

// --- 上から見た地図 ---

const mapCanvas = $("map");
const mctx = mapCanvas.getContext("2d");

function drawMap() {
  const c = mapCanvas, dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  if (!W || !H) return;
  if (c.width !== Math.round(W * dpr)) c.width = Math.round(W * dpr);
  if (c.height !== Math.round(H * dpr)) c.height = Math.round(H * dpr);
  const ctx = mctx;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0b0d10";
  ctx.fillRect(0, 0, W, H);
  const st = S.state;
  if (!st) return;
  const V = S.view;
  if (V.auto) {
    const b = bounds(st);
    V.cx = (b[0] + b[2]) / 2; V.cy = (b[1] + b[3]) / 2;
    V.scale = Math.min(W / (b[2] - b[0]), H / (b[3] - b[1]));
  }
  const X = (x) => W / 2 + (x - V.cx) * V.scale;
  const Y = (y) => H / 2 - (y - V.cy) * V.scale;
  const P = (p) => [X(p[0]), Y(p[1])];

  // 1 m ごとの線（薄く）
  ctx.strokeStyle = "rgba(255,255,255,.05)";
  ctx.lineWidth = 1;
  const xa = V.cx - W / 2 / V.scale, xb = V.cx + W / 2 / V.scale;
  const ya = V.cy - H / 2 / V.scale, yb = V.cy + H / 2 / V.scale;
  for (let x = Math.ceil(xa); x <= xb; x++) { ctx.beginPath(); ctx.moveTo(X(x), 0); ctx.lineTo(X(x), H); ctx.stroke(); }
  for (let y = Math.ceil(ya); y <= yb; y++) { ctx.beginPath(); ctx.moveTo(0, Y(y)); ctx.lineTo(W, Y(y)); ctx.stroke(); }

  // 格子：画素 (列, 行) → 窓の座標 → セッションの座標 → 画面、を 1 つの変換にして貼る
  const G = S.grid;
  if (G) {
    const i = G.info, res = i.resolution_m, [ox, oy] = i.origin, a = i.origin_pose;
    const ay = rad(a.yaw_deg), ca = Math.cos(ay), sa = Math.sin(ay), s = V.scale;
    const top = oy + G.h * res;
    const xs0 = a.x + ca * ox - sa * top, ys0 = a.y + sa * ox + ca * top;
    ctx.save();
    ctx.setTransform(dpr * s * ca * res, -dpr * s * sa * res, dpr * s * sa * res, dpr * s * ca * res,
      dpr * X(xs0), dpr * Y(ys0));
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(G.base, 0, 0);
    if (S.layers.frontier) ctx.drawImage(G.frontier, 0, 0);
    if (S.layers.high) ctx.drawImage(G.high, 0, 0);
    if (S.layers.conf && G.conf) ctx.drawImage(G.conf, 0, 0);
    if (S.layers.landmark && G.landmark) ctx.drawImage(G.landmark, 0, 0);
    ctx.restore();
  }

  const cur = currentPose(st);
  const car = st.car;

  // カメラの画角（54° + パン）と超音波（±15°、読んだ距離まで）
  if (cur) {
    const p = cur.pose;
    if (S.layers.fov) {
      const look = p.yaw - rad(panDeg()), half = rad(car.hfov_deg / 2), L = 1.2;
      const g = ctx.createRadialGradient(X(p.x), Y(p.y), 0, X(p.x), Y(p.y), L * V.scale);
      g.addColorStop(0, `rgba(${COLORS.cam},.32)`); g.addColorStop(1, `rgba(${COLORS.cam},0)`);
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.moveTo(X(p.x), Y(p.y));
      ctx.arc(X(p.x), Y(p.y), L * V.scale, -(look + half), -(look - half));
      ctx.closePath(); ctx.fill();
      ctx.strokeStyle = `rgba(${COLORS.cam},.7)`; ctx.lineWidth = 1;
      for (const a of [look - half, look + half]) {
        ctx.beginPath(); ctx.moveTo(X(p.x), Y(p.y));
        ctx.lineTo(X(p.x + 0.5 * Math.cos(a)), Y(p.y + 0.5 * Math.sin(a))); ctx.stroke();
      }
    }
    const last = st.stations.length ? st.stations[st.stations.length - 1] : null;
    const us = last && last.ultrasonic_cm;
    if (S.layers.sonar && us != null) {
      const ahead = car.front_m - car.cam_x_m;
      const sx = p.x + ahead * Math.cos(p.yaw), sy = p.y + ahead * Math.sin(p.yaw);
      const r = Math.min(us / 100, car.sonar_range_m) * V.scale, half = rad(car.sonar_half_deg);
      ctx.fillStyle = `rgba(${COLORS.sonar},.16)`;
      ctx.beginPath(); ctx.moveTo(X(sx), Y(sy));
      ctx.arc(X(sx), Y(sy), r, -(p.yaw + half), -(p.yaw - half)); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = `rgba(${COLORS.sonar},.95)`; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X(sx), Y(sy), r, -(p.yaw + half), -(p.yaw - half)); ctx.stroke();
      const m = p.yaw;
      ctx.fillStyle = `rgb(${COLORS.sonar})`; ctx.font = "11px ui-monospace, monospace";
      ctx.fillText(`${us.toFixed(0)} cm`, X(sx + (us / 100) * Math.cos(m)) + 4, Y(sy + (us / 100) * Math.sin(m)) - 4);
    }
  }

  // 目標までの Dubins の道と目標
  if (S.layers.plan && st.plan) {
    if (st.plan.path.length > 1) {
      ctx.setLineDash([6, 5]); ctx.strokeStyle = COLORS.nbv; ctx.lineWidth = 1.5;
      ctx.beginPath(); st.plan.path.forEach((q, k) => (k ? ctx.lineTo(...P(q)) : ctx.moveTo(...P(q))));
      ctx.stroke(); ctx.setLineDash([]);
    }
    const t = poseOf(st.plan.target);
    if (t) {
      ctx.strokeStyle = COLORS.front; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X(t.x), Y(t.y), 9, 0, 7); ctx.stroke();
      arrow(ctx, X(t.x), Y(t.y), X(t.x + 0.22 * Math.cos(t.yaw)), Y(t.y + 0.22 * Math.sin(t.yaw)), COLORS.front, 2);
      ctx.fillStyle = COLORS.front; ctx.font = "11px system-ui, sans-serif";
      ctx.fillText("目標", X(t.x) + 11, Y(t.y) + 14);
    }
  }

  // ステーション（位置・番号・向き）
  const win = new Set((st.window && st.window.stations) || []);
  for (const s of st.stations) {
    const p = poseOf(s.pose);
    if (!p) continue;
    const u = X(p.x), v = Y(p.y), inWin = win.has(s.id);
    arrow(ctx, u, v, X(p.x + 0.16 * Math.cos(p.yaw)), Y(p.y + 0.16 * Math.sin(p.yaw)), COLORS.st, 1.5);
    ctx.beginPath(); ctx.arc(u, v, 5, 0, 7);
    ctx.fillStyle = inWin ? COLORS.st : "#0b0d10"; ctx.fill();
    ctx.strokeStyle = COLORS.st; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = COLORS.st; ctx.font = "bold 11px ui-monospace, monospace";
    ctx.fillText(String(s.id), u - 14, v - 7);
  }

  // 走りながらの SLAM の道筋と今の位置（見失っている間は点線）
  const sl = st.slam;
  if (S.layers.slam && sl && sl.trail && sl.trail.length > 1) {
    ctx.strokeStyle = COLORS.slam; ctx.lineWidth = 1.5;
    if (sl.state !== "ok") ctx.setLineDash([4, 4]);
    ctx.beginPath(); sl.trail.forEach((q, k) => (k ? ctx.lineTo(...P(q)) : ctx.moveTo(...P(q)))); ctx.stroke();
    ctx.setLineDash([]);
    if (sl.pose && sl.pose.x != null) {
      const yaw = sl.pose.yaw != null ? sl.pose.yaw : 0;
      const u = X(sl.pose.x), v = Y(sl.pose.y);
      arrow(ctx, u, v, X(sl.pose.x + 0.14 * Math.cos(yaw)), Y(sl.pose.y + 0.14 * Math.sin(yaw)), COLORS.slam, 2);
      ctx.fillStyle = COLORS.slam; ctx.beginPath(); ctx.arc(u, v, 4, 0, 7); ctx.fill();
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillText(`SLAM ${sl.state || ""}`, u + 7, v - 7);
    }
  }

  // 次の 1 歩（区間ごと。前へは緑、後ろへは橙）と、走り終えたときの車
  if (S.layers.step && st.step) {
    const col = st.step.reverse ? COLORS.rev : COLORS.fwd;
    st.step.segments.forEach((g, k) => {
      ctx.strokeStyle = col; ctx.lineWidth = 3.5; ctx.globalAlpha = k % 2 ? 0.65 : 1;
      ctx.beginPath(); g.path.forEach((q, j) => (j ? ctx.lineTo(...P(q)) : ctx.moveTo(...P(q)))); ctx.stroke();
      ctx.globalAlpha = 1;
      const e = g.path[g.path.length - 1];
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(...P(e), 3, 0, 7); ctx.fill();
      const mid = g.path[Math.floor(g.path.length / 2)];
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillText(`${k + 1}`, X(mid[0]) + 6, Y(mid[1]) + 4);
    });
    footprint(ctx, X, Y, poseOf(st.step.end), car, col, true);
  }

  if (cur) footprint(ctx, X, Y, cur.pose, car, COLORS.car, false);

  scaleBar(ctx, V.scale, W, H);
  // 上が +y（左）、右が +x（最初の車の向き）
  ctx.fillStyle = "rgba(255,255,255,.45)"; ctx.font = "11px ui-monospace, monospace";
  ctx.fillText("+x →", W - 44, H - 10);
  ctx.fillText("↑ +y", W - 44, H - 24);
}

function arrow(ctx, x0, y0, x1, y1, color, width) {
  const a = Math.atan2(y1 - y0, x1 - x0), k = 6;
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - k * Math.cos(a - 0.45), y1 - k * Math.sin(a - 0.45));
  ctx.lineTo(x1 - k * Math.cos(a + 0.45), y1 - k * Math.sin(a + 0.45)); ctx.closePath(); ctx.fill();
}

// 車の外形（実寸）。pose はカメラの真下。後ろの車軸から前後の端までを描く
function footprint(ctx, X, Y, pose, car, color, ghost) {
  if (!pose) return;
  const c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
  const rx = pose.x - car.cam_x_m * c, ry = pose.y - car.cam_x_m * s, w = car.width_m / 2;
  const at = (a, b) => [X(rx + c * a - s * b), Y(ry + s * a + c * b)];
  const pts = [at(car.rear_m, -w), at(car.front_m, -w), at(car.front_m, w), at(car.rear_m, w)];
  ctx.beginPath(); pts.forEach((p, k) => (k ? ctx.lineTo(...p) : ctx.moveTo(...p))); ctx.closePath();
  ctx.setLineDash(ghost ? [4, 3] : []);
  ctx.fillStyle = ghost ? "rgba(0,0,0,0)" : "rgba(63,185,80,.22)";
  ctx.fill();
  ctx.strokeStyle = color; ctx.lineWidth = ghost ? 1.2 : 2; ctx.stroke(); ctx.setLineDash([]);
  if (!ghost) {
    // 前の端（バンパー）を太く、カメラの位置に点
    ctx.beginPath(); ctx.moveTo(...pts[1]); ctx.lineTo(...pts[2]); ctx.lineWidth = 3.5; ctx.stroke();
    ctx.fillStyle = `rgb(${COLORS.cam})`; ctx.beginPath(); ctx.arc(X(pose.x), Y(pose.y), 3, 0, 7); ctx.fill();
  }
}

function scaleBar(ctx, scale, W, H) {
  const want = 120 / scale;
  const nice = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10].find((v) => v >= want * 0.6) || 10;
  const L = nice * scale, x = 14, y = H - 16;
  ctx.fillStyle = "rgba(11,13,16,.7)"; ctx.fillRect(x - 6, y - 18, L + 60, 28);
  ctx.strokeStyle = "#d5dbe2"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x, y - 5); ctx.lineTo(x, y); ctx.lineTo(x + L, y); ctx.lineTo(x + L, y - 5); ctx.stroke();
  ctx.fillStyle = "#d5dbe2"; ctx.font = "11px ui-monospace, monospace";
  ctx.fillText(nice >= 1 ? `${nice} m` : `${Math.round(nice * 100)} cm`, x + L + 6, y);
}

// 地図を動かす（ドラッグで動かす・ホイールで寄る・ダブルクリックで全体）
(function mapMouse() {
  let drag = null;
  mapCanvas.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, cx: S.view.cx, cy: S.view.cy };
    mapCanvas.setPointerCapture(e.pointerId);
  });
  mapCanvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    S.view.auto = false;
    S.view.cx = drag.cx - (e.clientX - drag.x) / S.view.scale;
    S.view.cy = drag.cy + (e.clientY - drag.y) / S.view.scale;
    drawMap();
  });
  mapCanvas.addEventListener("pointerup", () => (drag = null));
  mapCanvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = mapCanvas.getBoundingClientRect(), V = S.view;
    const mx = e.clientX - r.left - r.width / 2, my = e.clientY - r.top - r.height / 2;
    const wx = V.cx + mx / V.scale, wy = V.cy - my / V.scale;
    V.auto = false;
    V.scale = Math.min(2000, Math.max(10, V.scale * Math.exp(-e.deltaY * 0.0015)));
    V.cx = wx - mx / V.scale; V.cy = wy + my / V.scale;
    drawMap();
  }, { passive: false });
  const fit = () => { S.view.auto = true; drawMap(); };
  mapCanvas.addEventListener("dblclick", fit);
  $("fit").addEventListener("click", fit);
  new ResizeObserver(drawMap).observe(mapCanvas);
})();

// --- 3D ---

const view3d = new View3D($("view3d"));

function update3D(liveOnly = false) {
  const st = S.state;
  if (!st) return;
  if (liveOnly && view3d.scene && view3d.scene.car) {
    view3d.scene.car.panDeg = panDeg();
    view3d.redraw();
    return;
  }
  const G = S.grid, cur = currentPose(st);
  const lines = [];
  if (st.plan && st.plan.path.length > 1) lines.push({ pts: st.plan.path, color: "rgba(227,231,235,.8)", width: 1.5, dash: [5, 4] });
  if (st.step) {
    const col = st.step.reverse ? COLORS.rev : COLORS.fwd;
    for (const g of st.step.segments) lines.push({ pts: g.path, color: col, width: 3, z: 0.006 });
  }
  const b = bounds(st);
  view3d.setScene({
    bounds: b,
    cells: G ? G.cells : { free: [], occ: [], frontier: [], high: [] },
    cellSize: G ? G.info.resolution_m : 0.05,
    cellYaw: G ? rad(G.info.origin_pose.yaw_deg) : 0,
    obstacleZ: 0.10,
    highZ: (G && G.info.high_min_m) || 0.15,
    cameraZ: 0.08,
    car: cur ? { pose: cur.pose, dims: st.car, panDeg: panDeg() } : null,
    stations: st.stations.filter((s) => s.pose).map((s) => ({ id: s.id, x: s.pose.x, y: s.pose.y })),
    lines,
  });
}

// --- 右の欄 ---

function fmt(v, digits = 1, unit = "") {
  if (v == null || Number.isNaN(v)) return "—";
  const t = Number(v).toFixed(digits);
  return `${Number(t) === 0 ? (0).toFixed(digits) : t}${unit}`; // -0 を 0 にする
}
function clock(t) {
  return new Date(t * 1000).toLocaleTimeString("ja-JP", { hour12: false });
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
function kv(rows) {
  return rows.map(([k, v, cls]) => `<dt>${esc(k)}</dt><dd class="${cls || ""}" title="${esc(v)}">${esc(v)}</dd>`).join("");
}

const REASONS = {
  unreachable: "見えている床を通って行ける所が無い",
  no_frontier: "見ていない所との境目が無い（調べ終えた）",
  too_little_floor: "床が少なすぎる",
  bad_registration: "位置合わせが崩れた",
  low_battery: "電池が少ない",
  step_limit: "ステーションの数の上限",
};

function renderPanel() {
  const st = S.state, r = st.reply || {};
  const badge = $("reply-badge");
  const kind = r.action === "move" ? (r.reverse ? "reverse" : "move") : r.action ? "done" : "";
  badge.className = "badge " + kind;
  badge.textContent = kind === "reverse" ? "move（後ろへ）" : r.action || "まだ無い";
  let html = "";
  if (r.action === "done") {
    const key = String(r.reason || "").split(":")[0];
    html += `<div class="reason">${esc(r.reason || "")}：${esc(REASONS[key] || "")}</div>`;
  }
  if (r.action === "move") {
    const segs = r.segments && r.segments.length ? r.segments : [{ steer_deg: r.steer_deg, distance_cm: r.distance_cm }];
    const col = r.reverse ? COLORS.rev : COLORS.fwd;
    html += `<table class="segs"><tr><th>区間</th><th>向き</th><th>ハンドル</th><th>長さ</th></tr>`;
    segs.forEach((g, k) => {
      html += `<tr><td><span class="dot" style="background:${col}"></span>${k + 1}</td><td>${r.reverse ? "後ろ" : "前"}</td>`
        + `<td>${fmt(g.steer_deg, 1, "°")}</td><td>${fmt(g.distance_cm, 1, " cm")}</td></tr>`;
    });
    html += "</table>";
    html += `<dl class="kv" style="margin-top:6px">${kv([
      ["速さ", fmt(r.speed, 0)],
      ["合わせて", fmt(segs.reduce((a, g) => a + (g.distance_cm || 0), 0), 1, " cm")],
      ["今の位置", r.pose ? `${fmt(r.pose.x, 2)}, ${fmt(r.pose.y, 2)} m / ${fmt(r.pose.yaw_deg, 0)}°` : "—"],
      ["目標", r.target ? `${fmt(r.target.x, 2)}, ${fmt(r.target.y, 2)} m / ${fmt(r.target.yaw_deg, 0)}°` : "—"],
      ["目標で見える", st.plan && st.plan.gain_cells != null ? `${st.plan.gain_cells} セル（道 ${fmt(st.plan.length_m, 2)} m）` : "—"],
    ])}</dl>`;
  }
  const lastId = st.stations.length ? st.stations[st.stations.length - 1].id : null;
  if (st.reply_station != null) {
    const behind = lastId != null && lastId !== st.reply_station;
    html += `<div class="reason">${String(st.reply_station).padStart(3, "0")} への返事${behind
      ? `。${String(lastId).padStart(3, "0")} には地図を作らずに返した（電池・上限）か、まだ作っている` : ""}</div>`;
  }
  if (st.next && r.action && st.next.action !== r.action) {
    html += `<div class="reason">地図の判断は ${esc(st.next.action)}（${esc(st.next.reason || "")}）。行き場が無いので来た道を戻る</div>`;
  }
  $("reply").innerHTML = html || `<div class="reason">まだ返していない</div>`;

  const last = st.stations[st.stations.length - 1] || {};
  const prev = last.prev_step || {};
  const bv = last.battery_v;
  const known = st.stations.filter((s) => s.pose).length;
  $("car").innerHTML = kv([
    ["ステーション", `${st.stations.length} か所（位置が分かる ${known}）`],
    ["最後", last.id != null ? `${String(last.id).padStart(3, "0")}（${clock(last.mtime)}）` : "—"],
    // Pi が電圧を読めなかったときは 0 が来る（そのときも地図のサーバーは low_battery でやめる）
    bv != null && bv < 1 ? ["電池", `${fmt(bv, 2, " V")}（読めていない）`, "warn"]
      : ["電池", fmt(bv, 2, " V"), bv == null ? "" : bv < 6.8 ? "bad" : bv < 7.2 ? "warn" : "good"],
    ["超音波", fmt(last.ultrasonic_cm, 0, " cm"), last.ultrasonic_cm != null && last.ultrasonic_cm < 20 ? "warn" : ""],
    ["前の 1 歩", prev.steer_deg != null ? `${fmt(prev.steer_deg, 1, "°")} × ${fmt(prev.seconds, 2, " 秒")}${prev.reverse ? "（後ろ）" : ""}` : "—"],
    ["走った見積もり", fmt(prev.est_distance_cm, 1, " cm")],
    ["止めたわけ", prev.stop_reason || "—", prev.stop_reason && prev.stop_reason !== "distance" ? "warn" : ""],
    ["窓", st.window ? (st.window.stations || []).join(", ") : "—"],
    ["境目のセル", S.grid ? `${S.grid.frontierCount}` : "—"],
    ["Pi ができること", (last.supports || []).join(", ") || "—"],
    // 走りながらの SLAM：状態・座標合わせのずれ・最後のキーフレームの古さ
    st.slam
      ? ["SLAM", `${st.slam.state || "—"}・ずれ ${fmt(st.slam.rmse_m != null ? st.slam.rmse_m * 100 : null, 1, " cm")}・${fmt(st.slam.age_s, 1, " 秒前")}`,
         st.slam.state === "ok" ? "good" : "warn"]
      : ["SLAM", "—"],
  ]);

  const t = st.timing_s || {};
  const order = ["save", "wait", "vggt", "points", "map", "write"];
  const names = { save: "保存", wait: "GPU 待ち", vggt: "VGGT", points: "点", map: "間取り", write: "書き出し" };
  const total = order.reduce((a, k) => a + (t[k] || 0), 0);
  let th = "";
  for (const k of order) {
    if (t[k] == null) continue;
    th += `<div class="bar"><span class="name">${names[k]}</span><span class="track"><span class="fill ${k}" style="width:${Math.min(100, (t[k] / 20) * 100)}%"></span></span><span class="val">${t[k].toFixed(2)} s</span></div>`;
  }
  if (th) {
    th += `<div class="bar"><span class="name">合わせて</span><span class="track"><span class="fill" style="width:${Math.min(100, (total / 20) * 100)}%;background:${total > 15 ? COLORS.rev : COLORS.fwd}"></span></span><span class="val">${total.toFixed(2)} s</span></div>`;
    if (t.gpu_peak_gb != null) th += `<div class="bar"><span class="name">GPU</span><span></span><span class="val">${t.gpu_peak_gb.toFixed(1)} GB</span></div>`;
  }
  $("timing").innerHTML = th || `<div class="reason">まだ無い</div>`;
}

// --- 撮った画像 ---

function renderThumbs() {
  const st = S.state;
  const ids = st.stations.map((s) => s.id);
  if (S.thumbAuto || !ids.includes(S.thumbStation)) S.thumbStation = ids[ids.length - 1] ?? null;
  const chips = $("station-chips");
  chips.innerHTML = "";
  for (const id of ids.slice(-30)) {
    const b = document.createElement("button");
    b.textContent = String(id);
    if (id === S.thumbStation) b.className = "on";
    b.onclick = () => { S.thumbStation = id; S.thumbAuto = id === ids[ids.length - 1]; renderThumbs(); };
    chips.appendChild(b);
  }
  const box = $("thumbs");
  const s = st.stations.find((x) => x.id === S.thumbStation);
  if (!s || !s.frames.length) { box.innerHTML = `<div class="empty">まだ無い</div>`; return; }
  // 上の段（正面）から下の段、左から右（パンは右が +）の順に並べる
  const r = (v) => Math.round(v || 0);
  const frames = s.frames.slice().sort((a, b) => (r(b.tilt_deg) - r(a.tilt_deg)) || (r(a.pan_deg) - r(b.pan_deg)));
  const key = `${st.session}/${s.id}/${s.mtime}`;
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  box.innerHTML = "";
  for (const f of frames) {
    const url = `/twin/api/${encodeURIComponent(st.session)}/station/${s.id}/${encodeURIComponent(f.file)}?v=${s.mtime}`;
    const d = document.createElement("div");
    d.className = "thumb";
    const label = `pan ${fmt(f.pan_deg, 0, "°")} tilt ${fmt(f.tilt_deg, 0, "°")}`;
    d.innerHTML = `<img loading="lazy" src="${url}" alt=""><span>${esc(label)}</span>`;
    d.onclick = () => openLightbox(url, `${String(s.id).padStart(3, "0")} / ${f.file}`);
    box.appendChild(d);
  }
}

function openLightbox(url, text) {
  const lb = $("lightbox");
  lb.querySelector("img").src = url;
  lb.querySelector("p").textContent = text;
  lb.hidden = false;
}
$("lightbox").onclick = () => ($("lightbox").hidden = true);

// --- 今のカメラ ---

function renderLive() {
  const L = S.live, sel = $("sender");
  const senders = (L && L.senders) || [];
  sel.hidden = senders.length < 2;
  if (senders.length >= 2) {
    const want = senders.map((s) => s.sender).join(",");
    if (sel.dataset.v !== want) {
      sel.dataset.v = want;
      sel.innerHTML = senders.map((s) => `<option>${esc(s.sender)}</option>`).join("");
      if (S.sender) sel.value = S.sender;
    }
  }
  const l = freshLive();
  const img = $("live-img"), empty = $("live-empty"), frame = img.parentElement;
  if (!l) {
    img.removeAttribute("src"); img.hidden = true; empty.hidden = false;
    empty.textContent = L && !L.enabled ? "今のフレームを読む所が決まっていない（SNOO_MAP_LIVE_DIR）"
      : "検出サーバーの今のフレームはまだ無い（compose の SNOO_LIVE_DIR）";
    $("live-kv").innerHTML = ""; $("live-age").textContent = "";
    clearBoxes();
    return;
  }
  const stale = l.age_s == null || l.age_s > LIVE_STALE_S;
  frame.classList.toggle("stale", stale);
  $("live-age").textContent = `${l.sender}・${l.age_s == null ? "?" : l.age_s < 90 ? l.age_s.toFixed(1) + " 秒前" : clock(l.t_recv)}${stale ? "（止まっている）" : ""}`;
  empty.hidden = true; img.hidden = false;
  if (l.has_image && (S.liveSeq !== `${l.sender}/${l.seq}` || !img.src)) {
    S.liveSeq = `${l.sender}/${l.seq}`;
    img.onload = () => drawBoxes(l);
    img.src = `/twin/api/live/${encodeURIComponent(l.sender)}.jpg?seq=${l.seq}&t=${l.t_recv}`;
  } else {
    drawBoxes(l);
  }
  const best = (l.dets || [])[0];
  const c = l.cmd || {};
  const clear = l.clear_skip ? `見ない（${l.clear_skip}）` : "clear_cm" in l ? fmt(l.clear_cm, 1, " cm") : "—";
  $("live-kv").innerHTML = kv([
    ["猫", best ? fmt(best.cat, 2) : "いない", best ? "good" : ""],
    ["snoo らしさ", best ? fmt(best.snoo, 2) : "—", best && best.snoo != null && best.snoo >= 0.5 ? "good" : ""],
    ["箱の数", String((l.dets || []).length)],
    ["clear_cm", clear, l.clear_cm != null && l.clear_cm < 15 ? "warn" : ""],
    ["clear の確からしさ", fmt(l.clear_conf, 2)],
    ["見えない手前", fmt(l.clear_blind_cm, 1, " cm")],
    ["走っている", c.moving == null ? "—" : c.moving ? "はい" : "いいえ"],
    ["ハンドル", fmt(c.steer_deg, 1, "°")],
    ["速さ", fmt(c.speed, 0)],
    ["パン / チルト", c.pan_deg == null ? "—" : `${fmt(c.pan_deg, 0, "°")} / ${fmt(c.tilt_deg, 0, "°")}`],
    ["検出", fmt(l.infer_ms, 1, " ms")],
    ["seq", String(l.seq)],
  ]);
}

function clearBoxes() {
  const cv = $("live-boxes");
  cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
}

// 猫の箱を画像の上に描く（箱は幅・高さに対する比）
function drawBoxes(l) {
  const img = $("live-img"), cv = $("live-boxes");
  const w = img.clientWidth, h = img.clientHeight;
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  cv.style.width = `${w}px`; cv.style.height = `${h}px`;
  cv.style.left = `${img.offsetLeft}px`; cv.style.top = `${img.offsetTop}px`;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  for (const d of l.dets || []) {
    const [x1, y1, x2, y2] = d.box;
    const snoo = d.snoo != null && d.snoo >= 0.5;
    const col = snoo ? "#3fb950" : "#f2b84b";
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);
    const text = `cat ${d.cat.toFixed(2)}${d.snoo != null ? ` snoo ${d.snoo.toFixed(2)}` : ""}`;
    ctx.font = "11px ui-monospace, monospace";
    const tw = ctx.measureText(text).width + 6;
    ctx.fillStyle = col; ctx.fillRect(x1 * w, Math.max(0, y1 * h - 15), tw, 15);
    ctx.fillStyle = "#0b0d10"; ctx.fillText(text, x1 * w + 3, Math.max(11, y1 * h - 4));
  }
}
new ResizeObserver(() => { const l = freshLive(); if (l) drawBoxes(l); }).observe($("live-img").parentElement);

// --- 始める ---

$("session").addEventListener("change", (e) => {
  // 手で選んだら、新しいセッションへは移らない
  S.follow = e.target.value === (S.sessions[0] && S.sessions[0].id) && $("follow").checked;
  $("follow").checked = S.follow;
  selectSession(e.target.value);
});
$("follow").addEventListener("change", (e) => { S.follow = e.target.checked; if (S.follow) pollSessions(); });
$("sender").addEventListener("change", (e) => { S.sender = e.target.value; S.liveSeq = null; renderLive(); });
for (const el of document.querySelectorAll("#layers input")) {
  el.addEventListener("change", () => { S.layers[el.dataset.layer] = el.checked; drawMap(); });
}

pollSessions();
pollLive();
setInterval(pollSessions, 5000);
setInterval(pollState, 1500);
setInterval(pollLive, 1000);
