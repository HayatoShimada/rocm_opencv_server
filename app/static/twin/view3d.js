// 格子と車を斜めから見た 3D。three.js などを持ち込まず、Canvas 2D に面を遠い順に塗る
// （画家のアルゴリズム）。セルは数千までなので、これで足りる。
// 座標はセッションの座標（m。x は最初の車の向き、y は左、z は上）。
"use strict";

class View3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.scene = null;
    this.az = -2.2; // 見る向き（rad。x 軸から左回り）
    this.el = 0.75; // 見下ろす角
    this.zoom = 1.0;
    this.target = [0, 0, 0];
    this.radius = 2;
    this.pending = false;
    this._mouse(canvas);
    new ResizeObserver(() => this.redraw()).observe(canvas);
  }

  _mouse(c) {
    let drag = null;
    c.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, az: this.az, el: this.el };
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener("pointermove", (e) => {
      if (!drag) return;
      this.az = drag.az - (e.clientX - drag.x) * 0.008;
      this.el = Math.min(1.5, Math.max(0.12, drag.el + (e.clientY - drag.y) * 0.006));
      this.redraw();
    });
    c.addEventListener("pointerup", () => (drag = null));
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.zoom = Math.min(8, Math.max(0.3, this.zoom * Math.exp(-e.deltaY * 0.0015)));
      this.redraw();
    }, { passive: false });
    c.addEventListener("dblclick", () => {
      this.az = -2.2; this.el = 0.75; this.zoom = 1; this.redraw();
    });
  }

  setScene(scene) {
    this.scene = scene;
    const b = scene.bounds;
    this.target = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 0];
    this.radius = Math.max(0.8, Math.hypot(b[2] - b[0], b[3] - b[1]) / 2);
    this.redraw();
  }

  redraw() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => { this.pending = false; this._draw(); });
  }

  _camera(W, H) {
    const d = (this.radius * 1.7) / this.zoom;
    const ce = Math.cos(this.el);
    const t = this.target;
    const eye = [t[0] + d * ce * Math.cos(this.az), t[1] + d * ce * Math.sin(this.az),
      t[2] + d * Math.sin(this.el)];
    const f = norm(sub(t, eye));
    const r = norm(cross(f, [0, 0, 1]));
    const u = cross(r, f);
    const focal = Math.min(W, H) * 1.1;
    return { eye, f, r, u, focal, W, H };
  }

  _draw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== Math.round(W * dpr)) c.width = Math.round(W * dpr);
    if (c.height !== Math.round(H * dpr)) c.height = Math.round(H * dpr);
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "#151a20"); g.addColorStop(1, "#0b0d10");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
    const s = this.scene;
    if (!s) return;
    const cam = this._camera(W, H);
    const proj = (p) => {
      const v = sub(p, cam.eye);
      const z = dot(v, cam.f);
      return [W / 2 + (cam.focal * dot(v, cam.r)) / z, H / 2 - (cam.focal * dot(v, cam.u)) / z, z];
    };
    const light = norm([0.4, 0.6, 1.0]);
    const items = []; // {z, draw}

    // 面を足す。pts は 3D の点、color は [r, g, b]、alpha は透明度。裏を向く面は塗らない
    const face = (pts, color, alpha = 1, cull = true, n = null) => {
      const q = pts.map(proj);
      if (q.some((p) => p[2] < 0.05)) return;
      if (cull) {
        let area = 0;
        for (let i = 0; i < q.length; i++) {
          const a = q[i], b = q[(i + 1) % q.length];
          area += a[0] * b[1] - b[0] * a[1];
        }
        if (area > 0) return; // 画面の y は下向きなので、表は時計回り（負）になる
      }
      n = n || norm(cross(sub(pts[1], pts[0]), sub(pts[2], pts[0])));
      const k = 0.5 + 0.5 * Math.max(0, dot(n, light));
      const col = `rgba(${(color[0] * k) | 0},${(color[1] * k) | 0},${(color[2] * k) | 0},${alpha})`;
      const z = q.reduce((a, p) => a + p[2], 0) / q.length;
      items.push({ z, draw: () => {
        ctx.beginPath();
        ctx.moveTo(q[0][0], q[0][1]);
        for (let i = 1; i < q.length; i++) ctx.lineTo(q[i][0], q[i][1]);
        ctx.closePath();
        ctx.fillStyle = col;
        ctx.fill();
        if (alpha === 1) { ctx.strokeStyle = col; ctx.lineWidth = 0.6; ctx.stroke(); }
      } });
    };
    // 底が z0、上が z1 の四角い柱。corners は上から見た 4 隅（左回り）
    const box = (corners, z0, z1, color, alpha = 1) => {
      const top = corners.map(([x, y]) => [x, y, z1]);
      const bot = corners.map(([x, y]) => [x, y, z0]);
      face(top, color, alpha);
      face(bot.slice().reverse(), color, alpha);
      for (let i = 0; i < 4; i++) {
        const j = (i + 1) % 4;
        face([bot[i], bot[j], top[j], top[i]], color, alpha);
      }
    };
    const line = (pts, color, width, z = 0.004, dash = null) => {
      for (let i = 1; i < pts.length; i++) {
        const a = proj([pts[i - 1][0], pts[i - 1][1], pts[i - 1][2] ?? z]);
        const b = proj([pts[i][0], pts[i][1], pts[i][2] ?? z]);
        if (a[2] < 0.05 || b[2] < 0.05) continue;
        items.push({ z: (a[2] + b[2]) / 2 - 0.02, draw: () => {
          ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
          ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
          ctx.lineCap = "round"; ctx.stroke(); ctx.setLineDash([]);
        } });
      }
    };

    // セル（格子は窓の向きに回っているので、4 隅を回して置く）
    const h = s.cellSize / 2, cy = Math.cos(s.cellYaw), sy = Math.sin(s.cellYaw);
    const quad = (x, y, shrink = 0) => {
      const e = h - shrink;
      return [[-e, -e], [e, -e], [e, e], [-e, e]].map(([a, b]) => [x + cy * a - sy * b, y + sy * a + cy * b]);
    };
    for (const [x, y] of s.cells.free) face(quad(x, y).map(([a, b]) => [a, b, 0]), [184, 192, 201], 1, true, [0, 0, 1]);
    for (const [x, y] of s.cells.frontier) face(quad(x, y).map(([a, b]) => [a, b, 0.001]), [242, 184, 75], 1, true, [0, 0, 1]);
    for (const [x, y] of s.cells.occ) box(quad(x, y), 0, s.obstacleZ, [229, 83, 75]);
    for (const [x, y] of s.cells.high) box(quad(x, y), s.highZ, s.highZ + 0.012, [83, 155, 245], 0.55);

    for (const l of s.lines) line(l.pts, l.color, l.width, l.z, l.dash);

    // 車：外形の箱と、カメラ（小さな箱）と、カメラの向き
    if (s.car) {
      const { pose, dims, panDeg } = s.car;
      const yaw = pose.yaw, c = Math.cos(yaw), sn = Math.sin(yaw);
      const rx = pose.x - dims.cam_x_m * c, ry = pose.y - dims.cam_x_m * sn; // 後ろの車軸
      const at = (a, b) => [rx + c * a - sn * b, ry + sn * a + c * b];
      const w = dims.width_m / 2;
      box([at(dims.rear_m, -w), at(dims.front_m, -w), at(dims.front_m, w), at(dims.rear_m, w)],
        0.02, dims.height_m - 0.04, [63, 185, 80]);
      const cx = pose.x, cyy = pose.y, cz = s.cameraZ;
      const k = 0.018;
      box([[cx - k, cyy - k], [cx + k, cyy - k], [cx + k, cyy + k], [cx - k, cyy + k]].map(([a, b]) => {
        const dx = a - cx, dy = b - cyy;
        return [cx + c * dx - sn * dy, cyy + sn * dx + c * dy];
      }), cz - k, cz + k, [57, 197, 207]);
      const look = yaw - (panDeg * Math.PI) / 180; // pan は右が +
      const half = (dims.hfov_deg / 2) * Math.PI / 180, L = 0.6;
      for (const a of [look - half, look + half]) {
        line([[cx, cyy, cz], [cx + L * Math.cos(a), cyy + L * Math.sin(a), cz]], "rgba(57,197,207,.55)", 1);
      }
      line([[cx, cyy, cz], [cx + L * Math.cos(look), cyy + L * Math.sin(look), cz]], "#39c5cf", 2);
    }

    // ステーション：細い柱と番号
    for (const st of s.stations) {
      const top = proj([st.x, st.y, 0.18]), bot = proj([st.x, st.y, 0]);
      if (top[2] < 0.05) continue;
      items.push({ z: bot[2] - 0.01, draw: () => {
        ctx.beginPath(); ctx.moveTo(bot[0], bot[1]); ctx.lineTo(top[0], top[1]);
        ctx.strokeStyle = "#b392f0"; ctx.lineWidth = 1.5; ctx.stroke();
        ctx.fillStyle = "#b392f0"; ctx.font = "11px ui-monospace, monospace";
        ctx.fillText(String(st.id), top[0] + 3, top[1]);
      } });
    }

    items.sort((a, b) => b.z - a.z);
    for (const it of items) it.draw();

    // 向きの目安（右下）：x（最初の車の向き）と y
    const o = [W - 46, H - 18];
    const axis = (v, color, name) => {
      const p0 = proj(this.target), p1 = proj([this.target[0] + v[0] * this.radius * 0.2,
        this.target[1] + v[1] * this.radius * 0.2, 0]);
      const dx = p1[0] - p0[0], dy = p1[1] - p0[1], n = Math.hypot(dx, dy) || 1;
      ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(o[0] + (dx / n) * 22, o[1] + (dy / n) * 22);
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = color; ctx.font = "11px ui-monospace, monospace";
      ctx.fillText(name, o[0] + (dx / n) * 28 - 3, o[1] + (dy / n) * 28 + 4);
    };
    axis([1, 0], "#e5534b", "x");
    axis([0, 1], "#3fb950", "y");
  }
}

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function norm(a) { const n = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / n, a[1] / n, a[2] / n]; }
