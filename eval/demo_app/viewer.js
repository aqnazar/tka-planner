// The 3D view, adapted from the planner's (web/viewer.js): the same scene model (a
// femoral and a tibial group, each carrying one pose, nodes with rest transforms), the
// same Plan-mode clip preview with resected-bone caps, plus the Fusion-built parts,
// which sit in the same groups and so follow a trial pose like the planner's own parts.
//
// What is drawn is the page's choice (the Scene panel): bones, the resected bone, the
// built parts and the knee's own cutting blocks are shown, hidden and X-rayed here;
// planes, axes and landmarks follow the planner's display controls.

import * as THREE from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { parseGlb } from "./glb.js";

const MM = 1000;
const CREASE_DEG = 35;     // normals are not averaged across sharper edges (the cut faces)
export const COLOURS = {
  bone: 0xe7dfcc, resected: 0xc98f6b, femoral: 0xc4ccd6, plate: 0xa7b1bd, insert: 0xf3f6fa,
  block: 0x5aa26a, plane: 0x005aaa, axis: 0xe07b00, landmark: 0xe07b00, background: 0xf3f5f8,
};

export class Viewer {
  constructor(container) {
    this.container = container;
    this.meshCache = new Map();
    this.nodes = new Map();
    this.parts = new Map();
    this.groups = new Map();
    this.clipPlanes = new Map();
    this.clipping = true;
    this.items = {};            // Scene panel state: key -> {on, alpha}
    this.shellAsBone = false;   // the resected bone drawn as bone (the knee before the cuts)
    this.frameHooks = [];

    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.localClippingEnabled = true;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(COLOURS.background);
    this.camera = new THREE.PerspectiveCamera(28, 1, 0.005, 50);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;

    this.root = new THREE.Group();
    this.scene.add(this.root);
    for (const id of ["femoral", "tibial", "world"]) {
      const g = new THREE.Group();
      g.matrixAutoUpdate = false;
      this.groups.set(id, g);
      this.root.add(g);
    }
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa4b0, 1.9));
    this.key = new THREE.DirectionalLight(0xffffff, 1.5);
    this.scene.add(this.key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.45);
    fill.position.set(-0.7, 0.6, -0.3);
    this.scene.add(fill);

    this.resize();
    new ResizeObserver(() => this.resize()).observe(container);
    this.renderer.setAnimationLoop(() => {
      if (this.tween) this.tween();
      this.controls.update();
      this.key.position.copy(this.camera.position).add(new THREE.Vector3(0.1, 0.05, 0.25));
      this.renderer.render(this.scene, this.camera);
      for (const f of this.frameHooks) f();
    });
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // -- the planner's scene ------------------------------------------------

  async setScene(snapshot, meshUrl) {
    this.meshUrl = meshUrl;
    for (const { mesh, cap } of this.nodes.values()) {
      mesh.parent?.remove(mesh);
      cap?.parent?.remove(cap);
    }
    this.nodes.clear();
    for (const [id, m] of Object.entries(snapshot.set_poses || {})) this.setPose(id, m);
    await Promise.all(snapshot.nodes.map((n) => this.addNode(n)));
  }

  // What a node is, for drawing and for the Scene panel.
  kind(node) {
    const t = node.tags || {};
    if (t.bone) return "bone";
    if (t.cutting_block) return t.display ? "block" : "blockshell";   // the tools stay hidden
    if (t.component || t.insert) return "library";        // the planner's catalogue preview
    if (t.shell) return "shell";
    if (t.plane) return "plane";
    if (t.axis) return "axis";
    if (t.landmark) return "landmark";
    return "other";
  }

  // The Scene panel key a node answers to.
  itemOf(node) {
    const k = this.kind(node);
    if (k === "bone") return node.name;                       // Femur, Tibia
    if (k === "shell") return "resected";
    if (k === "block") return node.tags.group === "femoral" ? "femoral_block" : "tibial_block";
    if (k === "library") return "library";
    return null;
  }

  async addNode(node) {
    if (!node.mesh) return;
    const geometry = await this.geometry(this.meshUrl(node.mesh));
    const k = this.kind(node);
    const colour = k === "bone" ? COLOURS.bone : k === "shell" ? COLOURS.resected : k === "block" ? COLOURS.block
      : k === "plane" ? COLOURS.plane : k === "axis" ? COLOURS.axis
      : k === "landmark" ? new THREE.Color(...(node.colour || [0.9, 0.6, 0.05]))
      : new THREE.Color(...(node.colour || [0.6, 0.6, 0.6]));
    const alpha = k === "plane" ? 0.18 : (node.alpha ?? 1);
    const material = new THREE.MeshStandardMaterial({
      color: colour, roughness: k === "bone" || k === "shell" ? 0.9 : 0.5, metalness: 0,
      transparent: alpha < 1, opacity: alpha, depthWrite: alpha >= 1,
      side: k === "plane" || k === "axis" ? THREE.DoubleSide : THREE.FrontSide,
      // The resected bone shares its cut faces with the bone: pushed back in depth, the
      // bone's face wins instead of the two fighting in stripes.
      // A plan plane lies on its committed cut face too: pushed back the same way.
      polygonOffset: k === "shell" || k === "block" || k === "plane", polygonOffsetFactor: 2, polygonOffsetUnits: 2,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = node.name;
    mesh.matrixAutoUpdate = false;
    mesh.matrix.copy(toMatrix(node.rest));
    if (k === "landmark") mesh.renderOrder = 2;
    const group = this.groups.get(node.set) || this.groups.get("world");
    group.add(mesh);
    let cap = null;
    if (k === "bone") {
      // Clipped open, the back faces of a closed bone are what shows through the cut;
      // drawn in cut-bone colour the preview reads as solid (the planner's own trick).
      cap = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ color: COLOURS.resected, side: THREE.BackSide }));
      cap.matrixAutoUpdate = false;
      cap.matrix.copy(mesh.matrix);
      group.add(cap);
    }
    const old = this.nodes.get(node.name);
    if (old) { old.mesh.parent?.remove(old.mesh); old.cap?.parent?.remove(old.cap); }
    const entry = { mesh, cap, node };
    this.nodes.set(node.name, entry);
    this.applyLook(entry);
    this.applyVisibility(entry);
    this.applyClip(entry);
    return entry;
  }

  applyVisibility(entry) {
    const k = this.kind(entry.node), item = this.items[this.itemOf(entry.node)];
    let on = entry.node.visible;
    if (k === "blockshell") on = false;                 // the same block, kept for the cut
    else if (item) on = item.on;
    entry.mesh.visible = on;
    if (entry.cap) entry.cap.visible = on && this.clipping && this.clipPlanes.has(entry.node.name);
  }

  applyLook(entry) {
    const k = this.kind(entry.node), item = this.items[this.itemOf(entry.node)];
    if (!["bone", "shell", "block", "library"].includes(k)) return;
    const m = entry.mesh.material;
    let a = item?.alpha ?? 1;
    if (k === "shell") m.color.setHex(this.shellAsBone ? COLOURS.bone : COLOURS.resected);
    m.transparent = a < 1; m.opacity = a; m.depthWrite = a >= 1;
    m.needsUpdate = true;
    if (entry.cap) {
      entry.cap.material.transparent = a < 1; entry.cap.material.opacity = a;
      entry.cap.material.depthWrite = a >= 1; entry.cap.material.needsUpdate = true;
    }
  }

  // The Scene panel: key in Femur, Tibia, resected, femoral, plate, insert,
  // femoral_block, tibial_block, library; {on, alpha}.
  setItem(key, state) {
    this.items[key] = { ...(this.items[key] || { on: true, alpha: 1 }), ...state };
    const s = this.items[key];
    const part = this.parts.get(key);
    if (part) {
      part.mesh.visible = s.on;
      const m = part.mesh.material;
      m.opacity = s.alpha; m.depthWrite = s.alpha > 0.95; m.transparent = true;
      return;
    }
    for (const e of this.nodes.values()) {
      if (this.itemOf(e.node) === key) { this.applyLook(e); this.applyVisibility(e); }
    }
  }

  setShellAsBone(on) {
    this.shellAsBone = on;
    for (const e of this.nodes.values()) if (this.kind(e.node) === "shell") this.applyLook(e);
  }

  async applyDelta(delta) {
    for (const [name, matrix] of Object.entries(delta.poses || {})) {
      if (name === "TibialSet") this.setPose("tibial", matrix);
      else if (name === "FemoralSet") this.setPose("femoral", matrix);
      else {
        const e = this.nodes.get(name);
        if (e) { e.mesh.matrix.copy(toMatrix(matrix)); e.cap?.matrix.copy(e.mesh.matrix); }
      }
    }
    for (const [name, visible] of Object.entries(delta.visibility || {})) {
      const e = this.nodes.get(name);
      if (e) { e.node.visible = visible; this.applyVisibility(e); }
    }
    await Promise.all(Object.entries(delta.meshes || {}).map(async ([name, id]) => {
      const e = this.nodes.get(name);
      if (!e) return delta.nodes?.[name] ? this.addNode({ ...delta.nodes[name], mesh: id }) : null;
      const g = await this.geometry(this.meshUrl(id));
      e.mesh.geometry = g;
      if (e.cap) e.cap.geometry = g;
      e.node.mesh = id;
      if (delta.nodes?.[name]) {
        e.node.rest = delta.nodes[name].rest ?? e.node.rest;
        e.mesh.matrix.copy(toMatrix(e.node.rest)); e.cap?.matrix.copy(e.mesh.matrix);
      }
    }));
  }

  setPose(id, rows) {
    const g = this.groups.get(id);
    if (g) g.matrix.copy(toMatrix(rows));
  }

  setClip(clip) {
    this.clipPlanes.clear();
    for (const [bone, p] of Object.entries(clip || {})) {
      // 0.3 mm proud of the plan's plane: on a committed cut the two coincide, and a clip
      // exactly on the cut face kept and dropped its pixels in stripes, showing the cap.
      this.clipPlanes.set(bone, new THREE.Plane(new THREE.Vector3(...p.normal), p.constant_mm / MM + 0.0003));
    }
    for (const e of this.nodes.values()) { this.applyClip(e); this.applyVisibility(e); }
  }

  setClipping(on) {
    this.clipping = on;
    for (const e of this.nodes.values()) { this.applyClip(e); this.applyVisibility(e); }
  }

  applyClip(e) {
    if (this.kind(e.node) !== "bone") return;
    const plane = this.clipPlanes.get(e.node.name);
    const planes = this.clipping && plane ? [plane] : null;
    e.mesh.material.clippingPlanes = planes;
    e.mesh.material.needsUpdate = true;
    if (e.cap) { e.cap.material.clippingPlanes = planes; e.cap.material.needsUpdate = true; }
  }

  // -- landmarks ----------------------------------------------------------------

  // Every landmark node: name, status, world position (three units), shown or not.
  landmarks() {
    const out = [];
    for (const e of this.nodes.values()) {
      if (this.kind(e.node) !== "landmark") continue;
      const p = new THREE.Vector3().setFromMatrixPosition(e.mesh.matrixWorld);
      out.push({ name: e.node.name, status: e.node.tags.status, position: p, visible: e.mesh.visible, set: e.node.set });
    }
    return out;
  }

  highlight(name) {
    for (const e of this.nodes.values()) {
      if (this.kind(e.node) !== "landmark") continue;
      const on = e.node.name === name;
      const s = on ? 2.2 : 1;
      const m = toMatrix(e.node.rest).multiply(new THREE.Matrix4().makeScale(s, s, s));
      e.mesh.matrix.copy(m);
      e.mesh.material.emissive?.setHex(on ? 0x663300 : 0x000000);
    }
  }

  // Screen position (CSS px in the container) of a world point; null behind the camera.
  project(p) {
    const v = p.clone().project(this.camera);
    if (v.z > 1) return null;
    return [(v.x + 1) / 2 * this.container.clientWidth, (1 - v.y) / 2 * this.container.clientHeight];
  }

  // Glide the orbit target (and the camera with it) to a world point.
  focus(p, distance = null) {
    const t0 = this.controls.target.clone(), c0 = this.camera.position.clone();
    const dir = c0.clone().sub(t0);
    if (distance) dir.setLength(distance);
    const t1 = p.clone(), c1 = p.clone().add(dir);
    let i = 0;
    this.tween = () => {
      i += 1; const k = Math.min(1, i / 24), e = k * k * (3 - 2 * k);
      this.controls.target.lerpVectors(t0, t1, e);
      this.camera.position.lerpVectors(c0, c1, e);
      if (k >= 1) this.tween = null;
    };
  }

  // -- the Fusion-built parts ---------------------------------------------------

  async setParts(parts) {
    for (const p of this.parts.values()) p.mesh.parent?.remove(p.mesh);
    this.parts.clear();
    await Promise.all(parts.map(async (p) => {
      const g = await this.geometry(`${p.file}?v=${p.stamp}`);
      const metal = p.id !== "insert";  // no environment map: keep metalness low or it renders black
      const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({
        color: COLOURS[p.id], metalness: metal ? 0.25 : 0, roughness: metal ? 0.35 : 0.55,
        transparent: true, opacity: 1, side: THREE.DoubleSide,
      }));
      mesh.matrixAutoUpdate = false;
      this.groups.get(p.set).add(mesh);
      this.parts.set(p.id, { mesh, part: p });
      if (this.items[p.id]) this.setItem(p.id, {});
    }));
  }

  // The parts are stored placed on the plan they were built for. When the plan moves, they
  // move with it: rest = (pose now) (pose built)^-1.
  setPartsRest(built, now) {
    for (const { mesh, part } of this.parts.values()) {
      const key = part.set === "femoral" ? "femoral" : "tibial";
      const m = toMatrix(now[key]).multiply(toMatrix(built[key]).invert());
      mesh.matrix.copy(m);
    }
  }

  async geometry(url) {
    if (this.meshCache.has(url)) return this.meshCache.get(url);
    const pending = fetch(url).then((r) => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.arrayBuffer(); })
      .then((buf) => {
        const { positions, indices } = parseGlb(buf);
        const g = creased(new Float32Array(positions), new Uint32Array(indices), CREASE_DEG);
        g.computeBoundingSphere();
        return g;
      });
    this.meshCache.set(url, pending);
    return pending;
  }

  // -- views ------------------------------------------------------------------

  // Scan frame: +X patient-left, +Y posterior, +Z proximal. `centre` in mm.
  view(name, centre, side, distance = 0.36) {
    this.tween = null;
    const lateral = side === "right" ? -1 : 1;     // the lateral side of a right knee is -X
    const dirs = {
      ap: [0, -1, 0.12], pa: [0, 1, 0.12], lateral: [lateral, 0, 0.1], medial: [-lateral, 0, 0.1],
      tibia: [0, -0.2, 1], femur: [0, -0.2, -1], iso: [lateral * 0.55, -0.8, 0.35],
    };
    const c = new THREE.Vector3(...centre).divideScalar(MM);
    this.controls.target.copy(c);
    this.camera.up.set(0, name === "tibia" || name === "femur" ? -1 : 0, name === "tibia" || name === "femur" ? 0 : 1);
    if (name === "femur") this.camera.up.set(0, 1, 0);
    this.camera.position.copy(c).add(new THREE.Vector3(...dirs[name]).normalize().multiplyScalar(distance));
    this.camera.lookAt(c);
    this.controls.update();
  }

  snapshot() { return this.renderer.domElement.toDataURL("image/png"); }
}

// Unindexed geometry whose vertex normals average only the faces around a vertex that
// lie within `deg` of the face being drawn: smooth bone, crisp cut faces and part edges.
// (Averaging across the cut edge shaded the long, thin cap triangles in streaks.)
function creased(pos, idx, deg) {
  const nV = pos.length / 3, nF = idx.length / 3, cos = Math.cos(deg * Math.PI / 180);
  const raw = new Float32Array(nF * 3), unit = new Float32Array(nF * 3);
  for (let f = 0; f < nF; f++) {
    const a = idx[3 * f] * 3, b = idx[3 * f + 1] * 3, c = idx[3 * f + 2] * 3;
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1;
    raw[3 * f] = nx; raw[3 * f + 1] = ny; raw[3 * f + 2] = nz;
    unit[3 * f] = nx / l; unit[3 * f + 1] = ny / l; unit[3 * f + 2] = nz / l;
  }
  const start = new Uint32Array(nV + 1);
  for (let i = 0; i < idx.length; i++) start[idx[i] + 1]++;
  for (let v = 0; v < nV; v++) start[v + 1] += start[v];
  const fill = start.slice(0, nV), around = new Uint32Array(idx.length);
  for (let i = 0; i < idx.length; i++) around[fill[idx[i]]++] = (i / 3) | 0;
  const P = new Float32Array(nF * 9), N = new Float32Array(nF * 9);
  for (let f = 0; f < nF; f++) {
    const fx = unit[3 * f], fy = unit[3 * f + 1], fz = unit[3 * f + 2];
    for (let k = 0; k < 3; k++) {
      const v = idx[3 * f + k], o = 9 * f + 3 * k;
      let sx = 0, sy = 0, sz = 0;
      for (let j = start[v]; j < start[v + 1]; j++) {
        const g = around[j];
        if (unit[3 * g] * fx + unit[3 * g + 1] * fy + unit[3 * g + 2] * fz >= cos) {
          sx += raw[3 * g]; sy += raw[3 * g + 1]; sz += raw[3 * g + 2];
        }
      }
      const l = Math.hypot(sx, sy, sz) || 1;
      P[o] = pos[3 * v]; P[o + 1] = pos[3 * v + 1]; P[o + 2] = pos[3 * v + 2];
      N[o] = sx / l; N[o + 1] = sy / l; N[o + 2] = sz / l;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(P, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(N, 3));
  return g;
}

export function toMatrix(rows) {
  const m = new THREE.Matrix4();
  m.set(rows[0][0], rows[0][1], rows[0][2], rows[0][3] / MM,
        rows[1][0], rows[1][1], rows[1][2], rows[1][3] / MM,
        rows[2][0], rows[2][1], rows[2][2], rows[2][3] / MM,
        rows[3][0], rows[3][1], rows[3][2], rows[3][3]);
  return m;
}
