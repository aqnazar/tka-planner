// One knee at a time: the Paper 1 script's parts against the planner's, in the knee's
// own STL frame. Meshes arrive as float32 positions (relative to case.json's
// centre_stl_mm) followed by uint32 indices; picked points are reported back in STL
// millimetres by adding the centre again, so they can be checked against the STL.

import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const GROUPS = {
  bone: "Bones",
  legacy: "Paper 1 script",
  like_for_like: "Planner, like-for-like",
  default: "Planner, default",
};

const host = document.getElementById("canvas");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0xf6f5f2);
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 1, 5000);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8478, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
scene.add(sun);
const root = new THREE.Group();
scene.add(root);
const markers = new THREE.Group();
scene.add(markers);

let current = null;      // case.json of the knee shown
let objects = {};        // layer or plane id -> THREE object
let picks = [];          // picked points, STL millimetres

function resize() {
  const { clientWidth: w, clientHeight: h } = host;
  renderer.setSize(w, h);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

function frame() {
  sun.position.copy(camera.position);
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

async function mesh(folder, layer) {
  const buffer = await (await fetch(`${folder}/${layer.file}`)).arrayBuffer();
  const positions = new Float32Array(buffer, 0, layer.n_vertices * 3);
  const indices = new Uint32Array(buffer, layer.n_vertices * 12, layer.n_triangles * 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({
    color: layer.colour, roughness: 0.6, metalness: 0.05, side: THREE.DoubleSide,
    transparent: layer.opacity < 1, opacity: layer.opacity, depthWrite: layer.opacity >= 1,
  });
  const object = new THREE.Mesh(geometry, material);
  object.renderOrder = layer.opacity < 1 ? 2 : 1;
  return object;
}

function plane(p) {
  const normal = new THREE.Vector3(...p.normal).normalize();
  const group = new THREE.Group();
  const disc = new THREE.Mesh(
    new THREE.CircleGeometry(42, 72),
    new THREE.MeshBasicMaterial({ color: p.colour, transparent: true, opacity: 0.28,
                                  side: THREE.DoubleSide, depthWrite: false }));
  const ring = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(
      Array.from({ length: 96 }, (_, i) => {
        const a = (i / 96) * Math.PI * 2;
        return new THREE.Vector3(42 * Math.cos(a), 42 * Math.sin(a), 0);
      })),
    new THREE.LineBasicMaterial({ color: p.colour }));
  group.add(disc, ring);
  group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
  group.position.set(...p.point);
  group.renderOrder = 3;
  return group;
}

function view(name) {
  if (!current) return;
  const f = current.frames.femoral || current.frames.tibial;
  const ant = new THREE.Vector3(...(f ? f.anterior : [0, -1, 0]));
  const lat = new THREE.Vector3(...(f ? f.lateral : [1, 0, 0]));
  const prox = new THREE.Vector3(...(f ? f.proximal : [0, 0, 1]));
  const dirs = {
    anterior: ant, posterior: ant.clone().negate(), lateral: lat,
    medial: lat.clone().negate(), proximal: prox, distal: prox.clone().negate(),
  };
  const target = new THREE.Vector3(0, 0, 0).addScaledVector(prox, -20);
  const up = (name === "proximal" || name === "distal") ? ant : prox;
  camera.up.copy(up);
  camera.position.copy(target).addScaledVector(dirs[name], 520);
  controls.target.copy(target);
  controls.update();
}
document.querySelectorAll("[data-view]").forEach((b) =>
  b.addEventListener("click", () => view(b.dataset.view)));

// ---- picking ----------------------------------------------------------------------------
const ray = new THREE.Raycaster();
const readout = document.getElementById("readout");
let downAt = null;
renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!document.getElementById("measure").checked || !downAt) return;
  if (Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return; // a drag
  const r = renderer.domElement.getBoundingClientRect();
  pickAt((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
});

// Pick at a fraction of the canvas (0..1 from the top left).
function pickAt(fx, fy) {
  ray.setFromCamera(new THREE.Vector2(fx * 2 - 1, -fy * 2 + 1), camera);
  const visible = Object.values(objects).filter((o) => o.visible && o.isMesh);
  const hit = ray.intersectObjects(visible, false)[0];
  if (!hit) return;
  const c = current.centre_stl_mm;
  const stl = [hit.point.x + c[0], hit.point.y + c[1], hit.point.z + c[2]];
  picks.push({ stl, local: hit.point.clone(), layer: hit.object.userData.label });
  // Transparent so it is drawn in the last pass, after the see-through bones.
  const dot = new THREE.Mesh(new THREE.SphereGeometry(1.5, 16, 12),
                             new THREE.MeshBasicMaterial({ color: 0xd62828, depthTest: false,
                                                           transparent: true, opacity: 1 }));
  dot.position.copy(hit.point);
  dot.renderOrder = 10;
  markers.add(dot);
  showPicks();
}
document.getElementById("clear").addEventListener("click", () => {
  picks = [];
  markers.clear();
  showPicks();
});

function showPicks() {
  if (!picks.length) { readout.style.display = "none"; return; }
  const fmt = (v) => v.map((x) => x.toFixed(2).padStart(9)).join(" ");
  const lines = picks.slice(-2).map((p, i) => `P${picks.length - Math.min(2, picks.length) + i + 1} ${fmt(p.stl)}  ${p.layer}`);
  if (picks.length >= 2) {
    const [a, b] = picks.slice(-2);
    const d = [0, 1, 2].map((k) => b.stl[k] - a.stl[k]);
    lines.push(`distance ${Math.hypot(...d).toFixed(2)} mm   (dx dy dz ${fmt(d)})`);
    const f = current.frames.femoral;
    if (f) {
      const along = (axis) => d.reduce((s, x, k) => s + x * axis[k], 0).toFixed(2);
      lines.push(`along femoral frame: anterior ${along(f.anterior)}  lateral ${along(f.lateral)}  proximal ${along(f.proximal)} mm`);
    }
  }
  readout.textContent = "STL frame, mm\n" + lines.join("\n");
  readout.style.display = "block";
}

// ---- panel ----------------------------------------------------------------------------
function swatch(colour) {
  return `<span class="swatch" style="background:${colour}"></span>`;
}

function layersPanel(doc) {
  const box = document.getElementById("layers");
  box.innerHTML = "";
  for (const [group, title] of Object.entries(GROUPS)) {
    const items = [...doc.layers.filter((l) => l.group === group),
                   ...doc.planes.filter((p) => p.group === group)];
    if (!items.length) continue;
    const div = document.createElement("div");
    div.className = "group";
    const all = document.createElement("label");
    all.innerHTML = `<input type="checkbox"> ${swatch(items[0].colour)}${title}`;
    div.appendChild(all);
    const list = document.createElement("div");
    list.className = "items";
    const boxes = [];
    for (const item of items) {
      const row = document.createElement("label");
      row.style.display = "block";
      const isPlane = !("file" in item);
      row.innerHTML = `<input type="checkbox"> ${item.label}${isPlane ? " (plane)" : ""}`;
      const cb = row.querySelector("input");
      cb.checked = objects[item.id] ? objects[item.id].visible : false;
      cb.addEventListener("change", () => {
        objects[item.id].visible = cb.checked;
        sync();
      });
      boxes.push(cb);
      list.appendChild(row);
    }
    const master = all.querySelector("input");
    const sync = () => {
      master.checked = boxes.every((b) => b.checked);
      master.indeterminate = !master.checked && boxes.some((b) => b.checked);
    };
    master.addEventListener("change", () => {
      boxes.forEach((b, i) => { b.checked = master.checked; objects[items[i].id].visible = master.checked; });
      sync();
    });
    sync();
    div.appendChild(list);
    box.appendChild(div);
  }
}

function cell(v) {
  if (v === null || v === undefined) return "";
  if (Array.isArray(v)) return v.map((x) => (typeof x === "number" ? x.toFixed(2) : x)).join(", ");
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
  return String(v);
}

function tablesPanel(doc) {
  const box = document.getElementById("tables");
  box.innerHTML = "";
  for (const t of doc.tables) {
    if (!t.rows.length) continue;
    const cols = Object.keys(t.rows[0]);
    const html = [`<h2>${t.title}</h2><table><thead><tr>`,
      ...cols.map((c) => `<th>${c}</th>`), "</tr></thead><tbody>",
      ...t.rows.map((r) => `<tr>${cols.map((c) => `<td>${cell(r[c])}</td>`).join("")}</tr>`),
      `</tbody></table><p class="note">${t.note || ""}</p>`];
    box.insertAdjacentHTML("beforeend", html.join(""));
  }
  const notes = document.getElementById("notes");
  notes.innerHTML = doc.notes.length
    ? `<h2>Notes</h2><ul>${doc.notes.map((n) => `<li>${n}</li>`).join("")}</ul>` : "";
}

function statusPanel(doc) {
  const s = doc.summary;
  const rows = [
    `<div class="row"><b>${s.case}</b> ${s.side} knee${s.reference ? " (full-length reference)" : ""}</div>`,
    `<div class="row">Paper 1 script: ${s.legacy_ok ? "completed" : '<span class="bad">failed</span>'}, size <b>${s.legacy_size ?? "-"}</b></div>`,
    ...s.flags.map((f) => `<div class="row flag">&#9888; ${f}</div>`),
    `<div class="row">Planner: ${s.planner === "accepted" ? "accepted" : `<span class="bad">refused</span> ${s.planner_reason || ""}`}</div>`,
  ];
  document.getElementById("status").innerHTML = rows.join("");
}

async function show(caseId) {
  document.querySelectorAll("#cases button").forEach((b) =>
    b.classList.toggle("active", b.dataset.case === caseId));
  history.replaceState(null, "", `#${caseId}`);
  const folder = `cases/${caseId}`;
  const doc = await (await fetch(`${folder}/case.json`)).json();
  root.clear();
  markers.clear();
  picks = [];
  showPicks();
  objects = {};
  current = doc;
  await Promise.all(doc.layers.map(async (layer) => {
    const o = await mesh(folder, layer);
    o.visible = layer.visible;
    o.userData.label = layer.label;
    objects[layer.id] = o;
    root.add(o);
  }));
  for (const p of doc.planes) {
    const o = plane(p);
    o.visible = p.visible !== false;
    objects[p.id] = o;
    root.add(o);
  }
  statusPanel(doc);
  layersPanel(doc);
  tablesPanel(doc);
  document.getElementById("caption").textContent =
    `${caseId}: all positions in the knee's STL frame (mm); planes are drawn 84 mm across`;
  view("anterior");
}

async function start() {
  resize();
  frame();
  const cases = await (await fetch("cases.json")).json();
  const nav = document.getElementById("cases");
  for (const c of cases) {
    const b = document.createElement("button");
    b.dataset.case = c.case;
    const flag = c.flags.length || c.planner !== "accepted" ? ' <span class="flag">&#9888;</span>' : "";
    b.innerHTML = `<div class="id">${c.case}${flag}</div><div class="sub">Paper 1 ${c.legacy_size ?? "-"} · planner ${c.planner}</div>`;
    b.addEventListener("click", () => show(c.case));
    nav.appendChild(b);
  }
  const wanted = location.hash.slice(1);
  await show(cases.some((c) => c.case === wanted) ? wanted : cases[0].case);
  // ?view=lateral&pick=0.5,0.6;0.5,0.7 replays clicks: how the page is checked headless.
  const params = new URLSearchParams(location.search);
  if (params.get("view")) view(params.get("view"));
  if (params.get("pick")) {
    camera.updateMatrixWorld();
    for (const pair of params.get("pick").split(";")) pickAt(...pair.split(",").map(Number));
  }
}
start();
