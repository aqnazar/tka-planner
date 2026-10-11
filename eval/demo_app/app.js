// Parametric TKA planner demo: the planner's own session (plan, commit, trial reduction)
// with the implant derived from the patient by rules and built in Fusion.
// Every clinical number comes from the server; this file only draws and asks.

import { Viewer, COLOURS } from "./viewer.js";

const el = (id) => document.getElementById(id);
const viewer = new Viewer(el("view"));
const S = { knees: [], knee: null, entry: null, session: null, schema: null, controls: {}, trial: {},
            stale: true, step: "anatomy", report: [], derived: null, kin: null, kinNow: null, playing: false,
            view: "ap", condyle: "medial", builtControls: {} };

async function api(method, path, body) {
  const r = await fetch(path, { method, headers: body ? { "Content-Type": "application/json" } : {},
                                body: body ? JSON.stringify(body) : undefined });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
async function busy(text, work) {
  el("busy").hidden = false; el("busy").firstElementChild.textContent = text;
  try { return await work(); } catch (e) { flag(e.message); return null; } finally { el("busy").hidden = true; }
}
function flag(msg) { const b = el("banner"); b.hidden = !msg; b.innerHTML = msg || ""; }
const mm = (v) => (typeof v === "string" ? parseFloat(v) : v);
const f1 = (v) => (v == null || isNaN(mm(v)) ? "&ndash;" : mm(v).toFixed(1));
const f2 = (v) => (v == null || isNaN(mm(v)) ? "&ndash;" : mm(v).toFixed(2));
const sgn = (v, d = 1) => (v == null || isNaN(v) ? "&ndash;" : `${v >= 0 ? "+" : "&minus;"}${Math.abs(v).toFixed(d)}`);
const hex = (c) => "#" + c.toString(16).padStart(6, "0");

// Sliders draw their fill from zero (two-sided controls) or from the left.
function paint(input) {
  const min = +input.min, max = +input.max, v = +input.value, pct = (x) => ((x - min) / (max - min)) * 100;
  const two = min < 0 && max > 0;
  input.style.setProperty("--lo", `${two ? pct(Math.min(0, v)) : 0}%`);
  input.style.setProperty("--hi", `${two ? pct(Math.max(0, v)) : pct(v)}%`);
}

// ---------------------------------------------------------------- session

const meshUrl = (id) => `/api/sessions/${S.session}/meshes/${id}.glb`;

async function apply(payload) {
  S.controls = payload.controls ?? S.controls;
  S.trial = payload.trial ?? S.trial;
  S.stale = Boolean(payload.stale);
  if (payload.scene) await viewer.setScene(payload.scene, meshUrl);
  else if (payload.delta) await viewer.applyDelta(payload.delta);
  if (payload.clip) viewer.setClip(payload.clip);
  if (payload.report) S.report = payload.report;
  if (payload.kinematics) { S.kinNow = payload.kinematics; kinNow(); }
  syncPanel(); readout();
}

let inflight = null, queued = null;
function sendControls(changes) {
  queued = { ...(queued || {}), ...changes };
  if (inflight) return inflight;
  inflight = (async () => {
    while (queued) {
      const batch = queued; queued = null;
      await apply(await api("POST", `/api/sessions/${S.session}/controls`, { changes: batch }));
      await followPlan();
      liveDerive();
    }
    inflight = null;
  })().catch((e) => { inflight = null; queued = null; flag(e.message); });
  return inflight;
}

// Display-only controls (planes, axes, landmarks): no re-plan, no derivation.
async function display(changes) {
  await apply(await api("POST", `/api/sessions/${S.session}/controls`, { changes }));
  sceneRows();
}

// The built parts follow the plan as it is adjusted (placement only: their sizes are the
// ones built, until the plan is derived and built again).
async function followPlan() {
  if (!S.entry) return;
  const now = await api("GET", `/api/sessions/${S.session}/poses`);
  viewer.setPartsRest(S.entry.poses, now);
  S.kin = null;
  implantStatus();
}

// The derivation is under a second, so it runs on every change of the plan: the
// parameters move with the sliders, and what differs from the parts built shows orange.
let deriveTimer = null;
function liveDerive() {
  clearTimeout(deriveTimer);
  deriveTimer = setTimeout(async () => {
    try {
      S.derived = await api("POST", `/api/sessions/${S.session}/parametric`, { name: S.knee.name });
    } catch (e) { return; }
    if (S.step === "implant") { renderImplant(); if (!el("drawer").hidden) figures(); }
    if (S.step === "build") renderBuild();
    readout(); implantStatus();
  }, 350);
}

function implantStatus() {
  const n = Object.keys(S.derived?.changed || {}).length;
  if (n && S.step !== "build" && S.step !== "anatomy") {
    flag(`The plan has moved: <b>${n} Fusion parameter${n > 1 ? "s" : ""}</b> differ from the parts built. The parts follow the new cuts but keep their built sizes. <button id="go-implant">See the derivation</button><button id="go-build" class="primary">Build in Fusion</button>`);
    el("go-implant")?.addEventListener("click", () => setStep("implant"));
    el("go-build")?.addEventListener("click", () => setStep("build"));
  } else if (S.stale && S.step === "reduce") {
    flag(`Reduce needs committed cuts. <button id="go-commit">Commit</button>`);
    el("go-commit")?.addEventListener("click", commit);
  } else flag("");
}

async function openKnee(name) {
  S.knee = S.knees.find((k) => k.name === name);
  S.derived = null; S.builtControls = {}; S.kin = null; S.kinNow = null;
  el("intro").hidden = true;
  const ok = await busy(`Opening ${S.knee.label}: measuring the bones, finding landmarks, planning`, async () => {
    const snap = await api("POST", "/api/sessions", { folder: S.knee.folder, side: S.knee.side, name: S.knee.name });
    if (S.session) api("DELETE", `/api/sessions/${S.session}`).catch(() => {});
    S.session = snap.session_id;
    S.schema = S.schema || await api("GET", "/api/schema");
    buildPanel();
    await apply(snap);
    // The plan the knee's parts were built for (patients: the tibial resection that
    // gives the insert its minimum thickness).
    if (Object.keys(S.knee.plan_controls || {}).length)
      await apply(await api("POST", `/api/sessions/${S.session}/controls`, { changes: S.knee.plan_controls }));
    S.entry = await api("GET", `/api/demo/knees/${name}`);
    await viewer.setParts(S.entry.parts);
    await followPlan();
    return true;
  });
  if (!ok) return;
  el("knee").querySelector(".k-label").textContent = S.knee.label;
  el("side").textContent = `${S.knee.side} knee`;
  history.replaceState(null, "", `?knee=${name}`);
  await commit();
  await display({ show_planes: false, show_axes: true, show_landmarks: true });
  landmarkLabels();
  setStep(S.step);
  setView(S.view);
  liveDerive();
}

async function commit() {
  await busy("Cutting the bones on the plan", async () => {
    await apply(await api("POST", `/api/sessions/${S.session}/commit`));
  });
}

// ---------------------------------------------------------------- intro

function intro() {
  const groups = [["reference", "Reference knee and two size extremes", "The reference bones (full-length CT), and the same knee scaled to a small and a large one."],
                  ["patient", "Patients", "Knee-only CT, about 150 mm of each bone: hip and ankle are not in view, so the mechanical axes are assumed."]];
  el("knee-cards").innerHTML = groups.map(([g, title, note]) => {
    const ks = S.knees.filter((k) => (k.group || "reference") === g);
    if (!ks.length) return "";
    return `<h3>${title}</h3><p class="hint">${note}</p><div class="cards">${ks.map((k) => {
      const a = k.anatomy || {};
      return `<button class="kcard${S.knee?.name === k.name ? " on" : ""}" data-knee="${k.name}">
        <span class="kc-name">${k.label}</span>
        <span class="kc-side">${k.side} knee${k.scale ? " &middot; scaled" : ""}</span>
        <span class="kc-facts mono">tibial cut ${f1(a.tibial_cut_ml_mm)} &times; ${f1(a.tibial_cut_ap_mm)} mm<br>nearest catalogue size ${a.nearest_catalogue_size ?? "&ndash;"}</span>
        <span class="kc-built">${k.built ? `built in Fusion &middot; femoral v${k.versions?.femoral ?? "?"}` : "not built yet"}</span></button>`;
    }).join("")}</div>`;
  }).join("");
  for (const b of el("knee-cards").querySelectorAll(".kcard")) b.onclick = () => openKnee(b.dataset.knee);
  el("intro").hidden = false;
}

// ---------------------------------------------------------------- panel (planner schema)

function buildPanel() {
  const host = el("controls"); host.textContent = "";
  for (const g of S.schema.controls) {
    if (g.group === "Display" || g.group === "Resection") continue;
    const box = document.createElement("fieldset");
    box.innerHTML = `<legend>${g.group}</legend>`;
    for (const c of g.controls) {
      if (c.name === "implant_mode") continue;
      box.append(control(c, () => S.controls[c.name], sendControls));
    }
    host.append(box);
  }
  const t = el("trial"); t.textContent = "";
  const box = document.createElement("fieldset");
  box.innerHTML = "<legend>Trial</legend>";
  for (const c of S.schema.trial) box.append(control(c, () => S.trial[c.name], trial));
  t.append(box);
}

async function trial(changes) {
  await apply(await api("POST", `/api/sessions/${S.session}/trial`, { changes, model: el("motion").value }));
}

function control(c, read, send) {
  const w = document.createElement("div"); w.className = "control"; if (c.help) w.title = c.help;
  const unit = c.unit || "";
  if (c.kind === "float") {
    w.innerHTML = `<div class="row"><span class="name">${c.label}</span><span class="value"></span></div><input type="range" min="${c.min}" max="${c.max}" step="${c.step}">`;
    const [input] = w.getElementsByTagName("input"), val = w.querySelector(".value");
    const show = (v) => { val.textContent = `${(+v).toFixed(1)} ${unit}`; paint(input); };
    input.oninput = () => { show(input.value); send({ [c.name]: +input.value }); };
    input.ondblclick = () => { const h = c.min <= 0 && c.max >= 0 ? 0 : c.min; input.value = h; show(h); send({ [c.name]: h }); };
    w.sync = () => { const v = +(read() ?? 0); input.value = v; show(v); w.classList.toggle("changed", c.min < 0 && c.max > 0 && Math.abs(v) > 1e-9); };
  } else if (c.kind === "bool") {
    w.innerHTML = `<label class="chk"><input type="checkbox"><span>${c.label}</span></label>`;
    const input = w.querySelector("input");
    input.onchange = () => send({ [c.name]: input.checked });
    w.sync = () => { input.checked = Boolean(read()); };
  } else {
    w.innerHTML = `<div class="row"><span class="name">${c.label}</span></div><select>${c.choices.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select>`;
    const s = w.querySelector("select");
    s.onchange = () => send({ [c.name]: s.value });
    w.sync = () => { s.value = read() ?? ""; };
  }
  return w;
}

function syncPanel() {
  for (const w of document.querySelectorAll(".control")) w.sync?.();
  const b = el("commit");
  b.textContent = S.stale ? "Commit cuts" : "Cuts committed";
  b.classList.toggle("stale", S.stale);
  el("state").innerHTML = S.knee ? `${S.stale ? "plan not committed" : "cuts committed"} &middot; ${S.knee.name}` : "";
}

// ---------------------------------------------------------------- the Scene panel (in the viewport)

const ITEMS = [
  ["Bone", [["Femur", "Femur", COLOURS.bone], ["Tibia", "Tibia", COLOURS.bone], ["resected", "Resected bone", COLOURS.resected]]],
  ["Implant (Fusion)", [["femoral", "Femoral component", COLOURS.femoral], ["plate", "Tibial plate", COLOURS.plate], ["insert", "Insert", COLOURS.insert]]],
  ["Cutting blocks", [["femoral_block", "Femoral block", COLOURS.block], ["tibial_block", "Tibial block", COLOURS.block]]],
];
const GUIDES = [["show_planes", "Cut planes"], ["show_axes", "Mechanical axes"], ["show_landmarks", "Landmarks"]];

// What each step shows; the Scene panel changes it until the next step.
const PRESETS = {
  anatomy: { Femur: [1, 1], Tibia: [1, 1], resected: [1, 1], femoral: [0, 1], plate: [0, 1], insert: [0, 1], femoral_block: [0, 0.35], tibial_block: [0, 0.35], library: [0, 0.6] },
  plan: { Femur: [1, 1], Tibia: [1, 1], resected: [1, 0.3], femoral: [1, 1], plate: [1, 1], insert: [1, 1], femoral_block: [0, 0.35], tibial_block: [0, 0.35], library: [0, 0.6] },
  implant: { Femur: [1, 1], Tibia: [1, 1], resected: [0, 0.3], femoral: [1, 1], plate: [1, 1], insert: [1, 1], femoral_block: [0, 0.35], tibial_block: [0, 0.35], library: [0, 0.6] },
  reduce: { Femur: [1, 1], Tibia: [1, 1], resected: [0, 0.3], femoral: [1, 1], plate: [1, 1], insert: [1, 1], femoral_block: [0, 0.35], tibial_block: [0, 0.35], library: [0, 0.6] },
  build: { Femur: [1, 0.35], Tibia: [1, 0.35], resected: [0, 0.3], femoral: [1, 1], plate: [1, 1], insert: [1, 1], femoral_block: [0, 0.35], tibial_block: [0, 0.35], library: [0, 0.6] },
};
const GUIDE_PRESETS = { anatomy: { show_planes: false, show_axes: true, show_landmarks: true },
                        plan: { show_planes: true, show_axes: false, show_landmarks: false } };

function preset(step) {
  for (const [k, [on, a]] of Object.entries(PRESETS[step])) viewer.setItem(k, { on: Boolean(on), alpha: a });
  viewer.setShellAsBone(step === "anatomy");
  const g = GUIDE_PRESETS[step] || { show_planes: false, show_axes: false, show_landmarks: false };
  const changed = Object.fromEntries(Object.entries(g).filter(([k, v]) => Boolean(S.controls[k]) !== v));
  if (Object.keys(changed).length && S.session) display(changed);
  sceneRows();
}

function sceneRows() {
  const host = el("scene-rows"); host.textContent = "";
  for (const [title, rows] of ITEMS) {
    const h = document.createElement("div"); h.className = "sp-group"; h.textContent = title; host.append(h);
    for (const [key, label, colour] of rows) {
      const s = viewer.items[key] || { on: true, alpha: 1 };
      const r = document.createElement("div"); r.className = "sp-row";
      r.innerHTML = `<label class="chk"><input type="checkbox" ${s.on ? "checked" : ""}><span><i class="sw" style="background:${hex(colour)}"></i>${label}</span></label>
        <input type="range" class="xray" min="5" max="100" value="${Math.round(s.alpha * 100)}" title="Opacity (X-ray)">`;
      const [cb, range] = r.querySelectorAll("input");
      paint(range);
      cb.onchange = () => viewer.setItem(key, { on: cb.checked });
      range.oninput = () => { paint(range); viewer.setItem(key, { alpha: range.value / 100, on: true }); cb.checked = true; };
      host.append(r);
    }
  }
  const h = document.createElement("div"); h.className = "sp-group"; h.textContent = "Guides"; host.append(h);
  for (const [name, label] of GUIDES) {
    const r = document.createElement("div"); r.className = "sp-row";
    r.innerHTML = `<label class="chk"><input type="checkbox" ${S.controls[name] ? "checked" : ""}><span>${label}</span></label>`;
    r.querySelector("input").onchange = (e) => display({ [name]: e.target.checked });
    host.append(r);
  }
  const r = document.createElement("div"); r.className = "sp-row";
  r.innerHTML = `<label class="chk" title="The planner's own preview: the catalogue part scaled to this knee"><input type="checkbox" ${viewer.items.library?.on ? "checked" : ""}><span>Catalogue part (preview)</span></label>`;
  r.querySelector("input").onchange = (e) => viewer.setItem("library", { on: e.target.checked, alpha: 0.6 });
  host.append(r);
  landmarkLabels();
}

// ---------------------------------------------------------------- landmarks

const LANDMARKS = {
  "femur.condyle_distal_medial": ["Distal medial condyle", "depth of the distal cut"],
  "femur.condyle_distal_lateral": ["Distal lateral condyle", "depth of the distal cut"],
  "femur.condyle_posterior_medial": ["Posterior medial condyle", "posterior cut, femoral rotation"],
  "femur.condyle_posterior_lateral": ["Posterior lateral condyle", "posterior cut, femoral rotation"],
  "femur.epicondyle_lateral": ["Lateral epicondyle", "transepicondylar axis"],
  "femur.epicondyle_medial_prominence": ["Medial epicondyle", "transepicondylar axis"],
  "femur.epicondyle_medial_sulcus": ["Medial epicondylar sulcus", "surgical epicondylar axis"],
  "femur.notch_centre": ["Intercondylar notch", "femoral mechanical axis, distal end"],
  "femur.canal_centre_distal": ["Femoral canal, distal", "anatomical axis"],
  "femur.canal_centre_proximal": ["Femoral canal, proximal", "anatomical axis"],
  "tibia.plateau_medial_lowest": ["Medial plateau, lowest", "depth of the tibial cut"],
  "tibia.plateau_lateral_lowest": ["Lateral plateau, lowest", "depth of the tibial cut"],
  "tibia.spine_medial": ["Medial tibial spine", "tibial centre"],
  "tibia.spine_lateral": ["Lateral tibial spine", "tibial centre"],
  "tibia.pcl_insertion_midpoint": ["PCL insertion", "tibial rotation (AP axis)"],
  "tibia.tubercle_medial_third": ["Tubercle, medial third", "tibial rotation (AP axis)"],
  "tibia.tubercle_patellar_tendon_medial_border": ["Patellar tendon, medial border", "tibial rotation"],
  "tibia.plateau_medial_anterior": ["Medial plateau, anterior", "posterior slope"],
  "tibia.plateau_medial_posterior": ["Medial plateau, posterior", "posterior slope"],
  "tibia.canal_centre_proximal": ["Tibial canal, proximal", "tibial mechanical axis"],
  "tibia.canal_centre_distal": ["Tibial canal, distal", "tibial mechanical axis"],
};
const lmName = (id) => LANDMARKS[id]?.[0] ?? id.split(".").pop().replaceAll("_", " ");
const NEAR_JOINT = /condyle|epicondyle|notch|plateau|spine|pcl/;

// A landmark that should sit at the joint but is far from it is shown, and flagged.
function lmCheck(l) {
  if (!S.entry || !NEAR_JOINT.test(l.name)) return null;
  const F = S.entry.poses.femoral, T = S.entry.poses.tibial;
  const c = [0, 1, 2].map((i) => (F[i][3] + T[i][3]) / 2000);
  const d = Math.hypot(l.position.x - c[0], l.position.y - c[1], l.position.z - c[2]) * 1000;
  return d > 60 ? `${d.toFixed(0)} mm from the joint: check` : null;
}

function landmarkList() {
  const all = viewer.landmarks();
  const facts = S.entry?.anatomy || {};
  const len = (name) => {
    const ms = ["", ".Shell"].map((s) => viewer.nodes.get(name + s)?.mesh).filter(Boolean);
    let lo = Infinity, hi = -Infinity;
    for (const m of ms) { m.geometry.computeBoundingBox(); lo = Math.min(lo, m.geometry.boundingBox.min.z); hi = Math.max(hi, m.geometry.boundingBox.max.z); }
    return isFinite(lo) ? ((hi - lo) * 1000).toFixed(0) : "&ndash;";
  };
  el("knee-facts").innerHTML = `<div class="big">
    <div><span>Femur in the scan</span><b>${len("Femur")} <small>mm</small></b></div>
    <div><span>Tibia in the scan</span><b>${len("Tibia")} <small>mm</small></b></div>
    <div><span>Tibial cut ML &times; AP</span><b>${f1(facts.tibial_cut_ml_mm)} <small>&times;</small> ${f1(facts.tibial_cut_ap_mm)}</b></div>
    <div><span>Nearest catalogue size</span><b>${facts.nearest_catalogue_size ?? "&ndash;"}</b></div></div>`;
  const host = el("landmark-list");
  host.innerHTML = ["femur", "tibia"].map((bone) => {
    const ls = all.filter((l) => l.name.startsWith(bone + "."));
    return `<h3>${bone === "femur" ? "Femur" : "Tibia"} &middot; ${ls.length} landmarks</h3><table class="lm">${ls.map((l) => {
      const chk = lmCheck(l);
      return `<tr data-lm="${l.name}" class="${chk ? "check" : ""}"><td><i class="dot ${l.status}"></i></td><td>${lmName(l.name)}<div class="from">${chk ? `<span class="warn">${chk}</span>` : (LANDMARKS[l.name]?.[1] ?? "")}</div></td><td class="num"><span class="tag ${l.status === "present" ? "ok" : "warn"}">${l.status}</span></td></tr>`;
    }).join("")}</table>`;
  }).join("") + `<p class="hint"><i class="dot estimated"></i> estimated: found automatically, for review &middot; <i class="dot present"></i> present: picked by hand &middot; <i class="dot derived"></i> derived from other points</p>`;
  for (const tr of host.querySelectorAll("tr[data-lm]")) {
    tr.onmouseenter = () => hover(tr.dataset.lm);
    tr.onmouseleave = () => hover(null);
    tr.onclick = () => { const l = viewer.landmarks().find((x) => x.name === tr.dataset.lm); if (l) viewer.focus(l.position, 0.22); };
  }
}

let hovered = null;
function hover(name) {
  hovered = name; viewer.highlight(name);
  for (const d of el("labels").children) d.classList.toggle("hot", d.dataset.lm === name);
  for (const tr of document.querySelectorAll("tr[data-lm]")) tr.classList.toggle("hot", tr.dataset.lm === name);
}

// Labels follow the landmarks on screen while they are shown.
function landmarkLabels() {
  const host = el("labels"); host.innerHTML = `<svg class="leaders"></svg>`;
  for (const l of viewer.landmarks()) {
    const d = document.createElement("div");
    d.className = "lm-label" + (lmCheck(l) ? " check" : ""); d.dataset.lm = l.name; d.textContent = lmName(l.name);
    d.onmouseenter = () => hover(l.name); d.onmouseleave = () => hover(null);
    host.append(d);
  }
}
// Placed below the point and pushed down past any label already placed, with a leader
// line back to the point, so a crowded plateau stays readable.
viewer.frameHooks.push(() => {
  const host = el("labels");
  const show = Boolean(S.controls.show_landmarks) && host.children.length > 1;
  host.hidden = !show;
  if (!show) return;
  const ls = new Map(viewer.landmarks().map((l) => [l.name, l]));
  const items = [];
  for (const d of host.querySelectorAll(".lm-label")) {
    const l = ls.get(d.dataset.lm), p = l && l.visible ? viewer.project(l.position) : null;
    d.style.display = p ? "" : "none";
    if (p) items.push({ d, p, w: d.offsetWidth || 120, h: d.offsetHeight || 16 });
  }
  items.sort((a, b) => a.p[1] - b.p[1]);
  const placed = [];
  let lines = "";
  for (const it of items) {
    let x = it.p[0] + 12, y = it.p[1] - it.h / 2;
    for (let k = 0; k < 30; k++) {
      const hit = placed.find((r) => x < r.x + r.w && x + it.w > r.x && y < r.y + r.h && y + it.h > r.y);
      if (!hit) break;
      y = hit.y + hit.h + 1;
    }
    placed.push({ x, y, w: it.w, h: it.h });
    it.d.style.transform = `translate(${x}px, ${y}px)`;
    lines += `<line x1="${it.p[0]}" y1="${it.p[1]}" x2="${x}" y2="${y + it.h / 2}"/><circle cx="${it.p[0]}" cy="${it.p[1]}" r="2"/>`;
  }
  host.querySelector("svg").innerHTML = lines;
});

// ---------------------------------------------------------------- readout

function parseReport(lines) {
  const out = []; let cur = null; const all = {};
  for (const line of lines) {
    const i = line.indexOf("|"); const k = i < 0 ? line : line.slice(0, i), v = i < 0 ? "" : line.slice(i + 1);
    if (k === "HEAD") { cur = { head: v, rows: [] }; out.push(cur); }
    else if (k === "CASE" || !line.trim()) continue;
    else if (k === "WARN") (cur || out[0])?.rows.push(["WARN", v]);
    else { cur?.rows.push([k, v]); all[k] = v; }
  }
  return { sections: out, all };
}

function readout() {
  if (!S.entry) return;
  const { sections, all } = parseReport(S.report);
  const e = S.entry, F = e.fusion;
  const tile = (label, v) => `<div><span>${label}</span><b>${v ?? "&ndash;"}</b></div>`;
  const val = (k) => (all[k] || "").replace(/ (mm|deg)$/, ` <small>$1</small>`);
  let h = `<div class="card"><h3>Alignment</h3><div class="big">
    ${tile("Valgus cut", val("Valgus cut"))}${tile("Posterior slope", val("Posterior slope"))}
    ${tile("Femoral flexion", val("Femoral flexion"))}${tile("Insert (Fusion)", `${f1(e.fusion.insert.parameters.Insert_Thickness)} <small>mm</small>`)}</div></div>`;
  const pair = (a, b) => `<td class="num">${(all[a] || "").replace(" mm", "")}</td><td class="num">${(all[b] || "").replace(" mm", "")}</td>`;
  h += `<div class="card"><h3>Resections (mm)</h3><table class="res"><tr><th></th><th class="num">Medial</th><th class="num">Lateral</th></tr>
    <tr><td class="k">Distal femur</td>${pair("Femoral medial", "Femoral lateral")}</tr>
    <tr><td class="k">Posterior femur</td>${pair("Femoral posterior medial", "Femoral posterior lateral")}</tr>
    <tr><td class="k">Proximal tibia</td>${pair("Tibial medial", "Tibial lateral")}</tr>
    <tr><td class="k">Anterior femur</td><td class="num" colspan="2">${(all["Femoral anterior"] || "").replace(" mm", "")}</td></tr>
    <tr><td class="k">Gap, extension</td>${pair("Gap medial", "Gap lateral")}</tr></table></div>`;
  const P = (part, k) => {
    const c = S.derived?.changed?.[`${part}.${k}`];
    return c ? `${f2(c[0])} <span class="changed-val">&rarr; ${f2(c[1])}</span>` : f2(F[part].parameters[k]);
  };
  h += `<div class="card"><h3>Implant, built in Fusion</h3><table>
    <tr><td class="k">Femoral width (Femur_ML)</td><td class="num">${P("femoral", "Femur_ML")}</td></tr>
    <tr><td class="k">Femoral box C / B / D</td><td class="num">${P("femoral", "C_Distal_cut_surface")} / ${P("femoral", "B_Posterior_chamfer_projection")} / ${P("femoral", "D_Anterior_chamfer_projection")}</td></tr>
    <tr><td class="k">Tray ML &times; AP</td><td class="num">${P("plate", "Tibial_Plateau_ML")} &times; ${P("plate", "Tibial_Plateau_AP")}</td></tr>
    <tr><td class="k">Insert thickness</td><td class="num">${P("insert", "Insert_Thickness")}</td></tr>
    <tr><td class="k">Catalogue equivalent</td><td class="num">${e.anatomy?.nearest_catalogue_size ?? ""}</td></tr></table>
    ${Object.entries(e.checks).map(([k, [v, w]]) => `<div class="mono" style="font-size:11px">${k}: <b>${v}</b> <span style="color:var(--dim)">${w}</span></div>`).join("")}
    ${["femoral", "plate", "insert"].map((p) => `<div style="margin-top:3px">${{ femoral: "Femoral", plate: "Plate", insert: "Insert" }[p]} <span class="mono">v${F[p].version ?? "?"}</span> ${health(F[p].health)}</div>`).join("")}</div>`;
  const skip = new Set(["Correction", "Resection depths", "Insert and extension gap"]);
  for (const s of sections) {
    if (skip.has(s.head)) continue;
    h += `<div class="card"><h3>${s.head}</h3><table>${s.rows.map(([k, v]) => k === "WARN" ? `<tr><td colspan="2" class="warn">${v}</td></tr>` : `<tr><td class="k">${k}</td><td class="num">${v}</td></tr>`).join("")}</table></div>`;
  }
  el("readout").innerHTML = h;
}

function health(h) {
  const real = (h || []).filter((x) => x[1] !== 3);
  if (!real.length) return `<span class="tag ok">healthy</span>`;
  return `<span class="tag warn" title="${real.map((x) => x[0] + ": " + (x[2] || "")).join("\n")}">${real.length} warning${real.length > 1 ? "s" : ""}</span>`;
}

// ---------------------------------------------------------------- figures (SVG)

function svgFrame(w, h, body) { return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet">${body}</svg>`; }
const path = (loops, tx) => loops.map((L) => "M" + L.map((p) => tx(p).map((v) => v.toFixed(1)).join(",")).join("L") + "Z").join("");

// The J-curves the figure draws: the bone's from the current plan (the live derivation)
// when there is one, the component's from the parts as built.
function jcurves() {
  const built = S.entry.jcurve, live = S.derived?.natural;
  if (!live) return built;
  const condyles = {};
  for (const [k, v] of Object.entries(built.condyles)) {
    const n = live.condyles[k] || {};
    condyles[k] = { ...v, bone: n.bone ?? v.bone, arcs: { bone: n.arcs?.bone ?? v.arcs?.bone, component: v.arcs?.component } };
  }
  return { ...built, box: live.box ?? built.box, condyles };
}

// Sagittal J-curve: CAD frame, +X anterior to the right, +Z proximal up.
function jcurveSVG(j, condyle, { w = 380, h = 330, labels = true } = {}) {
  const c = j.condyles[condyle]; if (!c) return "";
  const [x0, x1, z0, z1] = j.window, k = Math.min(w / (x1 - x0), h / (z1 - z0));
  const tx = ([x, z]) => [(x - x0) * k, h - (z - z0) * k];
  let b = `<rect width="${w}" height="${h}" fill="#fff"/>`;
  // Bone filled; the implant over it; then the natural outline on top, so the two
  // J-curves (natural, brown; implant, blue) read as two lines.
  b += `<path d="${path(c.bone, tx)}" fill="#efe8d8" stroke="none" fill-rule="evenodd"/>`;
  b += `<path d="M${j.box.map((p) => tx([p[0], p[1]]).map((v) => v.toFixed(1)).join(",")).join("L")}Z" fill="none" stroke="#005aaa" stroke-width="1.2" stroke-dasharray="6 4"/>`;
  if (c.component?.length) b += `<path d="${path(c.component, tx)}" fill="#005aaa" fill-opacity=".16" stroke="#005aaa" stroke-width="2.4" fill-rule="evenodd"/>`;
  b += `<path d="${path(c.bone, tx)}" fill="none" stroke="#8a5a1c" stroke-width="2" fill-rule="evenodd"/>`;
  // The fitted arcs (distal, posterior) and their centres; the posterior centre is the
  // condyle's flexion facet centre.
  for (const [who, colour] of [["bone", "#8a5a1c"], ["component", "#005aaa"]]) {
    const arcs = c.arcs?.[who]; if (!arcs) continue;
    for (const [part, a] of Object.entries(arcs)) {
      const [cx, cy] = tx(a.centre), r = a.radius_mm * k;
      b += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${colour}" stroke-width="1" stroke-dasharray="2 3" opacity=".75"/>`;
      b += `<circle cx="${cx}" cy="${cy}" r="${part === "posterior" ? 3.5 : 2.5}" fill="${part === "posterior" ? colour : "#fff"}" stroke="${colour}"/>`;
    }
  }
  const [ox, oz] = tx([0, 0]);
  b += `<line x1="${ox - 6}" y1="${oz}" x2="${ox + 6}" y2="${oz}" stroke="#e07b00"/><line x1="${ox}" y1="${oz - 6}" x2="${ox}" y2="${oz + 6}" stroke="#e07b00"/>`;
  if (labels) b += `<text x="${w - 8}" y="18" text-anchor="end" font-size="13" fill="#5d6b7a" font-family="Verdana">ANT &rarr;</text><text x="8" y="18" font-size="13" fill="#5d6b7a" font-family="Verdana">&uarr; PROX</text>`;
  return svgFrame(w, h, b);
}

// Every number of the J-curve, bone against implant, for one condyle.
function jcurveTable(j, condyle) {
  const c = j.condyles[condyle] || {}, ab = c.arcs?.bone || {}, ac = c.arcs?.component || {}, cu = c.cuts || {};
  const P = j.params || {};
  const r = (a) => a?.radius_mm, d = (a, b) => (a == null || b == null ? null : b - a);
  const ffc = (a) => (a ? `${f1(a.centre[0])}, ${f1(a.centre[1])}` : "&ndash;");
  const row = (k, a, b, delta, note = "") => `<tr><td class="k">${k}${note ? `<div class="from">${note}</div>` : ""}</td><td class="num">${a}</td><td class="num">${b}</td><td class="num">${delta}</td></tr>`;
  return `<table class="jt"><tr><th></th><th class="num">Bone</th><th class="num">Implant</th><th class="num">&Delta;</th></tr>
    ${row("Distal radius", f1(r(ab.distal)), f1(r(ac.distal)), sgn(d(r(ab.distal), r(ac.distal))), "circle fitted under the distal cut")}
    ${row("Posterior radius", f1(r(ab.posterior)), f1(r(ac.posterior)), sgn(d(r(ab.posterior), r(ac.posterior))), "circle fitted on the posterior condyle")}
    ${row("Flexion facet centre", ffc(ab.posterior), ffc(ac.posterior), "", "posterior circle's centre, x, z (mm)")}
    ${row("Distal: off / back", f1(cu.distal?.bone_removed_mm), f1(cu.distal?.component_mm), sgn(cu.distal?.joint_line_change_mm), "bone cut off, metal put back; &Delta; joint line")}
    ${row("Posterior: off / back", f1(cu.posterior?.bone_removed_mm), f1(cu.posterior?.component_mm), sgn(cu.posterior?.offset_change_mm), "&Delta; posterior condylar offset")}
    </table>
    <table class="jt"><tr><th colspan="2">Femoral sagittal parameters</th></tr>
    <tr><td class="k">C distal cut / B post. chamfer / D ant. chamfer</td><td class="num">${f2(P.C_Distal_cut_surface)} / ${f2(P.B_Posterior_chamfer_projection)} / ${f2(P.D_Anterior_chamfer_projection)}</td></tr>
    <tr><td class="k">Sagittal_Scale (C+B+D)/52.1</td><td class="num">${f2((P.C_Distal_cut_surface + P.B_Posterior_chamfer_projection + P.D_Anterior_chamfer_projection) / 52.1)}</td></tr>
    <tr><td class="k">Distal thickness 9 &times; ML/82</td><td class="num">${f2(P.distal_thickness)}</td></tr>
    <tr><td class="k">Section through the condyle centre</td><td class="num">y ${f1(c.y_mm)}</td></tr></table>`;
}

// Tibial cut from above: screen right = -X, screen down = +Y (posterior): anterior up.
function traySVG(t, side, { w = 380, h = 330 } = {}) {
  if (!t) return "";
  const all = [...t.outline_mm, ...(t.plate_outline_mm || [])];
  const xs = all.map((p) => -p[0]), ys = all.map((p) => p[1]);
  const pad = 14, mnx = Math.min(...xs), mxx = Math.max(...xs), mny = Math.min(...ys), mxy = Math.max(...ys);
  const k = Math.min((w - 2 * pad) / (mxx - mnx), (h - 2 * pad - 28) / (mxy - mny));
  const tx = ([x, y]) => [pad + (-x - mnx) * k, pad + 14 + (y - mny) * k];
  let b = `<rect width="${w}" height="${h}" fill="#fff"/>`;
  b += `<path d="${path([t.outline_mm], tx)}" fill="#c7b48f" fill-opacity=".55" stroke="#8a7a5c"/>`;
  if (t.plate_outline_mm) b += `<path d="${path([t.plate_outline_mm], tx)}" fill="#005aaa" fill-opacity=".16" stroke="#005aaa" stroke-width="2"/>`;
  const [ox, oy] = tx([0, 0]);
  b += `<line x1="${ox - 6}" y1="${oy}" x2="${ox + 6}" y2="${oy}" stroke="#e07b00"/><line x1="${ox}" y1="${oy - 6}" x2="${ox}" y2="${oy + 6}" stroke="#e07b00"/>`;
  const medialLeft = side === "right";   // +X (patient-left) is medial on a right knee; +X draws to the left
  b += `<text x="${w / 2}" y="14" text-anchor="middle" font-size="13" fill="#5d6b7a" font-family="Verdana">ANTERIOR</text>`;
  b += `<text x="4" y="${h / 2}" font-size="13" fill="#5d6b7a" font-family="Verdana">${medialLeft ? "MED" : "LAT"}</text><text x="${w - 4}" y="${h / 2}" text-anchor="end" font-size="13" fill="#5d6b7a" font-family="Verdana">${medialLeft ? "LAT" : "MED"}</text>`;
  b += `<text x="8" y="${h - 8}" font-size="13" font-family="IBM Plex Mono" fill="#1d2733">coverage ${(t.coverage * 100).toFixed(0)}% &middot; cut ${t.bone_area_mm2.toFixed(0)} &middot; tray ${t.tray_area_mm2.toFixed(0)} mm²</text>`;
  return svgFrame(w, h, b);
}

function legend() {
  return `<div class="legend"><span><i style="background:#8a5a1c"></i>natural J-curve (bone)</span><span><i style="background:#005aaa"></i>implant J-curve (Fusion)</span><span><i style="background:repeating-linear-gradient(90deg,#005aaa 0 5px,transparent 5px 8px)"></i>planned cuts</span><span><i style="background:repeating-linear-gradient(90deg,#5d6b7a 0 2px,transparent 2px 5px)"></i>fitted arcs (filled dot: flexion facet centre)</span></div>`;
}

function figures() {
  const j = jcurves(), tray = S.derived?.job?.tray_rule ?? S.entry.tray_rule, side = S.knee.side;
  const changed = Object.keys(S.derived?.changed || {}).length;
  el("figures").innerHTML = `<figure class="jfig"><figcaption><span>J-curve: sagittal section through the
      <span class="seg">${["medial", "lateral"].map((c) => `<button data-c="${c}" class="${S.condyle === c ? "on" : ""}">${c}</button>`).join("")}</span> condyle</span>
      <span class="mono">${changed ? "implant as built; the plan has moved: rebuild to update" : "implant as built"}</span></figcaption>
      <div class="jbody"><div class="jsvg">${jcurveSVG(j, S.condyle, { w: 420, h: 400 })}${legend()}</div><div class="jtab">${jcurveTable(j, S.condyle)}</div></div></figure>
    <figure><figcaption><span>Tray on the tibial cut (from above)</span><span class="mono">1 mm inset, no overhang</span></figcaption>${traySVG(tray, side, { w: 420, h: 400 })}</figure>`;
  for (const b of el("figures").querySelectorAll(".seg button")) b.onclick = () => { S.condyle = b.dataset.c; figures(); };
  el("drawer").hidden = false;
}

// ---------------------------------------------------------------- Implant step

function renderImplant() {
  const e = S.entry, d = S.derived, job = d?.job;
  const fem = job?.femoral ?? e.fusion.femoral.parameters, pla = job?.plate ?? e.fusion.plate.parameters, ins = job?.insert ?? e.fusion.insert.parameters;
  const ff = job?.femoral_fit ?? e.femoral_fit, tr = job?.tray_rule ?? e.tray_rule, plan = job?.plan ?? e.plan;
  const ch = d?.changed || {};
  const cv = (part, k, v) => { const c = ch[`${part}.${k}`]; return c ? `<span class="changed-val" title="built ${c[0]}">${f2(v)}</span>` : f2(v); };
  const row = (k, v, from) => `<tr><td class="k">${k}<div class="from">${from}</div></td><td class="num">${v}</td></tr>`;
  let h = d ? `<div class="flag">Derived from the current plan in ${d.seconds} s. ${Object.keys(ch).length ? `${Object.keys(ch).length} parameter(s) differ from the parts built (orange). Build them in step 5.` : "Same as the parts built."}</div>` : `<div class="hint">Showing the parameters the parts were built with.</div>`;
  h += `<div class="chain"><h3>Femoral component</h3>
    <table>${row("Femur_ML", cv("femoral", "Femur_ML", fem.Femur_ML), `bone at the cuts: ${f1(ff?.band_mm?.[1] - ff?.band_mm?.[0])} mm &minus; 2 &times; 1 mm inset + 1 (chart said ${f1(ff?.chart_femur_ml_mm)})`)}
      ${row("C distal cut", cv("femoral", "C_Distal_cut_surface", fem.C_Distal_cut_surface), "planned cut box on the bone")}
      ${row("B post. chamfer", cv("femoral", "B_Posterior_chamfer_projection", fem.B_Posterior_chamfer_projection), "planned cut box")}
      ${row("D ant. chamfer", cv("femoral", "D_Anterior_chamfer_projection", fem.D_Anterior_chamfer_projection), "planned cut box")}
      ${row("Sagittal_Scale", f2((mm(fem.C_Distal_cut_surface) + mm(fem.B_Posterior_chamfer_projection) + mm(fem.D_Anterior_chamfer_projection)) / 52.1), "(C+B+D)/52.1: J-curve radii, thicknesses")}
      ${row("ML slide", f2(ff?.ml_shift_mm), "centred on the cut surfaces")}</table></div>`;
  h += `<div class="chain"><h3>Tibial plate</h3>
    <table>${row("Tibial_Plateau_ML", cv("plate", "Tibial_Plateau_ML", pla.Tibial_Plateau_ML), "largest two-circle outline 1 mm inside the cut")}
      ${row("Tibial_Plateau_AP", cv("plate", "Tibial_Plateau_AP", pla.Tibial_Plateau_AP), "condyle &Oslash; + 4 (design)")}
      ${row("Condyles +X / &minus;X", `${cv("plate", "Left_Condyle_Diameter", pla.Left_Condyle_Diameter)} / ${cv("plate", "Right_Condyle_Diameter", pla.Right_Condyle_Diameter)}`, "each compartment on its own")}
      ${row("Tray_Offset X / Y", `${cv("plate", "Tray_Offset_X", pla.Tray_Offset_X)} / ${cv("plate", "Tray_Offset_Y", pla.Tray_Offset_Y)}`, "outline centred on the cut")}</table></div>`;
  h += `<div class="chain"><h3>Insert</h3><table>
      ${row("Insert_Thickness", cv("insert", "Insert_Thickness", ins.Insert_Thickness), `femoral origin ${f1(plan.h_mm)} above the cut &minus; 5 plate &minus; ${f1(plan.distal_mm)} distal`)}
      ${row("Outline", "plate pocket", "section of this knee's plate, &minus; 0.6 clearance")}
      ${row("Articular surface", "femoral J-curve", "swept from this knee's femoral component")}</table>
      ${plan.insert_below_min ? `<div class="flag">Below 6 mm: +${f1(plan.extra_tibial_resection_for_min_mm)} mm of tibial resection would reach 6 (Plan &rsaquo; Tibial cut &rsaquo; Resection).</div>` : ""}</div>`;
  el("derivation").innerHTML = h;
}

async function derive() {
  await busy("Deriving the implant from the current plan", async () => {
    S.derived = await api("POST", `/api/sessions/${S.session}/parametric`, { name: S.knee.name });
    renderImplant(); figures();
  });
}

// ---------------------------------------------------------------- Reduce step

async function loadKinematics() {
  if (S.stale) return;
  try {
    S.kin = await api("POST", `/api/sessions/${S.session}/kinematics`, { max_deg: 120, steps: 60, model: el("motion").value });
  } catch (e) { S.kin = null; el("kin-charts").innerHTML = `<div class="flag">${e.message}</div>`; return; }
  charts();
}

function kinNow() {
  const k = S.kinNow; if (!k) { el("kin-now").innerHTML = ""; return; }
  el("kin-now").innerHTML = `<div class="big">
    <div><span>Flexion</span><b>${f1(k.flexion_deg)} <small>deg</small></b></div>
    <div><span>Femoral rotation on tibia</span><b>${sgn(k.rotation_deg)} <small>deg ext.</small></b></div>
    <div><span>Medial contact, AP</span><b>${sgn(k.medial_ap_mm)} <small>mm</small></b></div>
    <div><span>Lateral contact, AP</span><b>${sgn(k.lateral_ap_mm)} <small>mm</small></b></div></div>
    <div class="hint" style="margin-top:4px">Lift-off of the other condyle: medial ${f1(k.lift_mm?.medial)}, lateral ${f1(k.lift_mm?.lateral)} mm. AP: anterior +, from extension.</div>`;
  charts();
}

// Two small charts: the femur's axial rotation (the model's, and the figure's), and each
// condyle's AP translation, with the current angle marked.
function chart({ w = 268, h = 140, series, ylabel, marker }) {
  const pad = { l: 30, r: 8, t: 10, b: 22 };
  const ys = series.flatMap((s) => s.pts.map((p) => p[1]));
  let lo = Math.min(0, ...ys), hi = Math.max(0, ...ys);
  const span = Math.max(4, hi - lo); lo -= span * 0.08; hi += span * 0.08;
  const X = (x) => pad.l + (x / 120) * (w - pad.l - pad.r), Y = (y) => pad.t + (1 - (y - lo) / (hi - lo)) * (h - pad.t - pad.b);
  let b = `<rect width="${w}" height="${h}" fill="#fff"/>`;
  for (let x = 0; x <= 120; x += 30) b += `<line x1="${X(x)}" x2="${X(x)}" y1="${pad.t}" y2="${h - pad.b}" stroke="#edf0f4"/><text x="${X(x)}" y="${h - 8}" font-size="9" text-anchor="middle" fill="#5d6b7a">${x}&deg;</text>`;
  const step = span > 12 ? 5 : 2;
  for (let y = Math.ceil(lo / step) * step; y <= hi; y += step) b += `<line x1="${pad.l}" x2="${w - pad.r}" y1="${Y(y)}" y2="${Y(y)}" stroke="${y === 0 ? "#c9d1db" : "#f3f5f8"}"/><text x="${pad.l - 4}" y="${Y(y) + 3}" font-size="9" text-anchor="end" fill="#5d6b7a">${y}</text>`;
  for (const s of series) {
    if (s.dots) b += s.pts.map((p) => `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="2.4" fill="${s.colour}"/>`).join("");
    else b += `<polyline points="${s.pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ")}" fill="none" stroke="${s.colour}" stroke-width="2" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""}/>`;
  }
  if (marker != null) b += `<line x1="${X(marker)}" x2="${X(marker)}" y1="${pad.t}" y2="${h - pad.b}" stroke="#e07b00" stroke-width="1.5"/>`;
  b += `<text x="${w - pad.r}" y="${pad.t + 9}" font-size="9" text-anchor="end" fill="#5d6b7a">${ylabel}</text>`;
  return svgFrame(w, h, b);
}

function charts() {
  if (!S.kin) { el("kin-charts").innerHTML = ""; return; }
  const fr = S.kin.frames, P = S.kin.published, at = S.kinNow?.flexion_deg ?? null;
  const rot0 = P.femur_tibia_rotation_deg[0];
  const pub = P.flexion_deg.map((f, i) => [f, P.femur_tibia_rotation_deg[i] - rot0]);
  const mp = P.flexion_deg.map((f, i) => [f, P.medial_ap_mm[i] - P.medial_ap_mm[0]]);
  const lp = P.flexion_deg.map((f, i) => [f, P.lateral_ap_mm[i] - P.lateral_ap_mm[0]]);
  el("kin-charts").innerHTML = `<h3>Femoral rotation on the tibia (deg, external +)</h3>
    <div class="mini">${chart({ series: [{ pts: pub, colour: "#9aa6b2", dots: true }, { pts: fr.map((f) => [f.flexion_deg, f.rotation_deg]), colour: "#005aaa" }], ylabel: "deg", marker: at })}</div>
    <div class="legend"><span><i style="background:#005aaa"></i>this knee, on its parts</span><span><i style="background:#9aa6b2"></i>published (Fig. 7), from extension</span></div>
    <h3>Contact AP translation (mm, anterior +)</h3>
    <div class="mini">${chart({ series: [{ pts: mp, colour: "#9aa6b2", dots: true }, { pts: lp, colour: "#c9b8a4", dots: true },
      { pts: fr.map((f) => [f.flexion_deg, f.medial_ap_mm]), colour: "#005aaa" }, { pts: fr.map((f) => [f.flexion_deg, f.lateral_ap_mm]), colour: "#e07b00", dash: "5 3" }], ylabel: "mm", marker: at })}</div>
    <div class="legend"><span><i style="background:#005aaa"></i>medial</span><span><i style="background:#e07b00"></i>lateral</span><span><i style="background:#9aa6b2"></i>published (Fig. 8)</span></div>
    <p class="hint">Model: the tibia keeps contact with the lower condyle of the built femoral component at every angle (its J-curve), and each condyle's contact moves back by the profile's translation; the rotation follows from the two and this knee's condyle spacing. ${S.kin.model === "fixed" ? "Profile off: contacts stay put, the femur only rolls." : `Profile: ${P.source}.`}</p>`;
}

async function playArc() {
  if (S.playing) { S.playing = false; return; }
  if (S.stale) { implantStatus(); return; }
  if (!S.kin) await loadKinematics();
  if (!S.kin) return;
  S.playing = true; el("play").textContent = "Stop";
  const fr = S.kin.frames;
  for (const f of [...fr, ...fr.slice().reverse()]) {
    if (!S.playing) break;
    viewer.setPose("tibial", f.pose);
    S.kinNow = f; kinNow();
    await new Promise((r) => setTimeout(r, 40));
  }
  S.playing = false; el("play").textContent = "Play flexion 0–120°";
  await trial({ flexion_deg: S.trial.flexion_deg ?? 0 });
}

// ---------------------------------------------------------------- Build step

function renderBuild() {
  const e = S.entry;
  el("build").disabled = !S.derived;
  const n = Object.keys(S.derived?.changed || {}).length;
  el("build").textContent = n ? `Build this plan in Fusion (${n} parameter${n > 1 ? "s" : ""} changed)` : "Rebuild in Fusion (no change)";
  el("build").title = S.derived ? "" : "Derive the implant from the plan first (step 3).";
  el("fusion").innerHTML = ["femoral", "plate", "insert"].map((p) => {
    const f = e.fusion[p];
    return `<details ${p === "femoral" ? "open" : ""}><summary><b>${{ femoral: "Femoral component", plate: "Tibial plate", insert: "Insert" }[p]}</b> <span class="mono">v${f.version ?? "?"}</span> ${health(f.health)}</summary>
      <div class="hint">${f.folder ?? ""}</div><table>${Object.entries(f.parameters).map(([k, v]) => `<tr><td class="k">${k}</td><td class="num">${f2(v)}</td></tr>`).join("")}</table></details>`;
  }).join("");
}

async function build() {
  try {
    await api("POST", `/api/sessions/${S.session}/build`, {});
  } catch (err) { flag(err.message); return; }
  el("build").disabled = true;
  const poll = async () => {
    const s = await api("GET", "/api/demo/build");
    el("buildlog").textContent = s.log.join("\n");
    el("state").textContent = `Fusion: ${s.state}`;
    if (s.state === "running") return setTimeout(poll, 2000);
    if (s.state === "done") {
      S.entry = await api("GET", `/api/demo/knees/${S.knee.name}`);
      await viewer.setParts(S.entry.parts);
      S.builtControls = { ...S.controls };
      S.derived = null;
      await followPlan(); readout(); renderBuild(); liveDerive();
    }
    el("build").disabled = false;
  };
  poll();
}

// ---------------------------------------------------------------- steps, views

async function setStep(step) {
  S.step = step;
  for (const b of document.querySelectorAll("#steps button")) b.classList.toggle("on", b.dataset.step === step);
  for (const p of document.querySelectorAll("[data-panel]")) p.hidden = p.dataset.panel !== step;
  for (const s of document.querySelectorAll("#pipeline [data-s]")) s.classList.toggle("now", s.dataset.s === step);
  viewer.setClipping(step === "plan");
  el("drawer").hidden = true;
  if (!S.entry) return;
  preset(step);
  if (step !== "reduce" && (S.trial.flexion_deg || S.trial.varus_valgus_deg || S.trial.drawer_ap_mm || S.trial.distraction_mm)) {
    await apply(await api("POST", `/api/sessions/${S.session}/trial/reset`, {}));
  }
  if (step === "anatomy") landmarkList();
  if (step === "implant") { renderImplant(); figures(); }
  if (step === "build") renderBuild();
  if (step === "reduce") { setView("lateral"); if (!S.kin) loadKinematics(); else charts(); kinNow(); }
  implantStatus();
}

function setView(v) {
  S.view = v;
  for (const b of document.querySelectorAll("#views button")) b.classList.toggle("on", b.dataset.view === v);
  if (!S.entry) return;
  const F = S.entry.poses.femoral, T = S.entry.poses.tibial;
  const c = [0, 1, 2].map((i) => (F[i][3] + T[i][3]) / 2);
  viewer.view(v, c, S.knee.side, v === "iso" ? 0.62 : 0.52);
}

function collapse(side) {
  const m = el("main"), cls = side === "left" ? "no-left" : "no-right";
  m.classList.toggle(cls);
  const hidden = m.classList.contains(cls), b = el(`toggle-${side}`);
  b.innerHTML = (side === "left") === hidden ? "&rsaquo;" : "&lsaquo;";
  b.title = `${hidden ? "Show" : "Hide"} the ${side === "left" ? "side panel" : "readout"}`;
}

// ---------------------------------------------------------------- wiring

for (const b of document.querySelectorAll("#steps button")) b.onclick = () => setStep(b.dataset.step);
for (const b of document.querySelectorAll("[data-go]")) b.onclick = () => setStep(b.dataset.go);
for (const b of document.querySelectorAll("#views button")) b.onclick = () => setView(b.dataset.view);
el("commit").onclick = commit;
el("reset").onclick = () => busy("Resetting", async () => {
  await apply(await api("POST", `/api/sessions/${S.session}/controls/reset`));
  if (Object.keys(S.knee.plan_controls || {}).length)   // back to the plan the parts were built for
    await apply(await api("POST", `/api/sessions/${S.session}/controls`, { changes: S.knee.plan_controls }));
  await followPlan(); liveDerive();
});
el("derive").onclick = derive;
el("open-figs").onclick = figures;
el("build").onclick = build;
el("play").onclick = playArc;
el("motion").onchange = async () => { S.kin = null; await trial({ flexion_deg: S.trial.flexion_deg ?? 0 }); loadKinematics(); };
el("reset-trial").onclick = async () => apply(await api("POST", `/api/sessions/${S.session}/trial/reset`, { model: el("motion").value }));
el("drawer-close").onclick = () => { el("drawer").hidden = true; };
el("knee").onclick = intro;
el("toggle-left").onclick = () => collapse("left");
el("toggle-right").onclick = () => collapse("right");
el("sp-toggle").onclick = () => {
  const p = el("scene-panel"); p.classList.toggle("collapsed");
  el("sp-toggle").innerHTML = p.classList.contains("collapsed") ? "+" : "&minus;";
};
el("intro").onclick = (e) => { if (e.target === el("intro") && S.entry) el("intro").hidden = true; };

// The server from the page: restart (new code, sessions dropped) or stop.
function serverNote(html) {
  let n = el("server-note");
  if (!n) { n = document.createElement("div"); n.id = "server-note"; document.body.append(n); }
  n.innerHTML = html;
}
async function serverControl(action) {
  const msg = action === "restart" ? "Restart the server? Open sessions are dropped; the knee reopens on the new server."
                                   : "Stop the server? The page stops working until it is started again.";
  if (!confirm(msg)) return;
  try { await api("POST", `/api/demo/server/${action}`); } catch (e) { flag(e.message); return; }
  if (action === "stop") {
    serverNote(`Server stopped.<small>Start it again with: python demo_server.py --port ${location.port || 80}</small>`);
    return;
  }
  serverNote("Restarting the server&hellip;");
  await new Promise((r) => setTimeout(r, 1500));
  for (let i = 0; i < 120; i++) {
    try { const r = await fetch("/api/demo/knees", { cache: "no-store" }); if (r.ok) { location.reload(); return; } } catch (e) { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 500));
  }
  serverNote("The server did not come back.<small>Start it by hand: python demo_server.py</small>");
}
el("srv-restart").onclick = () => serverControl("restart");
el("srv-stop").onclick = () => serverControl("stop");

window.__demo = { S, viewer, setStep, setView, openKnee };   // for scripted screenshots

(async function start() {
  S.knees = (await api("GET", "/api/demo/knees")).knees;
  const q = new URLSearchParams(location.search);
  if (q.get("step")) S.step = q.get("step");
  if (q.get("view")) S.view = q.get("view");
  if (q.get("knee") && S.knees.some((k) => k.name === q.get("knee"))) await openKnee(q.get("knee"));
  else intro();
})();
