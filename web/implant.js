// Anatomy, Implant, Build and the motion charts of Reduce: the parts of the one app that
// came from the parametric demo. Like the rest of the viewer it computes nothing
// clinical: the landmarks and their roles, the derived parameters and their rules, the
// J-curves, the tray outline and the kinematic readings all come from the server, and
// this file lays them out and draws them.

const NS = "http://www.w3.org/2000/svg";
const num = (v) => (typeof v === "string" ? parseFloat(v) : v);
const f1 = (v) => (v == null || Number.isNaN(num(v)) ? "–" : num(v).toFixed(1));
const f2 = (v) => (v == null || Number.isNaN(num(v)) ? "–" : num(v).toFixed(2));
const sgn = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const svg = (w, h, body) => `<svg xmlns="${NS}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet">${body}</svg>`;
const path = (loops, tx) => loops.map((L) => "M" + L.map((p) => tx(p).map((v) => v.toFixed(1)).join(",")).join("L") + "Z").join("");

export class Implant {
  constructor({ api, el, state, viewer, note, busy, applyEnvelope }) {
    Object.assign(this, { api, el, state, viewer, note, busy, applyEnvelope });
    this.derived = null;
    this.kin = null;
    this.kinNowInfo = null;
    this.condyle = "medial";
    this.polling = false;
    el("derive").addEventListener("click", () => busy(() => this.enterImplant()));
    el("figures-open").addEventListener("click", () => this.figures());
    el("figures-close").addEventListener("click", () => { el("figures").hidden = true; });
    el("build").addEventListener("click", () => this.build());
    el("motion").addEventListener("change", () => this.motionChanged());
    el("anatomy-show").addEventListener("click", () =>
      this.sendControls({ show_landmarks: true, show_axes: true, show_planes: false }));
  }

  get sid() { return this.state.sessionId; }
  sendControls(changes) { return this.state.sendControls(changes); }
  motion() { return this.el("motion").value; }

  // ------------------------------------------------------------ the implant badge

  onEnvelope(payload) {
    const implant = payload.implant;
    const host = this.el("implant-badge");
    if (!implant) { host.hidden = true; return; }
    host.hidden = false;
    const kind = implant.source !== "built" ? "standin" : implant.stale.length ? "stale" : "built";
    host.className = `implant-badge ${kind}`;
    const v = implant.versions || {};
    let html = `<div class="what">Implant: <b>${esc(implant.label)}</b></div>`;
    if (implant.source === "built") {
      html += `<div class="sub">Fusion femoral v${esc(v.femoral)}, plate v${esc(v.plate)}, insert v${esc(v.insert)}` +
        `${implant.built_at ? ` · built ${esc(implant.built_at)}` : ""}</div>`;
    } else if (implant.built_available) {
      html += `<div class="sub">A Fusion build exists; the stand-in is chosen (Plan › Parts shown).</div>`;
    } else if (!implant.derived && implant.derived_reason) {
      html += `<div class="sub">No derivation: ${esc(implant.derived_reason)}</div>`;
    }
    if (implant.stale.length) {
      html += `<details><summary>${implant.stale.length} parameter${implant.stale.length > 1 ? "s" : ""} differ from the build</summary><table>` +
        implant.stale.map((r) => `<tr><td>${esc(r.part)}.${esc(r.name)}</td><td class="num">${esc(f2(r.built))} → ${esc(f2(r.now))}</td></tr>`).join("") +
        `</table></details>`;
    }
    host.innerHTML = html;
    // The plan or the parts moved: the derivation and the motion shown are of the old
    // ones. A trial pose moves neither.
    const key = JSON.stringify([payload.controls, implant.label, implant.versions]);
    if (key !== this.planKey) {
      this.planKey = key;
      this.derived = null;
      this.kin = null;
    }
  }

  // ------------------------------------------------------------ Anatomy

  async enterAnatomy() {
    if (!this.sid) return;
    const out = await this.api("GET", `/api/sessions/${this.sid}/anatomy`);
    const f = out.facts;
    this.el("anatomy-facts").innerHTML = `<div class="tiles">
      <div><span>Femur in the scan</span><b>${f1(f.femur_in_scan_mm)} <small>mm</small></b></div>
      <div><span>Tibia in the scan</span><b>${f1(f.tibia_in_scan_mm)} <small>mm</small></b></div>
      <div><span>Tibial cut ML × AP</span><b>${f1(f.tibial_cut_ml_mm)} × ${f1(f.tibial_cut_ap_mm)}</b></div>
      <div><span>Nearest catalogue size</span><b>${esc(f.nearest_catalogue_size)}</b></div></div>`;
    const host = this.el("landmark-list");
    host.innerHTML = ["femur", "tibia"].map((bone) => {
      const rows = out.landmarks.filter((l) => l.bone === bone);
      return `<h3>${bone === "femur" ? "Femur" : "Tibia"} · ${rows.length} landmarks</h3><table class="lm">` +
        rows.map((l) => `<tr data-id="${esc(l.id)}"><td><i class="dot ${esc(l.status)}"></i></td>` +
          `<td>${esc(l.name)}<div class="from">${esc(l.role)}</div></td><td class="num tag">${esc(l.status)}</td></tr>`).join("") +
        `</table>`;
    }).join("") + `<p class="study-help"><i class="dot estimated"></i> estimated: found automatically, for review · <i class="dot present"></i> picked by hand · <i class="dot derived"></i> derived from other points</p>`;
    const byId = new Map(out.landmarks.map((l) => [l.id, l]));
    for (const row of host.querySelectorAll("tr[data-id]")) {
      row.addEventListener("click", () => {
        const l = byId.get(row.dataset.id);
        this.viewer.focus({ point_mm: l.position_mm, set: l.bone === "femur" ? "femoral" : "tibial", label: l.name });
        for (const r of host.querySelectorAll("tr.hot")) r.classList.remove("hot");
        row.classList.add("hot");
      });
    }
  }

  // ------------------------------------------------------------ Implant

  async enterImplant() {
    if (!this.sid) return;
    try {
      this.derived = await this.api("POST", `/api/sessions/${this.sid}/parametric`);
    } catch (error) {
      this.derived = null;
      this.el("derivation").innerHTML = `<div class="study-warn">${esc(error.message)}</div>`;
      return;
    }
    this.renderImplant();
  }

  renderImplant() {
    const d = this.derived;
    if (!d) return;
    const changed = Object.keys(d.changed || {}).length;
    const im = d.implant;
    let html = `<div class="flag ${changed ? "warn" : ""}">Derived from the current plan in ${d.seconds} s. ` +
      (im.built_available ? (changed ? `${changed} parameter${changed > 1 ? "s" : ""} differ from the parts built (marked). Build them in Build.`
                                     : "Same as the parts built.")
                          : "No Fusion build for this knee yet: the parts shown are a stand-in.") + `</div>`;
    const m = d.insert_minimum || {};
    if (m.tibial_resection_auto_mm) {
      html += `<div class="flag">+${f1(m.tibial_resection_auto_mm)} mm tibial resection for the ${esc(m.insert_minimum_mm)} mm insert minimum` +
        `${m.insert_minimum_clamped ? " (limit reached: minimum not met)" : ""}.</div>`;
    }
    let part = null;
    for (const r of d.parameters) {
      if (r.part !== part) {
        if (part) html += `</table></div>`;
        part = r.part;
        html += `<div class="chain"><h3>${esc(r.part_label)}</h3><table>`;
      }
      const value = /mm$/.test(String(r.value)) || !Number.isNaN(parseFloat(r.value)) ? f2(r.value) : esc(r.value);
      const mark = r.changed ? `<span class="changed-val" title="built ${esc(f2(r.changed[0]))}">${value}</span>` : value;
      html += `<tr class="${r.changed ? "changed" : ""}"><td class="k">${esc(r.name)}<div class="from">${esc(r.rule)}</div></td><td class="num">${mark}</td></tr>`;
    }
    if (part) html += `</table></div>`;
    if (d.jcurve_note) html += `<div class="study-warn">${esc(d.jcurve_note)}</div>`;
    this.el("derivation").innerHTML = html;
  }

  // The figures: the J-curve through each condyle, and the tray on the tibial cut.
  figures() {
    const d = this.derived;
    if (!d) { this.note("Derive the implant first."); return; }
    const host = this.el("figures-body");
    const j = d.jcurve;
    host.innerHTML = `<figure><figcaption>J-curve: sagittal section through the
        <span class="seg">${["medial", "lateral"].map((c) => `<button type="button" data-c="${c}" class="${this.condyle === c ? "on" : ""}">${c}</button>`).join("")}</span>
        condyle <small>(${esc(d.implant.label)})</small></figcaption>
        ${j ? jcurveSVG(j, this.condyle) + jcurveLegend() + jcurveTable(j, this.condyle) : `<p>${esc(d.jcurve_note)}</p>`}</figure>
      <figure><figcaption>Tray on the tibial cut, from above <small>(1 mm inset, no overhang)</small></figcaption>${traySVG(d.tray_rule, d.side)}</figure>`;
    for (const b of host.querySelectorAll(".seg button")) {
      b.addEventListener("click", () => { this.condyle = b.dataset.c; this.figures(); });
    }
    this.el("figures").hidden = false;
  }

  // ------------------------------------------------------------ Reduce

  async enterReduce() {
    this.el("kin-charts").innerHTML = "";
    if (this.state.stale) return;
    await this.loadKinematics();
  }

  async loadKinematics() {
    const model = this.motion() === "fixed" ? "fixed" : "published";
    try {
      this.kin = await this.api("POST", `/api/sessions/${this.sid}/kinematics`, { max_deg: 120, steps: 60, model });
    } catch (error) {
      this.kin = null;
      this.el("kin-charts").innerHTML = `<div class="study-warn">${esc(error.message)}</div>`;
      return;
    }
    this.charts();
  }

  async motionChanged() {
    this.kin = null;
    const payload = await this.api("POST", `/api/sessions/${this.sid}/trial`, { model: this.motion() });
    await this.applyEnvelope(payload);
    this.kinNow(payload.kinematics);
    await this.loadKinematics();
  }

  async frames() {
    if (!this.kin) await this.loadKinematics();
    return this.kin ? this.kin.frames : null;
  }

  kinNow(info) {
    this.kinNowInfo = info || null;
    const k = this.kinNowInfo;
    this.el("kin-now").innerHTML = !k ? "" : `<div class="tiles">
      <div><span>Flexion</span><b>${f1(k.flexion_deg)} <small>deg</small></b></div>
      <div><span>Femoral rotation on tibia</span><b>${sgn(k.rotation_deg)} <small>deg ext.</small></b></div>
      <div><span>Medial contact, AP</span><b>${sgn(k.medial_ap_mm)} <small>mm</small></b></div>
      <div><span>Lateral contact, AP</span><b>${sgn(k.lateral_ap_mm)} <small>mm</small></b></div></div>
      <div class="study-help">Lift-off of the other condyle: medial ${f1(k.lift_mm?.medial)}, lateral ${f1(k.lift_mm?.lateral)} mm. AP: anterior +, from extension.</div>`;
    this.charts();
  }

  charts() {
    if (!this.kin) { this.el("kin-charts").innerHTML = ""; return; }
    const fr = this.kin.frames, P = this.kin.published, at = this.kinNowInfo?.flexion_deg ?? null;
    const rot0 = P.femur_tibia_rotation_deg[0];
    const pub = P.flexion_deg.map((f, i) => [f, P.femur_tibia_rotation_deg[i] - rot0]);
    const mp = P.flexion_deg.map((f, i) => [f, P.medial_ap_mm[i] - P.medial_ap_mm[0]]);
    const lp = P.flexion_deg.map((f, i) => [f, P.lateral_ap_mm[i] - P.lateral_ap_mm[0]]);
    this.el("kin-charts").innerHTML = `<h3>Femoral rotation on the tibia (deg, external +)</h3>
      ${chart({ series: [{ pts: pub, colour: "#99a3b0", dots: true }, { pts: fr.map((f) => [f.flexion_deg, f.rotation_deg]), colour: "#4c8dff" }], ylabel: "deg", marker: at })}
      <div class="legend-row"><span><i style="background:#4c8dff"></i>this knee, on the parts shown</span><span><i style="background:#99a3b0"></i>published (Fig. 7)</span></div>
      <h3>Contact AP translation (mm, anterior +)</h3>
      ${chart({ series: [{ pts: mp, colour: "#99a3b0", dots: true }, { pts: lp, colour: "#c9b8a4", dots: true },
        { pts: fr.map((f) => [f.flexion_deg, f.medial_ap_mm]), colour: "#4c8dff" },
        { pts: fr.map((f) => [f.flexion_deg, f.lateral_ap_mm]), colour: "#ffb454", dash: "5 3" }], ylabel: "mm", marker: at })}
      <div class="legend-row"><span><i style="background:#4c8dff"></i>medial</span><span><i style="background:#ffb454"></i>lateral</span><span><i style="background:#99a3b0"></i>published (Fig. 8)</span></div>
      <p class="study-help">On ${esc(this.kin.parts)}: the tibia keeps contact with the lower condyle of the femoral component at every angle (its J-curve), and each condyle's contact moves back by the profile's translation. ${this.kin.model === "fixed" ? "Profile off: contacts stay put." : `Profile: ${esc(P.source)}.`}</p>`;
  }

  // ------------------------------------------------------------ Build

  async enterBuild() {
    if (!this.sid) return;
    if (!this.derived) await this.enterImplant();
    const state = await this.api("GET", "/api/build");
    this.renderBuild(state);
  }

  renderBuild(state) {
    const d = this.derived;
    const n = Object.keys(d?.changed || {}).length;
    const button = this.el("build");
    button.disabled = !d || state?.state === "running";
    button.textContent = !d?.implant?.built_available ? "Build this plan in Fusion"
      : n ? `Build this plan in Fusion (${n} parameter${n > 1 ? "s" : ""} changed)` : "Rebuild in Fusion (no change)";
    this.el("build-log").textContent = state?.log?.length ? state.log.join("\n") : "";
    this.el("build-state").textContent = state ? `Fusion build: ${state.state}${state.result?.error ? ` — ${state.result.error}` : ""}` : "";
    const b = d?.build;
    this.el("build-parts").innerHTML = !b ? `<p class="study-help">No build for this knee yet.</p>` :
      ["femoral", "plate", "insert"].map((p) => {
        const health = (b.health[p] || []).filter((h) => h[1] !== 3);
        return `<details><summary><b>${{ femoral: "Femoral component", plate: "Tibial plate", insert: "Insert" }[p]}</b> v${esc(b.versions[p])} ` +
          `<span class="tag ${health.length ? "warn" : "ok"}">${health.length ? `${health.length} warning${health.length > 1 ? "s" : ""}` : "healthy"}</span></summary><table>` +
          Object.entries(b.parameters[p] || {}).map(([k, v]) => `<tr><td class="k">${esc(k)}</td><td class="num">${f2(v)}</td></tr>`).join("") + `</table></details>`;
      }).join("") + `<p class="study-help">Built ${esc(b.built_at)}</p>`;
  }

  async build() {
    try {
      const state = await this.api("POST", `/api/sessions/${this.sid}/build`, {});
      this.renderBuild(state);
    } catch (error) {
      this.el("build-state").textContent = error.message;
      this.note(error.message);
      return;
    }
    if (this.polling) return;
    this.polling = true;
    const poll = async () => {
      const state = await this.api("GET", "/api/build");
      this.renderBuild(state);
      if (state.state === "running") { setTimeout(poll, 2000); return; }
      this.polling = false;
      if (state.state === "done") {
        await this.applyEnvelope(await this.api("POST", `/api/sessions/${this.sid}/implant/refresh`));
        this.derived = null;
        await this.enterBuild();
      }
    };
    setTimeout(poll, 1000);
  }
}

// -------------------------------------------------------------- drawing (no maths)

// Sagittal J-curve: CAD frame, +X anterior to the right, +Z proximal up.
function jcurveSVG(j, condyle, w = 420, h = 380) {
  const c = j.condyles[condyle];
  if (!c) return "";
  const [x0, x1, z0, z1] = j.window, k = Math.min(w / (x1 - x0), h / (z1 - z0));
  const tx = ([x, z]) => [(x - x0) * k, h - (z - z0) * k];
  let b = `<rect width="${w}" height="${h}" fill="#fff"/>`;
  b += `<path d="${path(c.bone, tx)}" fill="#efe8d8" stroke="none" fill-rule="evenodd"/>`;
  b += `<path d="M${j.box.map((p) => tx(p).map((v) => v.toFixed(1)).join(",")).join("L")}Z" fill="none" stroke="#005aaa" stroke-width="1.2" stroke-dasharray="6 4"/>`;
  if (c.component?.length) b += `<path d="${path(c.component, tx)}" fill="#005aaa" fill-opacity=".16" stroke="#005aaa" stroke-width="2.4" fill-rule="evenodd"/>`;
  b += `<path d="${path(c.bone, tx)}" fill="none" stroke="#8a5a1c" stroke-width="2" fill-rule="evenodd"/>`;
  for (const [who, colour] of [["bone", "#8a5a1c"], ["component", "#005aaa"]]) {
    const arcs = c.arcs?.[who];
    if (!arcs) continue;
    for (const [part, a] of Object.entries(arcs)) {
      const [cx, cy] = tx(a.centre), r = a.radius_mm * k;
      b += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${colour}" stroke-width="1" stroke-dasharray="2 3" opacity=".75"/>`;
      b += `<circle cx="${cx}" cy="${cy}" r="${part === "posterior" ? 3.5 : 2.5}" fill="${part === "posterior" ? colour : "#fff"}" stroke="${colour}"/>`;
    }
  }
  const [ox, oz] = tx([0, 0]);
  b += `<line x1="${ox - 6}" y1="${oz}" x2="${ox + 6}" y2="${oz}" stroke="#e07b00"/><line x1="${ox}" y1="${oz - 6}" x2="${ox}" y2="${oz + 6}" stroke="#e07b00"/>`;
  b += `<text x="${w - 8}" y="18" text-anchor="end" font-size="13" fill="#5d6b7a">ANT →</text><text x="8" y="18" font-size="13" fill="#5d6b7a">↑ PROX</text>`;
  return svg(w, h, b);
}

function jcurveLegend() {
  return `<div class="legend-row light"><span><i style="background:#8a5a1c"></i>natural J-curve (bone)</span><span><i style="background:#005aaa"></i>implant J-curve</span><span><i style="background:repeating-linear-gradient(90deg,#005aaa 0 5px,transparent 5px 8px)"></i>planned cuts</span><span><i style="background:repeating-linear-gradient(90deg,#5d6b7a 0 2px,transparent 2px 5px)"></i>fitted arcs</span></div>`;
}

function jcurveTable(j, condyle) {
  const c = j.condyles[condyle] || {}, ab = c.arcs?.bone || {}, ac = c.arcs?.component || {}, cu = c.cuts || {};
  const r = (a) => a?.radius_mm;
  const ffc = (a) => (a ? `${f1(a.centre[0])}, ${f1(a.centre[1])}` : "–");
  const row = (k, a, b, delta, note) => `<tr><td class="k">${k}<div class="from">${note}</div></td><td class="num">${a}</td><td class="num">${b}</td><td class="num">${delta}</td></tr>`;
  // The bone-implant differences are the server's where it gives them (the cuts);
  // the radii are shown side by side.
  return `<table class="jt"><tr><th></th><th class="num">Bone</th><th class="num">Implant</th><th class="num">Δ</th></tr>
    ${row("Distal radius", f1(r(ab.distal)), f1(r(ac.distal)), "", "circle fitted under the distal cut")}
    ${row("Posterior radius", f1(r(ab.posterior)), f1(r(ac.posterior)), "", "circle fitted on the posterior condyle")}
    ${row("Flexion facet centre", ffc(ab.posterior), ffc(ac.posterior), "", "posterior circle's centre, x, z (mm)")}
    ${row("Distal: off / back", f1(cu.distal?.bone_removed_mm), f1(cu.distal?.component_mm), sgn(cu.distal?.joint_line_change_mm), "bone cut off, metal put back; Δ joint line")}
    ${row("Posterior: off / back", f1(cu.posterior?.bone_removed_mm), f1(cu.posterior?.component_mm), sgn(cu.posterior?.offset_change_mm), "Δ posterior condylar offset")}
    </table>`;
}

// Tibial cut from above: screen right = -X, screen down = +Y (posterior): anterior up.
function traySVG(t, side, w = 420, h = 380) {
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
  const medialLeft = side === "right";
  b += `<text x="${w / 2}" y="14" text-anchor="middle" font-size="13" fill="#5d6b7a">ANTERIOR</text>`;
  b += `<text x="4" y="${h / 2}" font-size="13" fill="#5d6b7a">${medialLeft ? "MED" : "LAT"}</text><text x="${w - 4}" y="${h / 2}" text-anchor="end" font-size="13" fill="#5d6b7a">${medialLeft ? "LAT" : "MED"}</text>`;
  b += `<text x="8" y="${h - 8}" font-size="13" fill="#1d2733">coverage ${(t.coverage * 100).toFixed(0)}% · cut ${t.bone_area_mm2.toFixed(0)} · tray ${t.tray_area_mm2.toFixed(0)} mm²</text>`;
  return svg(w, h, b);
}

function chart({ w = 272, h = 140, series, ylabel, marker }) {
  const pad = { l: 30, r: 8, t: 10, b: 22 };
  const ys = series.flatMap((s) => s.pts.map((p) => p[1]));
  let lo = Math.min(0, ...ys), hi = Math.max(0, ...ys);
  const span = Math.max(4, hi - lo); lo -= span * 0.08; hi += span * 0.08;
  const X = (x) => pad.l + (x / 120) * (w - pad.l - pad.r), Y = (y) => pad.t + (1 - (y - lo) / (hi - lo)) * (h - pad.t - pad.b);
  let b = `<rect width="${w}" height="${h}" fill="#1d2126"/>`;
  for (let x = 0; x <= 120; x += 30) b += `<line x1="${X(x)}" x2="${X(x)}" y1="${pad.t}" y2="${h - pad.b}" stroke="#2c323a"/><text x="${X(x)}" y="${h - 8}" font-size="9" text-anchor="middle" fill="#99a3b0">${x}°</text>`;
  const step = span > 12 ? 5 : 2;
  for (let y = Math.ceil(lo / step) * step; y <= hi; y += step) b += `<line x1="${pad.l}" x2="${w - pad.r}" y1="${Y(y)}" y2="${Y(y)}" stroke="${y === 0 ? "#4a525c" : "#262c33"}"/><text x="${pad.l - 4}" y="${Y(y) + 3}" font-size="9" text-anchor="end" fill="#99a3b0">${y}</text>`;
  for (const s of series) {
    if (s.dots) b += s.pts.map((p) => `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="2.4" fill="${s.colour}"/>`).join("");
    else b += `<polyline points="${s.pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ")}" fill="none" stroke="${s.colour}" stroke-width="2" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""}/>`;
  }
  if (marker != null) b += `<line x1="${X(marker)}" x2="${X(marker)}" y1="${pad.t}" y2="${h - pad.b}" stroke="#ff7a59" stroke-width="1.5"/>`;
  b += `<text x="${w - pad.r}" y="${pad.t + 9}" font-size="9" text-anchor="end" fill="#99a3b0">${ylabel}</text>`;
  return `<div class="mini">${svg(w, h, b)}</div>`;
}
