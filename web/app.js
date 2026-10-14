// The controller: it holds the session id, builds the panel, and applies whatever the
// server sends back. It computes nothing clinical. Every number on the screen came from
// the server, which is what keeps the project's promise that the method behind a number
// is auditable rather than reinvented in JavaScript.
//
// The panel is built from a schema the server serves. Ranges like "the femoral cut may
// be flexed between -10 and +15 degrees" are clinical limits, and a viewer that wrote
// its own copy of them would be a second place for one to live.

import { Viewer } from "./viewer.js";

const state = {
  sessionId: null,
  schema: null,
  controls: {},
  trial: {},
  stale: true,
  mode: "plan",
  arc: null,
  playing: false,
};

const el = (id) => document.getElementById(id);
const viewer = new Viewer(el("view"));

// --------------------------------------------------------------------
// Talking to the server
// --------------------------------------------------------------------

async function api(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

// One request in flight at a time, with the latest change queued behind it. A slider
// drag fires far faster than a round trip, and without this the server would answer a
// queue of poses the user has already moved past.
let inFlight = null;
let queued = null;

function sendControls(changes) {
  queued = { ...(queued || {}), ...changes };
  if (inFlight) return inFlight;
  inFlight = (async () => {
    while (queued) {
      const batch = queued;
      queued = null;
      const payload = await api(
        "POST", `/api/sessions/${state.sessionId}/controls`, { changes: batch }
      );
      await applyEnvelope(payload);
    }
    inFlight = null;
  })().catch((error) => {
    inFlight = null;
    queued = null;
    note(`Could not apply the change: ${error.message}`);
  });
  return inFlight;
}

async function sendTrial(changes) {
  const payload = await api(
    "POST", `/api/sessions/${state.sessionId}/trial`, { changes }
  );
  await applyEnvelope(payload);
}

// --------------------------------------------------------------------
// Applying a response
// --------------------------------------------------------------------

async function applyEnvelope(payload) {
  state.controls = payload.controls ?? state.controls;
  state.trial = payload.trial ?? state.trial;
  state.stale = Boolean(payload.stale);

  if (payload.scene) {
    await viewer.setScene(payload.scene, meshUrl);
  } else if (payload.delta) {
    await viewer.applyDelta(payload.delta);
  }
  if (payload.clip) viewer.setClip(payload.clip);

  renderReport(payload.report || []);
  renderFindings(payload.findings || []);
  renderLegend(payload.fit_map);
  renderNotes(payload.delta?.notes || [], payload.commit);
  syncPanel();
  // A change to the plan invalidates the arc the server computed for the old one.
  state.arc = null;
}

const meshUrl = (meshId) => `/api/sessions/${state.sessionId}/meshes/${meshId}.glb`;

// --------------------------------------------------------------------
// The panel
// --------------------------------------------------------------------

function buildPanel() {
  const host = el("controls");
  host.textContent = "";
  for (const group of state.schema.controls) {
    const box = document.createElement("fieldset");
    const legend = document.createElement("legend");
    legend.textContent = group.group;
    box.append(legend);
    for (const control of group.controls) {
      box.append(buildControl(control, () => state.controls[control.name], sendControls));
    }
    host.append(box);
  }

  const trial = el("trial");
  trial.textContent = "";
  const box = document.createElement("fieldset");
  const legend = document.createElement("legend");
  legend.textContent = "Trial reduction";
  box.append(legend);
  for (const control of state.schema.trial) {
    box.append(buildControl(control, () => state.trial[control.name], (changes) =>
      sendTrial(changes)
    ));
  }
  trial.append(box);
}

function buildControl(control, read, send) {
  const wrapper = document.createElement("div");
  wrapper.className = "control";
  wrapper.dataset.name = control.name;
  if (control.help) wrapper.title = control.help;

  const row = document.createElement("div");
  row.className = "row";
  const name = document.createElement("span");
  name.className = "name";
  name.textContent = control.label;
  row.append(name);

  let input;
  if (control.kind === "float") {
    const value = document.createElement("span");
    value.className = "value";
    row.append(value);
    input = document.createElement("input");
    input.type = "range";
    input.min = control.min;
    input.max = control.max;
    input.step = control.step;
    input.addEventListener("input", () => {
      value.textContent = format(control, Number(input.value));
      send({ [control.name]: Number(input.value) });
    });
    // Double-clicking a slider returns it to the middle of nothing useful by default;
    // here it returns the control to the plan's own value, which is what a surgeon
    // reaching for a reset means.
    input.addEventListener("dblclick", () => {
      const home = control.min <= 0 && control.max >= 0 ? 0 : control.min;
      input.value = home;
      value.textContent = format(control, home);
      send({ [control.name]: home });
    });
    wrapper.append(row, input);
    wrapper.sync = () => {
      const current = Number(read() ?? 0);
      input.value = current;
      value.textContent = format(control, current);
      wrapper.classList.toggle("changed", isAdjusted(control, current));
    };
  } else if (control.kind === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.addEventListener("change", () => send({ [control.name]: input.checked }));
    const label = document.createElement("label");
    label.append(input, name);
    wrapper.append(label);
    wrapper.sync = () => { input.checked = Boolean(read()); };
  } else {
    input = document.createElement("select");
    for (const [value, label] of control.choices) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      input.append(option);
    }
    input.addEventListener("change", () => send({ [control.name]: input.value }));
    wrapper.append(row, input);
    wrapper.sync = () => { input.value = read() ?? ""; };
  }

  wrapper.setEnabled = (enabled) => { input.disabled = !enabled; };
  return wrapper;
}

function format(control, value) {
  const places = control.step < 0.5 ? 1 : 1;
  return `${value.toFixed(places)} ${control.unit || ""}`.trim();
}

function isAdjusted(control, value) {
  // A control sitting away from zero is a manual override of what the pipeline
  // computed, and the panel says so. Controls whose neutral value is not zero, such as
  // the tibial resection depth, are excluded: for them zero would be the odd state.
  return control.min < 0 && control.max > 0 && Math.abs(value) > 1e-9;
}

function syncPanel() {
  for (const wrapper of document.querySelectorAll(".control")) {
    wrapper.sync?.();
    const isTrial = wrapper.closest("#reduce-mode") !== null;
    wrapper.setEnabled?.(isTrial ? !state.stale : true);
  }

  el("commit").classList.toggle("stale", state.stale);
  el("commit").textContent = state.stale ? "Commit" : "Committed";
  el("commit").disabled = !state.sessionId;

  const banner = el("banner");
  if (state.mode === "reduce" && state.stale) {
    banner.hidden = false;
    banner.textContent =
      "The geometry no longer matches the plan. Reduce needs committed cuts.";
    const back = document.createElement("button");
    back.type = "button";
    back.textContent = "Back to Plan";
    back.addEventListener("click", () => setMode("plan"));
    banner.append(back);
  } else {
    banner.hidden = true;
    banner.textContent = "";
  }
}

// --------------------------------------------------------------------
// The readout
// --------------------------------------------------------------------

function renderReport(lines) {
  const host = el("report");
  host.textContent = "";
  for (const line of lines) {
    const [key, value] = splitLine(line);
    const row = document.createElement("div");
    if (key === "HEAD") {
      row.className = "head";
      row.textContent = value;
    } else if (key === "WARN") {
      row.className = "warn";
      row.textContent = value;
    } else if (key === "CASE") {
      row.className = "case";
      row.textContent = value;
    } else if (!line.trim()) {
      continue;
    } else {
      row.className = "line";
      const left = document.createElement("span");
      left.textContent = key;
      const right = document.createElement("span");
      right.textContent = value;
      row.append(left, right);
    }
    host.append(row);
  }
}

function splitLine(line) {
  const index = line.indexOf("|");
  return index === -1 ? [line, ""] : [line.slice(0, index), line.slice(index + 1)];
}

// --------------------------------------------------------------------
// Findings and the fit map's legend
// --------------------------------------------------------------------

const SEVERITY_ORDER = { error: 0, warning: 1, info: 2 };

// The server's findings, most severe first, as it worded them. One with a place gets a
// button that turns the view to it.
function renderFindings(findings) {
  const host = el("findings");
  host.textContent = "";
  const head = document.createElement("div");
  head.className = "head";
  const counts = { error: 0, warning: 0, info: 0 };
  for (const finding of findings) counts[finding.severity] = (counts[finding.severity] || 0) + 1;
  head.textContent =
    `Findings: ${counts.error} error, ${counts.warning} warning, ${counts.info} info`;
  host.append(head);

  const sorted = [...findings].sort(
    (a, b) => (SEVERITY_ORDER[a.severity] ?? 3) - (SEVERITY_ORDER[b.severity] ?? 3)
  );
  for (const finding of sorted) {
    const row = document.createElement("details");
    row.className = `finding ${finding.severity}`;
    if (finding.severity !== "info") row.open = finding.severity === "error";
    const summary = document.createElement("summary");
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = finding.severity;
    const code = document.createElement("code");
    code.textContent = finding.code;
    summary.append(badge, code);
    if (finding.location) {
      const show = document.createElement("button");
      show.type = "button";
      show.className = "show";
      show.textContent = "Show";
      show.title = `Turn the view to ${finding.location.label}`;
      show.addEventListener("click", (event) => {
        event.preventDefault();
        viewer.focus(finding.location);
      });
      summary.append(show);
    }
    const message = document.createElement("div");
    message.className = "message";
    message.textContent = finding.message;
    row.append(summary, message);
    host.append(row);
  }
}

// The legend is the server's: its stops, its ends and its words. Only the layout is
// here.
function renderLegend(fitMap) {
  const host = el("legend");
  host.textContent = "";
  if (!fitMap || !fitMap.available || !fitMap.visible) {
    host.hidden = true;
    return;
  }
  host.hidden = false;
  const { legend } = fitMap;
  state.fitLegend = legend;
  const span = legend.max_mm - legend.min_mm;
  const title = document.createElement("div");
  title.className = "title";
  title.textContent = `${legend.title} (${legend.unit})`;
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.style.background = `linear-gradient(to right, ${legend.stops
    .map(([value, [r, g, b]]) =>
      `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}) ` +
      `${(100 * (value - legend.min_mm)) / span}%`)
    .join(", ")})`;
  const ticks = document.createElement("div");
  ticks.className = "ticks";
  for (const [value] of legend.stops) {
    const tick = document.createElement("span");
    tick.style.left = `${(100 * (value - legend.min_mm)) / span}%`;
    tick.textContent = value > 0 ? `+${value}` : `${value}`;
    ticks.append(tick);
  }
  const words = document.createElement("div");
  words.className = "words";
  const left = document.createElement("span");
  left.textContent = legend.negative;
  const right = document.createElement("span");
  right.textContent = legend.positive;
  words.append(left, right);
  const foot = document.createElement("div");
  foot.className = "foot";
  foot.textContent = `${legend.zero} at 0; ${legend.clamped}. ` +
    (legend.resolution_mm ? `Drawn at ${legend.resolution_mm} mm. ` : "") +
    (fitMap.seconds !== null ? `Measured in ${fitMap.seconds} s at Commit.` : "");
  host.append(title, bar, ticks, words, foot);
}

function showValueUnderPointer(event) {
  const readout = el("legend").querySelector(".foot");
  if (!readout || el("legend").hidden) return;
  const hit = viewer.valueAt(event.clientX, event.clientY);
  if (!hit) return;
  const value = hit.value;
  const legend = state.fitLegend;
  const where = hit.name.endsWith("tibial") ? "Tibial cut" : "Distal femoral cut";
  readout.textContent = value === 0 ? `${where}: ${legend.zero}` :
    `${where}: ${value < 0 ? legend.negative : legend.positive} by ` +
    `${Math.abs(value).toFixed(1)} mm`;
}

function renderNotes(notes, commit) {
  const host = el("notes");
  host.textContent = "";
  const lines = [...notes];
  if (commit) {
    if (commit.computed.length) lines.push(`Cut: ${commit.computed.join(", ")}`);
    if (commit.reused.length) lines.push(`Reused from the store: ${commit.reused.join(", ")}`);
    if (commit.unchanged.length) lines.push(`Unchanged: ${commit.unchanged.join(", ")}`);
    for (const [node, record] of Object.entries(commit.records || {})) {
      lines.push(`${node}: ${record.backend}${record.fallback ? " (fell back)" : ""}`);
    }
  }
  for (const line of lines) {
    const row = document.createElement("div");
    row.textContent = line;
    host.append(row);
  }
}

function note(message) {
  const row = document.createElement("div");
  row.textContent = message;
  el("notes").prepend(row);
}

// --------------------------------------------------------------------
// Modes
// --------------------------------------------------------------------

function setMode(mode) {
  state.mode = mode;
  state.playing = false;
  el("plan-mode").hidden = mode !== "plan";
  el("reduce-mode").hidden = mode !== "reduce";
  for (const button of document.querySelectorAll("#modes button")) {
    button.classList.toggle("active", button.dataset.mode === mode);
  }
  // Plan mode previews the cut by clipping. Reduce mode works on geometry that has
  // really been cut, so clipping there would take a second bite out of it.
  viewer.setClipping(mode === "plan");
  syncPanel();
}

async function playArc() {
  if (state.playing) { state.playing = false; return; }
  if (!state.arc) {
    const payload = await api("POST", `/api/sessions/${state.sessionId}/arc`, {
      max_deg: Number(el("arc-max").value) || 120,
      steps: 48,
    });
    state.arc = payload.frames;
  }

  state.playing = true;
  el("play").textContent = "Stop";
  for (const frame of state.arc) {
    if (!state.playing) break;
    viewer.setPose("tibial", frame.pose);
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
  state.playing = false;
  el("play").textContent = "Play arc";
  // The arc is a scripted sweep, not a change of state: it ends where the dialled
  // trial controls left the tibia, not wherever the last frame happened to be.
  await sendTrial({ flexion_deg: state.trial.flexion_deg ?? 0 });
}

// --------------------------------------------------------------------
// Wiring
// --------------------------------------------------------------------

async function busy(work) {
  el("busy").hidden = false;
  try {
    return await work();
  } catch (error) {
    note(error.message);
    return null;
  } finally {
    el("busy").hidden = true;
  }
}

async function openCase({ folder, side, library }) {
  const payload = await api("POST", "/api/sessions", {
    folder,
    side,
    library: library || null,
  });
  state.sessionId = payload.session_id;
  state.schema = await api("GET", "/api/schema");
  buildPanel();
  await applyEnvelope(payload);
  setMode("plan");
  return payload;
}

el("open-form").addEventListener("submit", (event) => {
  event.preventDefault();
  busy(() => openCase({
    folder: el("folder").value.trim(),
    side: el("side").value,
    library: el("library").value.trim(),
  }));
});

el("case-picker").addEventListener("change", (event) => {
  const chosen = JSON.parse(event.target.value || "null");
  if (!chosen) return;
  el("folder").value = chosen.folder;
  el("side").value = chosen.side;
});

el("commit").addEventListener("click", () =>
  busy(async () => {
    const payload = await api("POST", `/api/sessions/${state.sessionId}/commit`);
    await applyEnvelope(payload);
  })
);

el("reset").addEventListener("click", () =>
  busy(async () => {
    const payload = await api(
      "POST", `/api/sessions/${state.sessionId}/controls/reset`
    );
    await applyEnvelope(payload);
  })
);

el("reset-trial").addEventListener("click", () =>
  busy(async () => {
    const payload = await api("POST", `/api/sessions/${state.sessionId}/trial/reset`);
    await applyEnvelope(payload);
  })
);

el("play").addEventListener("click", () => playArc());

el("export").addEventListener("click", () =>
  busy(async () => {
    const out = el("export-path").value.trim();
    if (!out) { note("Give a folder to export into."); return; }
    const payload = await api("POST", `/api/sessions/${state.sessionId}/export`, { out });
    note(`Exported to ${payload.out}`);
  })
);

el("open-report").addEventListener("click", () => {
  if (!state.sessionId) { note("Open a case first."); return; }
  window.open(`/api/sessions/${state.sessionId}/report`, "_blank");
});

// The fit value under the pointer, at most once a frame.
let pointerPending = false;
el("view").addEventListener("pointermove", (event) => {
  if (pointerPending) return;
  pointerPending = true;
  requestAnimationFrame(() => {
    pointerPending = false;
    showValueUnderPointer(event);
  });
});

el("clean").addEventListener("change", (event) =>
  viewer.setCleanViewport(event.target.checked)
);

for (const button of document.querySelectorAll("#modes button")) {
  button.addEventListener("click", () => setMode(button.dataset.mode));
}

(async function start() {
  let cases = [];
  try {
    ({ cases } = await api("GET", "/api/cases"));
    const picker = el("case-picker");
    for (const item of cases) {
      const option = document.createElement("option");
      option.value = JSON.stringify(item);
      option.textContent = `${item.case_id} (${item.side})`;
      picker.append(option);
    }
  } catch (error) {
    note(`Could not list cases: ${error.message}`);
  }

  // A launcher can name the case in the URL, so that starting the planner and opening
  // a patient are one action rather than two. This is the only reason the viewer reads
  // its own address: `?case=P009&side=left` opens that case, `?case=first` opens
  // whichever the server lists first, and `&commit=1` carves the plan once it is built.
  // The server still decides what a case is; the URL only picks from what it listed.
  const asked = new URLSearchParams(location.search);
  const wanted = asked.get("case");
  if (!wanted) return;

  const side = asked.get("side");
  const matches = (item) =>
    (wanted === "first" || item.case_id === wanted) && (!side || item.side === side);
  const chosen = cases.find(matches);
  if (!chosen) {
    note(`No case ${wanted}${side ? ` (${side})` : ""} among the ${cases.length} listed.`);
    return;
  }

  el("folder").value = chosen.folder;
  el("side").value = chosen.side;
  el("case-picker").value = JSON.stringify(chosen);
  await busy(async () => {
    await openCase({
      folder: chosen.folder,
      side: chosen.side,
      library: el("library").value.trim(),
    });
    if (asked.get("commit") === "1") {
      const payload = await api("POST", `/api/sessions/${state.sessionId}/commit`);
      await applyEnvelope(payload);
    }
  });
})();
