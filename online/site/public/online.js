// The online landmark study. The planner does not run here: every knee was computed on
// the study lead's machine by the planner itself and packed, encrypted, one file per
// knee. This page decrypts a knee in the browser and hands it to the planner's own
// viewer and Pick code (viewer.js, study.js, unchanged), answering the few requests
// that code makes from the file instead of from a server. A save goes to the study's
// store, and from there through the planner's own save_picks on the study lead's machine.

import { Viewer } from "./viewer.js";
import { Study } from "./study.js";

const el = (id) => document.getElementById(id);
const state = { controls: {} };
let config = null;
let keys = null;       // { aes: CryptoKey, token: hex }
let manifest = null;
let viewer = null;
let study = null;
let knee = null;       // the open knee's decrypted file
let progress = { saved: {}, complete: {} };
const meshUrls = new Map();

// --------------------------------------------------------------------
// Keys and files
// --------------------------------------------------------------------

function normalisePassphrase(text) {
  const bare = text.toUpperCase().replace(/[^A-Z0-9]/g, "");
  return bare.match(/.{1,4}/g)?.join("-") ?? "";
}

async function deriveKeys(passphrase) {
  const salt = Uint8Array.from(config.salt.match(/../g), (h) => parseInt(h, 16));
  const base = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(passphrase), "PBKDF2", false, ["deriveBits"]);
  const bits = new Uint8Array(await crypto.subtle.deriveBits(
    { name: "PBKDF2", hash: "SHA-256", salt, iterations: config.iterations }, base, 512));
  const aes = await crypto.subtle.importKey(
    "raw", bits.slice(0, 32), "AES-GCM", false, ["decrypt"]);
  const token = [...bits.slice(32)].map((b) => b.toString(16).padStart(2, "0")).join("");
  return { aes, token };
}

async function openFile(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not download ${path}.`);
  const sealed = new Uint8Array(await response.arrayBuffer());
  const plain = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: sealed.slice(0, 12) }, keys.aes, sealed.slice(12));
  const text = await new Response(
    new Blob([plain]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
  return JSON.parse(text);
}

async function call(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", "x-study-key": keys.token,
               ...(options.headers || {}) },
  });
  let payload = {};
  try { payload = await response.json(); } catch { /* empty */ }
  if (!response.ok) throw new Error(payload.error || `The study store said ${response.status}.`);
  return payload;
}

// --------------------------------------------------------------------
// What the planner's Pick code asks for, answered from the knee's file
// --------------------------------------------------------------------

async function api(method, path, body) {
  if (path.endsWith("/study/targets")) {
    return knee.targets[body.mode];
  }
  if (path.endsWith("/study/picks")) {
    const saved = await call("/api/save", {
      method: "POST",
      body: JSON.stringify({ case_id: knee.case_id, side: knee.side, body }),
    });
    const task = taskKey(knee.case_id, knee.side, body.mode, body.round);
    progress.saved[task] = { count: (progress.saved[task]?.count || 0) + 1 };
    progress.complete[task] = Boolean(body.complete);
    return { ...saved, landmarks: "the study's online store" };
  }
  throw new Error(`Not available online: ${path}`);
}

function note(message) {
  const row = document.createElement("div");
  row.textContent = message;
  el("notes").prepend(row);
}

async function busy(work, text = "") {
  el("busy").hidden = false;
  el("busy-text").textContent = text;
  try {
    return await work();
  } catch (error) {
    note(error.message);
    return null;
  } finally {
    el("busy").hidden = true;
  }
}

// --------------------------------------------------------------------
// Sign in
// --------------------------------------------------------------------

function remembered(key, fallback = "") {
  try { return localStorage.getItem(`landmark-study:${key}`) ?? fallback; } catch { return fallback; }
}
function remember(key, value) {
  try { localStorage.setItem(`landmark-study:${key}`, value); } catch { /* fine */ }
}

async function signIn() {
  const name = el("name").value.trim();
  const passphrase = normalisePassphrase(el("passphrase").value);
  el("signin-error").textContent = "";
  if (!name) { el("signin-error").textContent = "Enter your name."; return; }
  if (!passphrase) { el("signin-error").textContent = "Enter the study passphrase."; return; }
  await busy(async () => {
    keys = await deriveKeys(passphrase);
    try {
      manifest = await openFile("data/manifest.bin");
    } catch {
      keys = null;
      throw new Error("That passphrase does not open the study. Check it and try again.");
    }
    remember("name", name);
    remember("round", el("round").value);
    try { sessionStorage.setItem("landmark-study:pass", passphrase); } catch { /* fine */ }
    el("study-observer").value = name;
    el("study-round").value = el("round").value;
    el("who").textContent = `${name}, round ${el("round").value}`;
    await showKnees();
  }, "Opening the study").then(() => {
    if (!manifest) el("signin-error").textContent = el("notes").firstChild?.textContent || "";
  });
}

// --------------------------------------------------------------------
// The knee list
// --------------------------------------------------------------------

function taskKey(caseId, side, mode, round) {
  return `${caseId}_${side}_${mode}_r${round}`;
}

function status(caseId, side, mode) {
  const task = taskKey(caseId, side, mode, el("study-round").value);
  if (progress.complete[task]) return "done";
  return progress.saved[task] ? "partial" : "";
}

async function refreshProgress() {
  try {
    progress = await call(`/api/progress?observer=${encodeURIComponent(el("study-observer").value)}`);
  } catch (error) {
    note(`Progress could not be read (${error.message}); the list may be behind.`);
  }
}

async function showKnees() {
  el("signin").hidden = true;
  el("work").hidden = true;
  el("back").hidden = true;
  el("knees").hidden = false;
  document.body.classList.remove("picking");
  await refreshProgress();

  const groups = el("group");
  if (!groups.options.length) {
    for (const group of manifest.groups) {
      const option = document.createElement("option");
      option.value = group;
      option.textContent = `Group ${group}`;
      groups.append(option);
    }
    groups.value = remembered("group", manifest.groups[0]);
  }
  el("credit").textContent = manifest.credit;
  renderList("list-study", manifest.knees.filter((k) => k.group === "study"), "count-study");
  renderList("list-group", manifest.knees.filter((k) => k.group === groups.value), "count-group");
}

function renderList(host, knees, countId) {
  const list = el(host);
  list.textContent = "";
  let done = 0;
  for (const item of knees) {
    const row = document.createElement("div");
    row.className = "knee";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = `${item.case_id}, ${item.side} knee`;
    const blankState = status(item.case_id, item.side, "blank");
    const adjustState = status(item.case_id, item.side, "adjust");
    if (blankState === "done" && adjustState === "done") done += 1;
    const blank = button("1. No markers", blankState, () => openKnee(item, "blank"));
    const adjust = button("2. Automatic markers", adjustState, () => openKnee(item, "adjust"));
    if (blankState !== "done") {
      adjust.disabled = true;
      adjust.title = "Finish No markers on this knee first.";
    }
    row.append(name, blank, adjust);
    list.append(row);
  }
  el(countId).textContent = `${done} of ${knees.length} done`;
}

function button(label, stateName, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = label;
  if (stateName) b.classList.add(stateName);
  b.addEventListener("click", onClick);
  return b;
}

// --------------------------------------------------------------------
// Picking on one knee
// --------------------------------------------------------------------

async function openKnee(item, mode) {
  await busy(async () => {
    knee = await openFile(item.file);
    for (const url of meshUrls.values()) URL.revokeObjectURL(url);
    meshUrls.clear();
    for (const [id, data] of Object.entries(knee.meshes)) {
      const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
      meshUrls.set(id, URL.createObjectURL(new Blob([bytes], { type: "model/gltf-binary" })));
    }
    el("notes").textContent = "";
    el("knees").hidden = true;
    el("work").hidden = false;
    el("back").hidden = false;
    document.body.classList.add("picking");
    el("knee-title").textContent = `${item.case_id}, ${item.side} knee: ` +
      (mode === "blank" ? "No markers" : "Automatic markers");
    if (!viewer) {
      viewer = new Viewer(el("view"));
      study = new Study({
        viewer, api, el, note, busy, state,
        sessionId: () => "online",
        sendControls: async () => {},
      });
      const stop = study.stopPick.bind(study);
      study.stopPick = () => { stop(); showKnees(); };
    }
    viewer.meshUrl = (id) => meshUrls.get(id);
    window.dispatchEvent(new Event("resize"));
    el("pick-start-mode").value = mode;
    await study.startPick();
  }, "Opening the knee");
}

// --------------------------------------------------------------------

(async function start() {
  config = await (await fetch("config.json")).json();
  el("name").value = remembered("name");
  el("round").value = remembered("round", "1");
  try { el("passphrase").value = sessionStorage.getItem("landmark-study:pass") || ""; } catch { /* fine */ }
  el("enter").addEventListener("click", signIn);
  el("passphrase").addEventListener("keydown", (e) => { if (e.key === "Enter") signIn(); });
  el("group").addEventListener("change", () => { remember("group", el("group").value); showKnees(); });
  el("back").addEventListener("click", () => {
    if (study?.pick && !window.confirm("Leave this knee? Unsaved markers stay as a draft in this browser.")) return;
    if (study?.pick) study.stopPick(); else showKnees();
  });
})();
