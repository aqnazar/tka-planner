// The online landmark study. The planner does not run here: every knee was computed on
// the study lead's machine by the planner itself and packed, encrypted, one file per
// knee. This page decrypts a knee in the browser and hands it to the planner's own
// viewer and Pick code (viewer.js, study.js; Russian through study.js's say hook), answering the few requests
// that code makes from the file instead of from a server. A save goes to the study's
// store, and from there through the planner's own save_picks on the study lead's machine.

import * as THREE from "./vendor/three.module.js";
import { Viewer } from "./viewer.js";
import { Study } from "./study.js";
import { say, translateTargets } from "./ru.js";
import { orientation, checkPick, Compass } from "./orient.js";

const SIDE = { left: "левый", right: "правый" };
const MODE = { blank: "1  Без маркеров", adjust: "2  Автоматические маркеры" };
const CREDIT = "Суставы VSD: Fischer M.C.M., Sci. Data 10, 763 (2023), VSDFullBodyBoneModels, " +
  "CC BY-NC-SA 4.0; обрезаны до длины КТ коленного сустава.";

const el = (id) => document.getElementById(id);
const state = { controls: {} };
let config = null;
let keys = null;       // { aes: CryptoKey, token: hex }
let manifest = null;
let viewer = null;
let study = null;
let knee = null;       // the open knee's decrypted file
let orient = null;     // its medial/lateral and front/back directions (orient.js)
let compass = null;
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
  if (!response.ok) throw new Error(`Не удалось загрузить ${path}.`);
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
  if (!response.ok) throw new Error(payload.error || `Ошибка хранилища исследования: ${response.status}.`);
  return payload;
}

// --------------------------------------------------------------------
// What the planner's Pick code asks for, answered from the knee's file
// --------------------------------------------------------------------

async function api(method, path, body) {
  if (path.endsWith("/study/targets")) {
    return translateTargets(knee.targets[body.mode]);
  }
  if (path.endsWith("/study/picks")) {
    const saved = await call("/api/save", {
      method: "POST",
      body: JSON.stringify({ case_id: knee.case_id, side: knee.side, body }),
    });
    const task = taskKey(knee.case_id, knee.side, body.mode, body.round);
    progress.saved[task] = { count: (progress.saved[task]?.count || 0) + 1 };
    progress.complete[task] = Boolean(body.complete);
    return { ...saved, landmarks: "онлайн-хранилище" };
  }
  throw new Error(`Недоступно онлайн: ${path}`);
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
  if (!name) { el("signin-error").textContent = "Введите ваше имя."; return; }
  if (!passphrase) { el("signin-error").textContent = "Введите пароль исследования."; return; }
  await busy(async () => {
    await configReady;
    keys = await deriveKeys(passphrase);
    try {
      manifest = await openFile("data/manifest.bin");
    } catch {
      keys = null;
      throw new Error("Этот пароль не открывает исследование. Проверьте его и попробуйте снова.");
    }
    remember("name", name);
    remember("round", el("round").value);
    try { sessionStorage.setItem("landmark-study:pass", passphrase); } catch { /* fine */ }
    el("study-observer").value = name;
    el("study-round").value = el("round").value;
    el("who").textContent = `${name}, раунд ${el("round").value}`;
    await showKnees();
  }, "Открываем исследование").then(() => {
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
    note(`Не удалось прочитать прогресс (${error.message}); список может быть неактуален.`);
  }
}

async function showKnees() {
  el("signin").hidden = true;
  el("work").hidden = true;
  el("back").hidden = true;
  el("knees").hidden = false;
  document.body.classList.remove("picking");
  compass?.set(null);
  await refreshProgress();

  const groups = el("group");
  if (!groups.options.length) {
    for (const group of manifest.groups) {
      const option = document.createElement("option");
      option.value = group;
      option.textContent = `Группа ${group}`;
      groups.append(option);
    }
    groups.value = remembered("group", manifest.groups[0]);
  }
  el("credit").textContent = CREDIT;
  const study = manifest.knees.filter((k) => k.group === "study");
  const mine = manifest.knees.filter((k) => k.group === groups.value);
  const done = renderList("list-study", study, "count-study") +
    renderList("list-group", mine, "count-group");
  el("total-done").textContent = String(done);
  el("total-of").textContent = `из ${study.length + mine.length} суставов выполнено`;
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
    name.textContent = item.case_id;
    const side = document.createElement("small");
    side.textContent = `${SIDE[item.side] ?? item.side} коленный сустав`;
    name.append(side);
    const blankState = status(item.case_id, item.side, "blank");
    const adjustState = status(item.case_id, item.side, "adjust");
    if (blankState === "done" && adjustState === "done") done += 1;
    const blank = button(MODE.blank, blankState, () => openKnee(item, "blank"));
    const adjust = button(MODE.adjust, adjustState, () => openKnee(item, "adjust"));
    if (blankState !== "done") {
      adjust.disabled = true;
      adjust.title = "Сначала завершите «Без маркеров» на этом суставе.";
    }
    row.append(name, blank, adjust);
    list.append(row);
  }
  el(countId).textContent = `${done} из ${knees.length}`;
  return done;
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
    orient = orientation(knee, item.group === "study");
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
    el("knee-title").textContent = `${item.case_id} ${SIDE[item.side] ?? item.side} · ${MODE[mode]}`;
    if (!viewer) {
      viewer = new Viewer(el("view"));
      // The house style's light viewport (as in the demo planner): paper, no floor grid.
      viewer.scene.background = new THREE.Color(0xf3f5f8);
      viewer.grid.visible = false;
      study = new Study({
        viewer, api, el, note, busy, state,
        sessionId: () => "online",
        sendControls: async () => {},
        say,
        checkPlacement: (target, mm, pick) => checkPick(target, mm, pick, orient),
      });
      compass = new Compass(el("view"), viewer);
      const stop = study.stopPick.bind(study);
      study.stopPick = () => { stop(); showKnees(); };
    }
    viewer.meshUrl = (id) => meshUrls.get(id);
    window.dispatchEvent(new Event("resize"));
    el("pick-start-mode").value = mode;
    await study.startPick();
    compass.set(orient);
  }, "Открываем сустав");
}

// --------------------------------------------------------------------

// The remembered name and round are filled in at once, and never over what is already
// typed; the study's settings load beside them, and Start waits for them.
const configReady = fetch("config.json").then((r) => r.json()).then((c) => { config = c; });

(function start() {
  if (!el("name").value) el("name").value = remembered("name");
  el("round").value = remembered("round", "1");
  try {
    if (!el("passphrase").value) el("passphrase").value = sessionStorage.getItem("landmark-study:pass") || "";
  } catch { /* fine */ }
  el("enter").addEventListener("click", signIn);
  configReady.then(() => { el("enter").disabled = false; el("enter").title = ""; },
                   () => { el("signin-error").textContent = "Исследование не загрузилось. Обновите страницу."; });
  el("passphrase").addEventListener("keydown", (e) => { if (e.key === "Enter") signIn(); });
  el("group").addEventListener("change", () => { remember("group", el("group").value); showKnees(); });
  el("back").addEventListener("click", () => {
    if (study?.pick && !window.confirm("Выйти из этого сустава? Несохранённые маркеры останутся черновиком в этом браузере.")) return;
    if (study?.pick) study.stopPick(); else showKnees();
  });
})();
