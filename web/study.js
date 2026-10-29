// The observer studies: Pick (landmark picking, E3) and Review (plan verdicts, E13).
//
// Like the rest of the viewer this records and draws; it computes nothing clinical.
// What to pick, in which order and with which definition, comes from the server, and so
// do the review's reasons and severities. Every pick and verdict is saved on the server;
// the browser keeps only a draft of unsaved picks, so a closed tab does not lose them.
//
// Pick works on the whole uncut bones. The surgeon is taken through the markers one at
// a time: a click on the bone puts the current marker there, a drag of the current
// marker slides it over the surface, and orbiting works as everywhere else. In "adjust" mode every marker starts
// at the automatic estimate, which stays on screen as a ghost, and can be reset to it.

const SKIP_REASONS = [
  "Not identifiable",
  "Outside the scan",
  "Obscured by osteophyte or deformity",
  "Segmentation defect",
];
// A press that moves less than this, in pixels, is a click and places the marker; more
// is an orbit and does not.
const CLICK_SLOP_PX = 5;

export class Study {
  constructor({ viewer, api, el, note, busy, sessionId, sendControls, state }) {
    Object.assign(this, { viewer, api, el, note, busy, sessionId, sendControls, state });
    this.pick = null;
    this.reviewStarted = null;
    this.reviewSaved = [];
    this.wire();
  }

  async loadSchema() {
    if (!this.schema) this.schema = await this.api("GET", "/api/study/schema");
    this.buildReviewForm();
  }

  observer() {
    const observer = this.el("study-observer").value.trim();
    const round = Number(this.el("study-round").value);
    if (!observer) throw new Error("Enter the observer's code first.");
    return { observer, round };
  }

  // ------------------------------------------------------------------
  // Pick
  // ------------------------------------------------------------------

  async startPick() {
    const { observer, round } = this.observer();
    const mode = this.el("pick-start-mode").value;
    const given = await this.api(
      "POST", `/api/sessions/${this.sessionId()}/study/targets`, { mode }
    );
    await this.viewer.enterPick(given.bones);
    const auto = new Map(given.targets.filter((t) => t.auto_mm).map((t) => [t.id, t.auto_mm]));
    this.pick = {
      ...given, observer, round, auto,
      positions: new Map(auto),
      skipped: new Map(),
      seconds: new Map(),
      index: 0,
      enteredAt: performance.now(),
      started: new Date().toISOString(),
      startedAt: performance.now(),
    };
    this.restoreDraft();
    for (const target of given.targets) {
      if (auto.has(target.id)) {
        this.viewer.setGhost(target.id, { bone: target.bone, mm: auto.get(target.id) });
      }
    }
    this.viewer.setGhostsVisible(this.el("pick-ghosts").checked);
    this.el("pick-setup").hidden = true;
    this.el("pick-run").hidden = false;
    this.el("pick-ghost-row").hidden = mode !== "adjust";
    this.el("pick-reset-one").textContent =
      mode === "adjust" ? "Reset to automatic" : "Remove marker";
    this.drawAll();
    this.go(this.pick.index);
  }

  stopPick() {
    if (!this.pick) return;
    this.pick = null;
    this.viewer.exitPick();
    this.el("pick-setup").hidden = false;
    this.el("pick-run").hidden = true;
  }

  current() {
    return this.pick?.targets[this.pick.index] ?? null;
  }

  go(index) {
    const pick = this.pick;
    if (!pick) return;
    this.chargeTime();
    pick.index = Math.max(0, Math.min(pick.targets.length - 1, index));
    const target = this.current();
    const view = this.el("pick-view").value;
    if (view === "auto") {
      this.viewer.showPickBones(target.bone);
      if (this.lastBone !== target.bone) this.viewer.framePickBone(target.bone);
      this.lastBone = target.bone;
    }
    this.drawAll();
    this.renderPrompt();
  }

  // Time on a marker counts while it is the current one, across every visit.
  chargeTime() {
    const pick = this.pick;
    const target = this.current();
    const now = performance.now();
    if (target) {
      pick.seconds.set(target.id,
        (pick.seconds.get(target.id) || 0) + (now - pick.enteredAt) / 1000);
    }
    pick.enteredAt = now;
  }

  place(mm) {
    const target = this.current();
    if (!target) return;
    this.pick.positions.set(target.id, mm);
    this.pick.skipped.delete(target.id);
    this.drawMarker(target);
    this.renderPrompt();
    this.saveDraft();
  }

  skip() {
    const target = this.current();
    if (!target) return;
    this.pick.positions.delete(target.id);
    this.pick.skipped.set(target.id, this.el("pick-skip-reason").value);
    this.viewer.removeMarker(target.id);
    this.saveDraft();
    this.go(this.pick.index + 1);
  }

  resetOne() {
    const target = this.current();
    if (!target) return;
    this.pick.skipped.delete(target.id);
    if (this.pick.auto.has(target.id)) {
      this.pick.positions.set(target.id, this.pick.auto.get(target.id));
    } else {
      this.pick.positions.delete(target.id);
    }
    this.drawMarker(target);
    this.renderPrompt();
    this.saveDraft();
  }

  resetAll() {
    const what = this.pick.mode === "adjust"
      ? "Put every marker back at its automatic position?"
      : "Remove every marker placed so far?";
    if (!window.confirm(what)) return;
    this.pick.positions = new Map(this.pick.auto);
    this.pick.skipped.clear();
    this.drawAll();
    this.go(0);
    this.saveDraft();
  }

  drawMarker(target) {
    const mm = this.pick.positions.get(target.id);
    if (!mm) {
      this.viewer.removeMarker(target.id);
      return;
    }
    const isCurrent = target === this.current();
    this.viewer.setMarker(target.id, {
      bone: target.bone, mm, number: target.number,
      state: isCurrent ? "current" : "placed",
    });
  }

  drawAll() {
    for (const target of this.pick.targets) this.drawMarker(target);
    this.renderList();
  }

  renderPrompt() {
    const pick = this.pick;
    const target = this.current();
    const done = pick.targets.filter((t) => this.statusOf(t) !== "todo").length;
    this.el("pick-progress").textContent =
      `Marker ${target.number} of ${pick.targets.length} (${done} done)`;
    this.el("pick-name").textContent =
      `${target.number}) ${target.bone}: ${target.name}`;
    this.el("pick-definition").textContent = target.definition;
    const status = this.statusOf(target);
    const moved = this.movedMm(target);
    this.el("pick-status").textContent =
      status === "skipped" ? `Skipped: ${pick.skipped.get(target.id)}` :
      status === "todo" ? "Click the bone to place this marker." :
      moved === null ? "Placed. Drag it to move it over the surface." :
      `Placed, ${moved.toFixed(1)} mm from the automatic position. Drag it to move it.`;
    this.el("pick-prev").disabled = pick.index === 0;
    this.el("pick-next").textContent =
      pick.index === pick.targets.length - 1 ? "Last marker" : "Next";
    this.el("pick-next").disabled = pick.index === pick.targets.length - 1;
    this.el("pick-finish").disabled = done < pick.targets.length;
    this.renderList();
  }

  // Shown to the surgeon in adjust mode, where the starting point is no secret.
  movedMm(target) {
    const auto = this.pick.auto.get(target.id);
    const mm = this.pick.positions.get(target.id);
    if (!auto || !mm) return null;
    return Math.hypot(mm[0] - auto[0], mm[1] - auto[1], mm[2] - auto[2]);
  }

  statusOf(target) {
    if (this.pick.skipped.has(target.id)) return "skipped";
    if (!this.pick.positions.has(target.id)) return "todo";
    // An automatic marker the surgeon has not yet been shown has not been checked.
    if (this.pick.mode === "adjust" && !this.pick.seconds.has(target.id)
        && target !== this.current()) return "todo";
    return "placed";
  }

  renderList() {
    const host = this.el("pick-list");
    host.textContent = "";
    for (const target of this.pick.targets) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = `pick-item ${this.statusOf(target)}`;
      if (target === this.current()) row.classList.add("current");
      row.textContent = `${target.number}) ${target.name}`;
      row.title = `${target.bone}: ${target.definition}`;
      row.addEventListener("click", () => this.go(target.number - 1));
      host.append(row);
    }
  }

  async savePicks(complete) {
    const pick = this.pick;
    this.chargeTime();
    const picks = [];
    for (const target of pick.targets) {
      const status = this.statusOf(target);
      if (status === "todo") continue;
      const seconds = Math.round((pick.seconds.get(target.id) || 0) * 10) / 10;
      picks.push(status === "skipped"
        ? { id: target.id, position_mm: null, reason: pick.skipped.get(target.id), seconds }
        : { id: target.id, position_mm: pick.positions.get(target.id), seconds });
    }
    if (!picks.length) throw new Error("Nothing has been placed yet.");
    const saved = await this.api("POST", `/api/sessions/${this.sessionId()}/study/picks`, {
      observer: pick.observer, round: pick.round, mode: pick.mode, complete,
      started_utc: pick.started,
      total_seconds: Math.round((performance.now() - pick.startedAt) / 100) / 10,
      picks,
    });
    this.note(`Saved: ${saved.placed} placed, ${saved.skipped} skipped, in ${saved.landmarks}`);
    if (complete) {
      this.clearDraft();
      this.stopPick();
    }
    return saved;
  }

  // -- the pointer: click to place, drag a marker to slide it --------------------

  pointerDown(event) {
    if (!this.pick || event.button !== 0) return;
    // Only the current marker can be grabbed. Another one is reached from the list or
    // with Previous, so a click beside a placed neighbour places the current marker.
    if (this.viewer.isOnMarker(event.clientX, event.clientY, this.current().id)) {
      this.dragging = true;
      this.viewer.setControlsEnabled(false);
      this.el("view").setPointerCapture(event.pointerId);
      return;
    }
    this.pressed = { x: event.clientX, y: event.clientY };
  }

  pointerMove(event) {
    if (!this.pick || !this.dragging) return;
    const mm = this.viewer.pickHit(event.clientX, event.clientY, this.current().bone);
    if (mm) this.place(mm);
  }

  pointerUp(event) {
    if (!this.pick) return;
    if (this.dragging) {
      this.dragging = false;
      this.viewer.setControlsEnabled(true);
      this.el("view").releasePointerCapture?.(event.pointerId);
      return;
    }
    const pressed = this.pressed;
    this.pressed = null;
    if (!pressed) return;
    if (Math.hypot(event.clientX - pressed.x, event.clientY - pressed.y) > CLICK_SLOP_PX) {
      return;
    }
    const mm = this.viewer.pickHit(event.clientX, event.clientY, this.current().bone);
    if (mm) {
      this.place(mm);
    } else {
      this.el("pick-status").textContent =
        `That is not on the ${this.current().bone.toLowerCase()}. Click on its surface.`;
    }
  }

  // -- the draft: unsaved picks kept in this browser only -------------------------

  draftKey() {
    const p = this.pick;
    return `tka-study-draft:${p.case_id}:${p.side}:${p.observer}:${p.mode}:r${p.round}`;
  }

  saveDraft() {
    try {
      localStorage.setItem(this.draftKey(), JSON.stringify({
        index: this.pick.index,
        positions: [...this.pick.positions],
        skipped: [...this.pick.skipped],
        seconds: [...this.pick.seconds],
      }));
    } catch { /* storage unavailable: the draft is a convenience only */ }
  }

  restoreDraft() {
    let draft = null;
    try {
      draft = JSON.parse(localStorage.getItem(this.draftKey()) || "null");
    } catch { draft = null; }
    if (!draft) return;
    if (!window.confirm("Unsaved picks for this case, observer and round were found. " +
                        "Continue from them?")) {
      this.clearDraft();
      return;
    }
    this.pick.positions = new Map(draft.positions);
    this.pick.skipped = new Map(draft.skipped);
    this.pick.seconds = new Map(draft.seconds);
    this.pick.index = draft.index || 0;
  }

  clearDraft() {
    try { localStorage.removeItem(this.draftKey()); } catch { /* nothing kept */ }
  }

  // ------------------------------------------------------------------
  // Review
  // ------------------------------------------------------------------

  buildReviewForm() {
    const reasons = this.el("review-reasons");
    reasons.textContent = "";
    for (const [key, label] of this.schema.reasons) {
      const item = document.createElement("label");
      item.className = "check";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.value = key;
      item.append(box, document.createTextNode(label));
      reasons.append(item);
    }
    const severities = this.el("review-severity");
    severities.textContent = "";
    for (const [key, label] of this.schema.severities) {
      const item = document.createElement("label");
      item.className = "check";
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "review-severity";
      radio.value = key;
      item.append(radio, document.createTextNode(label));
      severities.append(item);
    }
    this.syncReview();
  }

  enterReview() {
    this.reviewStarted = performance.now();
    this.el("review-plan").value = this.state.controls.philosophy || "mechanical";
    this.syncReview();
  }

  syncReview() {
    const verdict = document.querySelector("input[name=review-verdict]:checked")?.value;
    this.el("review-details").hidden = verdict !== "not_ok";
    const adjusted = Object.entries(this.state.controls || {}).some(
      ([name, value]) => /_delta_|_shift_|coronal_correction/.test(name) && Number(value)
    );
    this.el("review-adjusted").hidden = !adjusted;
  }

  async saveReview() {
    const { observer, round } = this.observer();
    const verdict = document.querySelector("input[name=review-verdict]:checked")?.value;
    if (!verdict) throw new Error("Mark the plan OK or NOT OK.");
    const reasons = [...this.el("review-reasons").querySelectorAll("input:checked")]
      .map((box) => box.value);
    const severity =
      document.querySelector("input[name=review-severity]:checked")?.value || null;
    const saved = await this.api("POST", `/api/sessions/${this.sessionId()}/study/review`, {
      observer, round, verdict,
      reasons: verdict === "not_ok" ? reasons : [],
      severity: verdict === "not_ok" ? severity : null,
      comment: this.el("review-comment").value,
      seconds: Math.round((performance.now() - this.reviewStarted) / 100) / 10,
    });
    this.reviewSaved.push(saved.record);
    this.renderSaved();
    this.clearReviewForm();
    this.reviewStarted = performance.now();
    this.note(`Verdict saved to ${saved.file}`);
  }

  clearReviewForm() {
    for (const input of document.querySelectorAll(
      "#review-mode input[type=checkbox], #review-mode input[type=radio]")) {
      input.checked = false;
    }
    this.el("review-comment").value = "";
    this.syncReview();
  }

  renderSaved() {
    const host = this.el("review-saved");
    host.textContent = "";
    for (const record of this.reviewSaved) {
      const row = document.createElement("div");
      row.className = `saved ${record.verdict}`;
      row.textContent = `${record.case_id} (${record.side}), ${record.alignment}, ` +
        `round ${record.round}: ${record.verdict === "ok" ? "OK" : "NOT OK"}` +
        (record.severity ? ` (${record.severity}; ${record.reasons.join(", ")})` : "");
      host.append(row);
    }
  }

  // ------------------------------------------------------------------
  // Wiring
  // ------------------------------------------------------------------

  wire() {
    const on = (id, type, handler) => this.el(id).addEventListener(type, handler);
    on("pick-start", "click", () => this.busy(() => this.startPick()));
    on("pick-prev", "click", () => this.go(this.pick.index - 1));
    on("pick-next", "click", () => this.go(this.pick.index + 1));
    on("pick-skip", "click", () => this.skip());
    on("pick-reset-one", "click", () => this.resetOne());
    on("pick-reset-all", "click", () => this.resetAll());
    on("pick-save", "click", () => this.busy(() => this.savePicks(false)));
    on("pick-finish", "click", () => this.busy(() => this.savePicks(true)));
    on("pick-quit", "click", () => {
      if (window.confirm("Stop picking? Unsaved markers stay as a draft in this browser.")) {
        this.stopPick();
      }
    });
    on("pick-ghosts", "change", (event) => this.viewer.setGhostsVisible(event.target.checked));
    on("pick-view", "change", (event) => {
      const view = event.target.value;
      this.viewer.showPickBones(view === "auto" ? this.current()?.bone : view);
    });

    const view = this.el("view");
    // Captured on the way down, so a press on a marker turns orbiting off before the
    // controls on the canvas below have seen it.
    view.addEventListener("pointerdown", (event) => this.pointerDown(event), true);
    view.addEventListener("pointermove", (event) => this.pointerMove(event));
    view.addEventListener("pointerup", (event) => this.pointerUp(event));
    window.addEventListener("keydown", (event) => {
      if (!this.pick || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
      if (event.key === "Enter" || event.key === "n") this.go(this.pick.index + 1);
      if (event.key === "Backspace" || event.key === "b") this.go(this.pick.index - 1);
    });

    on("review-plan", "change", (event) =>
      this.sendControls({ philosophy: event.target.value }));
    for (const radio of document.querySelectorAll("input[name=review-verdict]")) {
      radio.addEventListener("change", () => this.syncReview());
    }
    on("review-save", "click", () => this.busy(() => this.saveReview()));

    const reasons = this.el("pick-skip-reason");
    for (const reason of SKIP_REASONS) {
      const option = document.createElement("option");
      option.value = option.textContent = reason;
      reasons.append(option);
    }
  }
}
