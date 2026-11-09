// Which way is medial on the open knee: the labels in the 3D view, and a check that a
// marker was not put on the wrong side of the knee (or on a condyle instead of an
// epicondyle). The first surgeon's first knee had the tibia mirrored, medial for
// lateral, and the fibular head placed on a tibia whose scan has no fibula.
//
// The directions come from the planner's automatic landmarks, which are in the knee's
// file for mode 2. Only directions and coarse heights are used, never a position, so
// mode 1 stays blind: a warning says which side, not where.

import * as THREE from "./vendor/three.module.js";

const MEDIAL = new Set([
  "femur.epicondyle_medial_sulcus", "femur.epicondyle_medial_prominence",
  "femur.condyle_distal_medial", "femur.condyle_posterior_medial",
  "tibia.plateau_medial_lowest", "tibia.plateau_medial_anterior",
  "tibia.plateau_medial_posterior",
]);
const LATERAL = new Set([
  "femur.epicondyle_lateral", "femur.condyle_distal_lateral",
  "femur.condyle_posterior_lateral", "tibia.plateau_lateral_lowest", "fibula.head_apex",
]);
const SPINES = { "tibia.spine_medial": "tibia.spine_lateral",
                 "tibia.spine_lateral": "tibia.spine_medial" };
const DISTAL = new Set(["femur.condyle_distal_medial", "femur.condyle_distal_lateral"]);
const EPICONDYLE = new Set(["femur.epicondyle_lateral", "femur.epicondyle_medial_sulcus",
                            "femur.epicondyle_medial_prominence"]);

const SIDE_MARGIN_MM = 3;        // this far onto the other side of the knee's centre
const DISTAL_MAX_MM = 10;        // a distal condyle point higher than this above the femur's end
const EPICONDYLE_MIN_MM = 12;    // an epicondyle point lower than this
const EPICONDYLE_OUT_MM = 20;    // an epicondyle point nearer the middle than this (they sit ~40 mm out)
const LABEL_REACH_MM = 55;       // the labels sit this far out from the bone's centre
const LABEL_HIDE_PX = 28;        // a pair seen end-on is hidden

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const length = (a) => Math.hypot(...a);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]];
const mean = (points) => (points.length
  ? scale(points.reduce(add, [0, 0, 0]), 1 / points.length) : null);

// The knee's lateral, anterior and proximal directions (scan millimetres, LPS), each
// bone's centre, and the height of the femur's distal end. ``study`` is true for our
// patients, whose scans hold no fibula.
export function orientation(knee, study) {
  const auto = new Map(knee.targets.adjust.targets.filter((t) => t.auto_mm)
    .map((t) => [t.id, t.auto_mm]));
  const left = knee.side === "left";
  const guess = [left ? 1 : -1, 0, 0];            // LPS: the patient's left is +x
  let sum = [0, 0, 0];
  for (const [lateral, medial] of [
    ["femur.epicondyle_lateral", "femur.epicondyle_medial_prominence"],
    ["femur.condyle_posterior_lateral", "femur.condyle_posterior_medial"],
    ["femur.condyle_distal_lateral", "femur.condyle_distal_medial"],
    ["tibia.plateau_lateral_lowest", "tibia.plateau_medial_lowest"],
  ]) {
    if (!auto.has(lateral) || !auto.has(medial)) continue;
    const d = sub(auto.get(lateral), auto.get(medial));
    d[2] = 0;
    if (length(d) > 1) sum = add(sum, scale(d, 1 / length(d)));
  }
  let lateral = length(sum) > 0.5 ? scale(sum, 1 / length(sum)) : guess;
  if (dot(lateral, guess) < 0.5) lateral = guess;  // a planner miss never turns the knee round
  const proximal = [0, 0, 1];
  const anterior = cross(left ? lateral : scale(lateral, -1), proximal);
  const pick = (ids) => ids.filter((id) => auto.has(id)).map((id) => auto.get(id));
  const ofBone = (prefix) => [...auto].filter(([id]) => id.startsWith(prefix)).map(([, p]) => p);
  const femur = auto.get("femur.notch_centre") ?? mean(ofBone("femur."));
  const tibia = mean(pick(["tibia.spine_medial", "tibia.spine_lateral"])) ?? mean(ofBone("tibia."));
  const distal = pick(["femur.condyle_distal_medial", "femur.condyle_distal_lateral"]);
  return {
    lateral, anterior, proximal, study,
    centre: { Femur: femur, Tibia: tibia },
    distalEnd: distal.length ? Math.min(...distal.map((p) => dot(p, proximal))) : null,
  };
}

const keep = "\n\nОставить точку здесь? (ОК — оставить, Отмена — убрать)";

// A warning for a marker just put down, or null.
export function checkPick(target, mm, pick, o) {
  if (!o) return null;
  const name = `«${target.name}»`;
  if (target.id === "fibula.head_apex" && o.study) {
    return `${name}: на КТ пациентов исследования малоберцовой кости нет, ` +
      "точка стоит на большеберцовой кости. Пропустите этот маркер кнопкой " +
      "«Не могу поставить» с причиной «Вне области сканирования»." + keep;
  }
  const centre = o.centre[target.bone];
  const side = centre ? dot(sub(mm, centre), o.lateral) : null;
  if (side !== null && MEDIAL.has(target.id) && side > SIDE_MARGIN_MM) {
    return `${name} — МЕДИАЛЬНЫЙ ориентир, а точка стоит на ЛАТЕРАЛЬНОЙ стороне ` +
      "сустава. Сверьтесь с подписями «Медиально / Латерально» на экране." + keep;
  }
  if (side !== null && LATERAL.has(target.id) && side < -SIDE_MARGIN_MM) {
    return `${name} — ЛАТЕРАЛЬНЫЙ ориентир, а точка стоит на МЕДИАЛЬНОЙ стороне ` +
      "сустава. Сверьтесь с подписями «Медиально / Латерально» на экране." + keep;
  }
  if (side !== null && EPICONDYLE.has(target.id) && Math.abs(side) < EPICONDYLE_OUT_MM) {
    return `${name} — точка на НАДмыщелке, на боковой поверхности бедра, а эта точка ` +
      "стоит близко к середине сустава (блок или вырезка?)." + keep;
  }
  const other = SPINES[target.id] && pick.positions.get(SPINES[target.id]);
  if (other) {
    const medialOne = target.id === "tibia.spine_medial";
    const further = dot(sub(mm, other), o.lateral);
    if ((medialOne && further > 0) || (!medialOne && further < 0)) {
      return "Медиальный межмыщелковый бугорок оказался латеральнее латерального. " +
        "Проверьте стороны по подписям на экране." + keep;
    }
  }
  if (target.bone === "Femur" && o.distalEnd !== null) {
    const height = dot(mm, o.proximal) - o.distalEnd;
    if (DISTAL.has(target.id) && height > DISTAL_MAX_MM) {
      return `${name} — самая дистальная точка мыщелка (на суставной поверхности), ` +
        `а точка стоит на ${height.toFixed(0)} мм выше дистального конца бедра. ` +
        "Возможно, это надмыщелок?" + keep;
    }
    if (EPICONDYLE.has(target.id) && height < EPICONDYLE_MIN_MM) {
      return `${name} — точка на НАДмыщелке (боковая поверхность, выше суставной ` +
        `поверхности), а эта точка всего в ${Math.max(0, height).toFixed(0)} мм от ` +
        "дистального конца бедра — это похоже на мыщелок." + keep;
    }
  }
  return null;
}

// Medial, lateral, front and back, written beside the bone in the 3D view.
export class Compass {
  constructor(host, viewer) {
    this.viewer = viewer;
    this.o = null;
    this.box = document.createElement("div");
    this.box.className = "compass";
    this.labels = {};
    for (const [key, text] of [["medial", "Медиально"], ["lateral", "Латерально"],
                               ["anterior", "Спереди"], ["posterior", "Сзади"]]) {
      const label = document.createElement("span");
      label.className = `compass-${key}`;
      label.textContent = text;
      this.box.append(label);
      this.labels[key] = label;
    }
    host.append(this.box);
    const loop = () => { this.update(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  set(o) {
    this.o = o;
    this.box.hidden = !o;
  }

  update() {
    const o = this.o;
    const bones = this.viewer.pick?.bones;
    if (!o || !bones) { this.box.hidden = true; return; }
    const shown = [...bones].filter(([, mesh]) => mesh.visible);
    const centres = shown.map(([name]) => o.centre[name]).filter(Boolean);
    if (!shown.length || !centres.length) { this.box.hidden = true; return; }
    this.box.hidden = false;
    const centre = mean(centres);
    const matrix = shown[0][1].matrixWorld;
    const camera = this.viewer.camera;
    const rect = this.box.getBoundingClientRect();
    const screen = (mm) => {
      const p = new THREE.Vector3(...mm.map((v) => v / 1000)).applyMatrix4(matrix).project(camera);
      return { x: (p.x + 1) / 2 * rect.width, y: (1 - p.y) / 2 * rect.height, behind: p.z > 1 };
    };
    const middle = screen(centre);
    const place = (a, b, along) => {
      const ends = [screen(add(centre, scale(along, LABEL_REACH_MM))),
                    screen(add(centre, scale(along, -LABEL_REACH_MM)))];
      const endOn = Math.hypot(ends[0].x - middle.x, ends[0].y - middle.y) < LABEL_HIDE_PX;
      for (const [label, at] of [[a, ends[0]], [b, ends[1]]]) {
        label.hidden = endOn || at.behind;
        label.style.transform = `translate(${at.x}px, ${at.y}px) translate(-50%, -50%)`;
      }
    };
    place(this.labels.lateral, this.labels.medial, o.lateral);
    place(this.labels.anterior, this.labels.posterior, o.anterior);
  }
}
