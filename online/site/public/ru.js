// Russian for the online landmark study. The surgeons read Russian; what is saved stays
// as the planner records it (landmark ids, English skip reasons), so the analysis is the
// same whichever language the page was in.

import { fill } from "./study.js";

// The planner's Pick messages (web/study.js), by their English text.
const MESSAGES = {
  "Enter the surgeon's name first.": "Сначала введите имя хирурга.",
  "Reset to automatic": "Вернуть автоматическую",
  "Remove marker": "Удалить маркер",
  "Put every marker back at its automatic position?":
    "Вернуть все маркеры в автоматические позиции?",
  "Remove every marker placed so far?": "Удалить все поставленные маркеры?",
  "Marker {number} of {total} ({done} done)": "Маркер {number} из {total} (готово: {done})",
  "Femur": "Бедренная кость",
  "Tibia": "Большеберцовая кость",
  "Skipped: {reason}": "Пропущен: {reason}",
  "Click the bone to place this marker.": "Щёлкните по кости, чтобы поставить этот маркер.",
  "Placed. Drag it to move it over the surface.":
    "Поставлен. Перетащите его, чтобы сдвинуть по поверхности.",
  "Placed, {mm} mm from the automatic position. Drag it to move it.":
    "Поставлен, {mm} мм от автоматической позиции. Перетащите, чтобы сдвинуть.",
  "Last marker": "Последний маркер",
  "Next": "Далее",
  "Nothing has been placed yet.": "Ещё ничего не поставлено.",
  "Saved: {placed} placed, {skipped} skipped, in {where}":
    "Сохранено: поставлено {placed}, пропущено {skipped} ({where})",
  "That is not on the {bone}. Click on its surface.":
    "Щелчок мимо кости ({bone}). Щёлкните по её поверхности.",
  "Unsaved picks for this case, observer and round were found. Continue from them?":
    "Найдены несохранённые точки для этого сустава, хирурга и раунда. Продолжить с них?",
  "Stop picking? Unsaved markers stay as a draft in this browser.":
    "Остановить? Несохранённые маркеры останутся черновиком в этом браузере.",
  // Skip reasons: shown in Russian, saved in English.
  "Not identifiable": "Не определяется",
  "Outside the scan": "Вне области сканирования",
  "Obscured by osteophyte or deformity": "Скрыт остеофитом или деформацией",
  "Segmentation defect": "Дефект сегментации",
};

export function say(english, vars) {
  return fill(MESSAGES[english] ?? english, vars);
}

// Each landmark's Russian name (shown before the English one) and definition.
export const LANDMARKS = {
  "femur.epicondyle_lateral": ["Латеральный надмыщелок",
    "Наиболее выступающая точка латерального надмыщелка. Общая для хирургической и анатомической трансэпикондилярных осей."],
  "femur.epicondyle_medial_sulcus": ["Борозда медиального надмыщелка",
    "Самая глубокая точка борозды на медиальном надмыщелке — именно борозды, а не окружающей вершины. Медиальный конец ХИРУРГИЧЕСКОЙ трансэпикондилярной оси (sTEA)."],
  "femur.epicondyle_medial_prominence": ["Вершина медиального надмыщелка",
    "Наиболее выступающая точка медиального надмыщелка. Медиальный конец АНАТОМИЧЕСКОЙ трансэпикондилярной оси (aTEA)."],
  "femur.condyle_distal_medial": ["Медиальный мыщелок, дистальная точка",
    "Самая дистальная точка медиального мыщелка (вдоль проксимальной оси системы координат)."],
  "femur.condyle_distal_lateral": ["Латеральный мыщелок, дистальная точка",
    "Самая дистальная точка латерального мыщелка."],
  "femur.condyle_posterior_medial": ["Медиальный мыщелок, задняя точка",
    "Самая задняя точка медиального мыщелка."],
  "femur.condyle_posterior_lateral": ["Латеральный мыщелок, задняя точка",
    "Самая задняя точка латерального мыщелка."],
  "femur.notch_centre": ["Центр межмыщелковой вырезки",
    "Центр межмыщелковой вырезки у её крыши. Используется как центр коленного сустава на бедре — намеренно вместо середины трансэпикондилярной оси, чтобы фронтальные и ротационные построения не опирались на общий ориентир."],
  "femur.trochlear_groove_anterior": ["Борозда блока, передняя точка",
    "Самая глубокая точка борозды блока бедренной кости в её переднем отделе. Передний конец линии Уайтсайда."],
  "tibia.spine_medial": ["Медиальный межмыщелковый бугорок",
    "Вершина медиального межмыщелкового бугорка (медиальная ость большеберцовой кости)."],
  "tibia.spine_lateral": ["Латеральный межмыщелковый бугорок",
    "Вершина латерального межмыщелкового бугорка (латеральная ость большеберцовой кости)."],
  "tibia.pcl_insertion_midpoint": ["Центр прикрепления ЗКС",
    "Середина площадки прикрепления задней крестообразной связки к большеберцовой кости. Задний конец линии Акаги."],
  "tibia.tubercle_patellar_tendon_medial_border": ["Медиальный край связки надколенника",
    "Медиальный край прикрепления связки надколенника к бугристости большеберцовой кости. Передний конец линии Акаги в исходном определении (Akagi et al., 2004)."],
  "tibia.tubercle_medial_third": ["Бугристость, граница медиальной трети",
    "Граница медиальной и средней третей бугристости большеберцовой кости. Альтернативный ротационный ориентир, отличающийся от линии Акаги примерно на 2–4°; записывается, чтобы сравнить оба метода."],
  "tibia.plateau_medial_lowest": ["Медиальное плато, самая глубокая точка",
    "Самая дистальная точка медиальной суставной поверхности, с учётом износа. От неё отсчитывается медиальная резекция большеберцовой кости."],
  "tibia.plateau_lateral_lowest": ["Латеральное плато, самая глубокая точка",
    "Самая дистальная точка латеральной суставной поверхности."],
  "tibia.plateau_medial_anterior": ["Медиальное плато, передний край",
    "Передний край медиального плато в сагиттальной плоскости через середину отдела. Вместе с задним краем задаёт собственный задний наклон."],
  "tibia.plateau_medial_posterior": ["Медиальное плато, задний край",
    "Задний край медиального плато в сагиттальной плоскости через середину отдела."],
  "fibula.head_apex": ["Верхушка головки малоберцовой кости",
    "Верхушка головки малоберцовой кости. Дополнительный фронтальный ориентир; доступен во всех случаях, так как малоберцовая кость сегментирована вместе с большеберцовой."],
};

// A knee's targets with Russian definitions and both names: "Русское (English)".
export function translateTargets(given) {
  return {
    ...given,
    targets: given.targets.map((target) => {
      const ru = LANDMARKS[target.id];
      return ru ? { ...target, name: `${ru[0]} (${target.name})`, definition: ru[1] } : target;
    }),
  };
}
