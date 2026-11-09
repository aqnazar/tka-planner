// What one surgeon has saved so far, by knee, mode and round: so the knee list shows
// progress on any computer. Names only the surgeon's own saves, never anyone else's.
import { getStore } from "@netlify/blobs";
import { AUTH_HASH } from "../lib/config.mjs";
import { authorised, observerKey, observerHash, reply } from "../lib/common.mjs";

export default async (request) => {
  if (!(await authorised(request.headers.get("x-study-key"), AUTH_HASH))) {
    return reply(401, { error: "Неверный пароль исследования." });
  }
  const key = observerKey(new URL(request.url).searchParams.get("observer"));
  if (!key) return reply(400, { error: "Не указано имя хирурга." });
  const prefix = `picks/${await observerHash(key)}/`;
  const store = getStore({ name: "landmark-study", consistency: "strong" });
  const { blobs } = await store.list({ prefix });
  // picks/<who>/<case>_<side>_<mode>_r<round>/<id>: the folder says it all.
  const saved = {};
  for (const { key: path } of blobs) {
    const [, , task, id] = path.split("/");
    if (!saved[task] || saved[task].last < id) {
      saved[task] = { last: id, count: (saved[task]?.count || 0) + 1 };
    } else {
      saved[task].count += 1;
    }
  }
  const complete = {};
  await Promise.all(Object.entries(saved).map(async ([task, { last }]) => {
    const record = await store.get(`${prefix}${task}/${last}`, { type: "json" });
    complete[task] = Boolean(record?.body?.complete);
  }));
  return reply(200, { saved, complete });
};

export const config = { path: "/api/progress" };
