// Every save, for the author's pull (online/pull.py). Needs the admin key, which only
// the author's machine has; surgeons cannot read each other's picks.
//   GET  -> { keys: [...] }                every save's key
//   POST { keys: [...] } (<= 100) -> { records: [{ key, value }] }
import { getStore } from "@netlify/blobs";
import { ADMIN_HASH } from "../lib/config.mjs";
import { authorised, reply } from "../lib/common.mjs";

export default async (request) => {
  if (!(await authorised(request.headers.get("x-admin-key"), ADMIN_HASH))) {
    return reply(401, { error: "Not the admin key." });
  }
  const store = getStore({ name: "landmark-study", consistency: "strong" });
  if (request.method === "GET") {
    const { blobs } = await store.list({ prefix: "picks/" });
    return reply(200, { keys: blobs.map((b) => b.key) });
  }
  const { keys } = await request.json();
  if (!Array.isArray(keys) || keys.length > 100 ||
      !keys.every((k) => typeof k === "string" && k.startsWith("picks/"))) {
    return reply(400, { error: "Up to 100 save keys." });
  }
  const records = await Promise.all(
    keys.map(async (key) => ({ key, value: await store.get(key, { type: "json" }) })));
  return reply(200, { records: records.filter((r) => r.value) });
};

export const config = { path: "/api/list" };
