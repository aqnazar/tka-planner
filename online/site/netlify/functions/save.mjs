// One surgeon's picks on one knee, kept as sent. Never overwritten: every save is a new
// blob, and the study's pull (online/pull.py) writes them through the planner's own
// save_picks on the author's machine. Only those who know the passphrase can save.
import { getStore } from "@netlify/blobs";
import { AUTH_HASH, KNEES, LANDMARKS, MODES } from "../lib/config.mjs";
import { authorised, observerKey, observerHash, reply } from "../lib/common.mjs";

const MAX_BODY = 64 * 1024;

export default async (request) => {
  if (request.method !== "POST") return reply(405, { error: "POST only." });
  if (!(await authorised(request.headers.get("x-study-key"), AUTH_HASH))) {
    return reply(401, { error: "The study passphrase is not right." });
  }
  const text = await request.text();
  if (text.length > MAX_BODY) return reply(413, { error: "Too large." });
  let sent;
  try { sent = JSON.parse(text); } catch { return reply(400, { error: "Not JSON." }); }

  const { case_id: caseId, side, body } = sent || {};
  if (!KNEES.has(`${caseId}:${side}`)) return reply(400, { error: "Unknown knee." });
  const problem = check(body);
  if (problem) return reply(400, { error: problem });

  const key = observerKey(body.observer);
  const received = new Date().toISOString();
  const id = `${received.replace(/[-:.]/g, "")}-${crypto.randomUUID().slice(0, 8)}`;
  const path = `picks/${await observerHash(key)}/${caseId}_${side}_${body.mode}` +
    `_r${body.round}/${id}`;
  const store = getStore({ name: "landmark-study", consistency: "strong" });
  await store.setJSON(path, {
    received_utc: received, case_id: caseId, side, source: "online", body,
  });
  const placed = body.picks.filter((p) => p.position_mm).length;
  return reply(200, { saved: id, placed, skipped: body.picks.length - placed });
};

function check(body) {
  if (!body || typeof body !== "object") return "No picks were sent.";
  if (!observerKey(body.observer) || String(body.observer).length > 200) {
    return "Enter the surgeon's name.";
  }
  if (!Number.isInteger(body.round) || body.round < 1 || body.round > 20) return "Bad round.";
  if (!MODES.has(body.mode)) return "Bad mode.";
  if (!Array.isArray(body.picks) || !body.picks.length || body.picks.length > 40) {
    return "No picks were sent.";
  }
  for (const pick of body.picks) {
    if (!LANDMARKS.has(pick?.id)) return `Not a landmark of the study: ${pick?.id}`;
    const at = pick.position_mm;
    if (at !== null && at !== undefined &&
        !(Array.isArray(at) && at.length === 3 && at.every(Number.isFinite))) {
      return `Bad position for ${pick.id}.`;
    }
  }
  return null;
}

export const config = { path: "/api/save" };
