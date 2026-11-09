// Shared by the study functions.

export function reply(status, payload) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}

export async function sha256Hex(text) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// Compare the hash of what was sent with the hash on file, in constant time.
export async function authorised(sent, expectedHash) {
  if (!sent || typeof sent !== "string" || sent.length > 256) return false;
  const got = await sha256Hex(sent);
  let diff = got.length ^ expectedHash.length;
  for (let i = 0; i < got.length && i < expectedHash.length; i += 1) {
    diff |= got.charCodeAt(i) ^ expectedHash.charCodeAt(i);
  }
  return diff === 0;
}

// The same normalisation as tka_planner.study.observer_key: case, spacing and Unicode
// form do not make a different surgeon.
export function observerKey(name) {
  return String(name ?? "").normalize("NFKC").trim().split(/\s+/).join(" ").toLowerCase();
}

export async function observerHash(key) {
  return (await sha256Hex(key)).slice(0, 16);
}
