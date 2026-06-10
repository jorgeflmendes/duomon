export async function post(path, payload = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  let body = null;
  try {
    body = await response.json();
  } catch (_error) {
    body = null;
  }
  if (!response.ok) {
    throw new Error(body?.error || `${path} failed: ${response.status}`);
  }
  return body || {};
}

export async function getJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} failed: ${response.status}`);
  return response.json();
}

export function latestJob(payload) {
  const jobs = Object.values(payload.jobs || {});
  jobs.sort((a, b) => (b.started_at || 0) - (a.started_at || 0));
  return jobs[0] || null;
}
