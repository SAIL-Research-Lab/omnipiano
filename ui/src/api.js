const JSON_HEADERS = { "Content-Type": "application/json" };

async function responseJson(response) {
  const payload = await response.json().catch(() => ({
    ok: false,
    error: `Local API returned HTTP ${response.status}`,
  }));
  if (!response.ok) throw new Error(payload.error ?? `HTTP ${response.status}`);
  return payload;
}

export async function getCatalog() {
  return responseJson(await fetch("/api/catalog"));
}

export async function getEnvironments(track) {
  return responseJson(await fetch(`/api/environments?track=${encodeURIComponent(track)}`));
}

export async function postConfig(endpoint, config, extra = {}) {
  return responseJson(await fetch(`/api/${endpoint}`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ config, ...extra }),
  }));
}

export async function getRuns() {
  return responseJson(await fetch("/api/runs"));
}

export async function stopRun(runId) {
  return responseJson(await fetch("/api/stop", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ run_id: runId }),
  }));
}
