// Browser storage is scoped to the origin, including the bedside port.
async function bedsideRequest(path, options = {}) {
  const headers = new Headers(options.headers);
  const token = localStorage.getItem("bedside-session");
  if (token) headers.set("Authorization", "Bearer " + token);
  return fetch(path, { ...options, headers, cache: "no-store" });
}
