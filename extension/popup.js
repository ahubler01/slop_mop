const enabled = document.getElementById("enabled");
const server = document.getElementById("server");
const statsTable = document.getElementById("stats");
const hint = document.getElementById("hint");

chrome.storage.local.get({ enabled: true }, (saved) => (enabled.checked = saved.enabled));
enabled.addEventListener("change", () => chrome.storage.local.set({ enabled: enabled.checked }));

chrome.runtime.sendMessage({ type: "health" }, (response) => {
  const up = response?.ok;
  server.innerHTML = up
    ? 'Server: <span class="ok">running</span>'
    : 'Server: <span class="bad">not reachable</span>';
  if (!up) hint.textContent = "Start it in the project folder: uv run python server.py";
});

chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
  if (!tab) return;
  chrome.tabs.sendMessage(tab.id, { type: "stats" }, (stats) => {
    if (chrome.runtime.lastError || !stats) {
      if (!hint.textContent) hint.textContent = "Open x.com (reload the tab after installing) to see counts.";
      return;
    }
    const rows = [["Posts scored", stats.scored], ["Shown", stats.show], ["Blurred", stats.blur], ["Hidden", stats.hide]];
    if (stats.errors) rows.push(["Errors", stats.errors]);
    statsTable.innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  });
});
