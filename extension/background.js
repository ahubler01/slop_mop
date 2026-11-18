// Talks to the local server (server.py). The content script can't call
// http://127.0.0.1 itself from an https://x.com page, so every request goes through here.
const SERVER = "http://127.0.0.1:8765";

async function call(path, body) {
  const response = await fetch(SERVER + path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  let work;
  if (message.type === "score") {
    work = call("/score", { posts: [message.post] }).then((json) => json.results[message.post.id]);
  } else if (message.type === "feedback") {
    work = call("/feedback", { request_id: message.requestId, value: message.value });
  } else if (message.type === "health") {
    work = call("/health");
  } else {
    return false;
  }
  work
    .then((result) => sendResponse({ ok: true, result }))
    .catch((error) => sendResponse({ ok: false, error: String(error) }));
  return true; // keeps sendResponse alive for the async answer
});

// After an install or a reload, the copies of content.js already running in open
// x.com tabs are cut off from the extension. Inject a fresh copy so those tabs
// keep working without a manual page reload.
chrome.runtime.onInstalled.addListener(async () => {
  const tabs = await chrome.tabs.query({ url: ["https://x.com/*", "https://twitter.com/*"] });
  for (const tab of tabs) {
    try {
      await chrome.scripting.insertCSS({ target: { tabId: tab.id }, files: ["content.css"] });
      await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
    } catch (error) {
      console.warn(`Slop Mop: could not refresh tab ${tab.id}: ${error}`);
    }
  }
});
