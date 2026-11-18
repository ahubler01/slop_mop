// Slop Mop content script: finds each post on x.com as it appears, asks the
// local server for show / blur / hide, and applies it.
//
// X changes its markup from time to time. Every selector we depend on is here.
(() => {
const SEL = {
  post: 'article[data-testid="tweet"]',
  text: '[data-testid="tweetText"]',
  photo: '[data-testid="tweetPhoto"]',
  photoImg: '[data-testid="tweetPhoto"] img',
  videoPoster: "video[poster]",
  permalinkTime: 'a[href*="/status/"] time',
};
const PENDING_TIMEOUT_MS = 8000; // if the server hasn't answered by then, show the post as is
const IMAGE_WAIT_RETRIES = 6; // photos load lazily: wait up to 6 x 250 ms for their URLs

const results = new Map(); // post id -> server result (X re-creates articles while you scroll)
const stats = { scored: 0, show: 0, blur: 0, hide: 0, errors: 0 };
let enabled = true;
let serverDown = false;

// --- Surviving an extension reload ---------------------------------------------
// Reloading the extension cuts this script off from it ("Extension context
// invalidated") while it keeps running in the page. background.js then injects
// a fresh copy. Each copy stamps itself as the page's owner; a cut-off copy stops,
// and only cleans up if no newer copy has taken over.
const INSTANCE = Math.random().toString(36).slice(2);
document.documentElement.dataset.slopmopOwner = INSTANCE;
let stopped = false;
let observer = null;

function alive() {
  if (stopped) return false;
  let ok = false;
  try {
    ok = Boolean(chrome.runtime?.id);
  } catch {}
  if (!ok) shutdown();
  return ok;
}

function shutdown() {
  if (stopped) return;
  stopped = true;
  observer?.disconnect();
  if (document.documentElement.dataset.slopmopOwner !== INSTANCE) return; // a newer copy is in charge
  for (const article of document.querySelectorAll(SEL.post)) {
    clearDecoration(article);
    delete article.dataset.slopmop;
  }
}

function send(message, callback) {
  if (!alive()) return;
  try {
    chrome.runtime.sendMessage(message, (response) => {
      if (stopped) return;
      callback?.(response, chrome.runtime.lastError);
    });
  } catch {
    shutdown();
  }
}

// --- Reading a post ---------------------------------------------------------

function textOf(element) {
  // innerText drops emoji, which X renders as <img alt="😂">; the model counts emoji.
  let out = "";
  const walk = (node) => {
    if (node.nodeType === Node.TEXT_NODE) out += node.textContent;
    else if (node.nodeName === "IMG") out += node.alt || "";
    else if (node.nodeName === "BR") out += "\n";
    else node.childNodes.forEach(walk);
  };
  walk(element);
  return out;
}

function postId(article) {
  const href = article.querySelector(SEL.permalinkTime)?.closest("a")?.getAttribute("href") || "";
  const match = href.match(/\/status\/(\d+)/);
  if (match) return match[1];
  // Ads have no permalink: fall back to a hash of the text.
  const text = [...article.querySelectorAll(SEL.text)].map(textOf).join("\n");
  let hash = 0;
  for (const ch of text) hash = (hash * 31 + ch.codePointAt(0)) | 0;
  return text ? `t${hash}` : null;
}

function imageUrls(article) {
  const urls = [...article.querySelectorAll(SEL.photoImg)]
    .map((img) => img.currentSrc || img.src)
    .filter((src) => src && src.includes("pbs.twimg.com/media/"));
  const posters = [...article.querySelectorAll(SEL.videoPoster)]
    .map((video) => video.getAttribute("poster"))
    .filter((src) => src && src.includes("pbs.twimg.com/"));
  return [...new Set([...urls, ...posters])];
}

function imagesStillLoading(article) {
  const photos = article.querySelectorAll(SEL.photo).length;
  return photos > 0 && article.querySelectorAll(SEL.photoImg).length < photos;
}

// --- Showing the decision ---------------------------------------------------
// State lives in data-slopmop-state, not in a class: X's React re-renders the
// article's class attribute and would wipe ours, but leaves unknown attributes alone.

const BADGES = { blur: "Unsure", hide: "Hidden" };
const HEADLINES = { blur: "Blurred by Slop Mop", hide: "Hidden by Slop Mop" };

// pipeline.REASONS sentences -> a few words each. Unknown reasons show as they come.
const SHORT_REASONS = {
  "the image looks AI-generated": "AI-generated image",
  "the wording reads like bait": "Bait wording",
  "it teases instead of telling (this / why / what)": "Teaser, no info",
  "it talks to you directly": "Talks at you",
  "it starts with a number (listicle)": "Listicle",
  "lots of ! and ?": "Lots of !?",
  "lots of CAPITALS": "ALL CAPS",
  "many hashtags": "Hashtag spam",
  "many emoji": "Emoji spam",
  "very short": "Very short",
};

const SVG_NS = "http://www.w3.org/2000/svg";
function mopIcon() {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", "slopmop-icon");
  svg.setAttribute("aria-hidden", "true");
  for (const d of [
    "M18.5 2.5 13 11",                                   // handle
    "M9 11h8l1.5 3h-11z",                                // head
    "M8.5 14 7.5 20.5M11 14l-.4 6.5M13 14v6.5M15 14l.4 6.5M17.5 14l1 6.5", // strands
  ]) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    svg.append(path);
  }
  return svg;
}

function setState(article, state) {
  if (state) article.dataset.slopmopState = state;
  else delete article.dataset.slopmopState;
}

function clearDecoration(article) {
  setState(article, null);
  article.querySelector(":scope > .slopmop-bar")?.remove();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function buildBar(article, result) {
  const bar = el("div", "slopmop-bar");

  const badge = el("div", "slopmop-badge");
  const meter = el("span", "slopmop-meter");
  meter.append(el("span", "slopmop-meter-fill"));
  meter.firstChild.style.width = `${Math.round(result.p_brainrot * 100)}%`;
  badge.append(el("span", "slopmop-badge-label", BADGES[result.decision]),
               el("span", "slopmop-p", `${Math.round(result.p_brainrot * 100)}%`), meter);
  badge.title = `P(brainrot) = ${result.p_brainrot.toFixed(3)}`
    + (result.ai_image_score != null ? `\nAI image: ${result.ai_image_score.toFixed(2)}` : "")
    + `\nBait text: ${result.text_bait_score.toFixed(2)}`;

  const headline = el("div", "slopmop-headline");
  headline.append(mopIcon(), el("span", "", HEADLINES[result.decision]));
  const reasons = el("ul", "slopmop-reasons");
  const tags = result.reasons?.length ? result.reasons.map((r) => SHORT_REASONS[r] || r) : ["No single reason"];
  for (const tag of tags) reasons.append(el("li", "", tag));
  const text = el("div", "slopmop-text");
  text.append(headline, reasons);

  const button = el("button", "slopmop-show", "Show");
  bar.append(badge, text, button);

  // The whole post is a link on X: keep our clicks from opening it.
  bar.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (!button.contains(event.target) || stopped) return;
    article.dataset.slopmopRevealed = "1";
    clearDecoration(article);
    // Revealing a hidden post means the hide was wrong: that's the false-hide signal.
    if (result.decision === "hide") {
      send({ type: "feedback", requestId: result.request_id, value: -1 });
    }
  });
  return bar;
}

function apply(article, result) {
  clearDecoration(article);
  if (!enabled || !result || result.decision === "show" || article.dataset.slopmopRevealed) return;
  setState(article, result.decision);
  article.prepend(buildBar(article, result)); // its own row on top: never over the post
}

function needsRepair(article) {
  const state = article.dataset.slopmopState;
  return (state === "blur" || state === "hide") && !article.querySelector(":scope > .slopmop-bar");
}

// Match X's theme (light, dim or dark) from the page background.
function syncTheme() {
  const [r, g, b] = (getComputedStyle(document.body).backgroundColor.match(/\d+/g) || [255, 255, 255]).map(Number);
  const theme = 0.299 * r + 0.587 * g + 0.114 * b < 128 ? "dark" : "light";
  if (document.documentElement.dataset.slopmopTheme !== theme) document.documentElement.dataset.slopmopTheme = theme;
}

// --- Scoring ----------------------------------------------------------------

function score(article, id, attempt = 0) {
  if (!alive()) return;
  if (imagesStillLoading(article) && attempt < IMAGE_WAIT_RETRIES) {
    setTimeout(() => score(article, id, attempt + 1), 250);
    return;
  }
  const post = {
    id,
    text: [...article.querySelectorAll(SEL.text)].map(textOf).join("\n"),
    images: imageUrls(article),
  };
  const timer = setTimeout(() => article.dataset.slopmopState === "pending" && setState(article, null), PENDING_TIMEOUT_MS);

  send({ type: "score", post }, (response, error) => {
    clearTimeout(timer);
    if (error || !response?.ok) {
      stats.errors++;
      serverDown = true;
      setState(article, null);
      delete article.dataset.slopmop; // try again on the next scan
      return;
    }
    serverDown = false;
    results.set(id, response.result);
    stats.scored++;
    stats[response.result.decision]++;
    // The article may have been re-used for another post while we waited.
    if (article.dataset.slopmop === id) apply(article, response.result);
  });
}

function scan() {
  if (!enabled || !alive()) return;
  syncTheme();
  for (const article of document.querySelectorAll(SEL.post)) {
    const id = postId(article);
    if (!id) continue;
    if (article.dataset.slopmop === id) {
      if (needsRepair(article) && results.has(id)) apply(article, results.get(id)); // X removed our bar
      continue;
    }
    article.dataset.slopmop = id;
    delete article.dataset.slopmopRevealed;
    if (results.has(id)) {
      apply(article, results.get(id));
    } else if (!serverDown || Math.random() < 0.1) {
      // While the server is down, only retry now and then instead of hammering it.
      setState(article, "pending");
      score(article, id);
    }
  }
}

let scheduled = false;
function scheduleScan() {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => {
    scheduled = false;
    scan();
  });
}

// --- On / off and the popup ---------------------------------------------------

function setEnabled(value) {
  enabled = value;
  for (const article of document.querySelectorAll(SEL.post)) {
    if (!enabled) clearDecoration(article);
    else delete article.dataset.slopmop; // re-apply from the cache on the next scan
  }
  if (enabled) scheduleScan();
}

chrome.storage.local.get({ enabled: true }, (saved) => setEnabled(saved.enabled));
chrome.storage.onChanged.addListener((changes) => {
  if (changes.enabled) setEnabled(changes.enabled.newValue);
});
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "stats") sendResponse({ ...stats, serverDown });
});

// Take over posts a previous (reloaded) copy decorated: their buttons are dead.
for (const article of document.querySelectorAll(SEL.post)) {
  clearDecoration(article);
  delete article.dataset.slopmop;
}
observer = new MutationObserver(scheduleScan);
observer.observe(document.body, { childList: true, subtree: true });
scheduleScan();
})();
