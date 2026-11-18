"""
Local scoring server for the Slop Mop browser extension (extension/).

    uv run python server.py          # listens on http://127.0.0.1:8765

The extension sends the posts it sees on x.com; this server downloads their
images from X's image CDN, runs pipeline.score_post on each post, logs it like
the Gradio app does (a hash, never the text), and answers show / blur / hide.

    GET  /health                      -> {"ok": true}
    POST /score     {"posts": [{"id", "text", "images": [url, ...]}]}
                    -> {"results": {post_id: {"decision", "p_brainrot", "reasons", "request_id", ...}}}
    POST /feedback  {"request_id", "value"}   value: -1 = "this hide was wrong", +1 = "good call"

Only 127.0.0.1 is bound, so nothing outside this machine can reach it.
"""

import io
import json
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import env
import logbook
import pipeline

HOST, PORT = "127.0.0.1", 8765
IMAGE_HOST = "pbs.twimg.com"      # X's image CDN: the only host we download from
MAX_IMAGES_PER_POST = 4
MAX_IMAGE_BYTES = 8 * 1024 * 1024
CACHE_SIZE = 5000                 # post id -> result, so scrolling back costs nothing

_model_lock = threading.Lock()    # one post through CLIP at a time
_cache: OrderedDict[str, dict] = OrderedDict()
_cache_lock = threading.Lock()
_downloads = ThreadPoolExecutor(max_workers=8)


def image_url(url: str) -> str | None:
    """X's CDN URL at the 'medium' size (about 1200 px, close to what the probe was trained on), or None if not X's CDN."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname != IMAGE_HOST:
        return None
    query = dict(urllib.parse.parse_qsl(parts.query))
    if "name" in query:
        query["name"] = "medium"
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def download(url: str) -> io.BytesIO | None:
    safe = image_url(url)
    if safe is None:
        return None
    try:
        request = urllib.request.Request(safe, headers={"User-Agent": "slop-mop-local/1.0"})
        with urllib.request.urlopen(request, timeout=8) as response:
            data = response.read(MAX_IMAGE_BYTES + 1)
        return io.BytesIO(data) if len(data) <= MAX_IMAGE_BYTES else None
    except Exception as error:  # a missing image should not block the post's text score
        print(f"image download failed: {error}")
        return None


def score(post: dict) -> dict:
    post_id = str(post.get("id") or "")
    with _cache_lock:
        if post_id and post_id in _cache:
            _cache.move_to_end(post_id)
            return _cache[post_id]

    started = time.perf_counter()
    urls = [u for u in post.get("images", []) if isinstance(u, str)][:MAX_IMAGES_PER_POST]
    images = [image for image in _downloads.map(download, urls) if image is not None]
    with _model_lock:
        record = pipeline.score_post(str(post.get("text") or ""), images)
    record = logbook.log_request(record | {"latency_s": round(time.perf_counter() - started, 3), "source": "x-extension"})

    fused = record["outputs"]["fusion"]
    result = {
        "request_id": record["id"],
        "decision": fused["decision"],
        "p_brainrot": fused["p_brainrot"],
        "reasons": fused["reasons"],
        "ai_image_score": record["outputs"]["vision"]["ai_image_score"],
        "text_bait_score": fused["text_bait_score"],
        "latency_s": record["latency_s"],
    }
    if post_id:
        with _cache_lock:
            _cache[post_id] = result
            while len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
    return result


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/score":
                posts = body.get("posts", [])[:50]
                return self._send(200, {"results": {str(p.get("id")): score(p) for p in posts}})
            if self.path == "/feedback":
                value = int(body["value"])
                if value not in (-1, 1):
                    raise ValueError("value must be -1 or 1")
                logbook.log_feedback(str(body["request_id"]), value)
                return self._send(200, {"ok": True})
            self._send(404, {"error": "not found"})
        except Exception as error:
            print(f"error on {self.path}: {error!r}")
            self._send(400, {"error": str(error)})

    def log_message(self, fmt, *args):  # quieter than the default one-line-per-request
        if not str(args[1] if len(args) > 1 else "").startswith("2"):
            super().log_message(fmt, *args)


def main() -> None:
    env.load_env()
    print(logbook.start())
    print("Loading models (CLIP + probes)...")
    pipeline.models()
    pipeline.fusion("warm-up", pipeline.vision([]))   # loads CLIP now, not on the first post; not logged
    print(f"Slop Mop server ready on http://{HOST}:{PORT}. Load extension/ in Chrome and open x.com.")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
