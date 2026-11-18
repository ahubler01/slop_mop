"""
The product: one X post in, show / blur / hide out.

    image ──► VISION: CLIP + probe ──► P(AI image) ─┐
    text ───► CLIP text score + simple features ────┴─► TABULAR fusion ──► P(brainrot) ──► show / blur / hide

The vision component's output is a column of the tabular component's input:
the fusion model never sees pixels, only the probability the probe gives.

`raw_input` follows the template's convention (text, or a file path), with one
addition so a post can carry both: a last line `image: <path>`. A bare image
path is a post with no text.

    "You won't believe what this cat did\\nimage: eval/inputs/cat.jpg"

train.py writes every model and threshold this file loads into artifacts/models/.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

import components
import config
import decision

MODELS = config.ARTIFACTS / "models"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

VERSIONS = {
    "vision": f"{config.CLIP_MODEL}+logreg-probe-jpegmix-reddit-v1",
    "fusion": "logreg-fusion-v1",
    "decision": "cost-band-v1",
}

REASONS = {
    "image_ai_score": "the image looks AI-generated",
    "text_bait_score": "the wording reads like bait",
    "teaser_words": "it teases instead of telling (this / why / what)",
    "second_person": "it talks to you directly",
    "starts_with_number": "it starts with a number (listicle)",
    "n_punct": "lots of ! and ?",
    "share_caps": "lots of CAPITALS",
    "n_hashtags": "many hashtags",
    "n_emoji": "many emoji",
    "n_words": "very short",
}


@lru_cache(maxsize=1)
def models():
    if not (MODELS / "config.json").is_file():
        raise RuntimeError("No trained models in artifacts/models/. Run `uv run python train.py` first.")
    settings = json.loads((MODELS / "config.json").read_text())
    return (joblib.load(MODELS / "vision_probe.joblib"), joblib.load(MODELS / "text_probe.joblib"),
            joblib.load(MODELS / "fusion.joblib"), settings)


def read_input(raw_input: str) -> tuple[str, Path | None]:
    """(text, image path or None)."""
    raw = str(raw_input).strip()
    if len(raw) < 300 and Path(raw).suffix.lower() in IMAGE_SUFFIXES and Path(raw).is_file():
        return "", Path(raw)
    lines = raw.splitlines()
    if lines and lines[-1].lower().startswith("image:"):
        path = Path(lines[-1].split(":", 1)[1].strip())
        return "\n".join(lines[:-1]).strip(), path if path.is_file() else None
    return raw, None


def vision(images: list) -> dict:
    """Component 1: probability the image is AI-generated, scored as X would serve it.

    `images` holds paths or file-like objects. A post with several images gets the
    highest score: one AI image is enough to call the post AI.
    """
    if not images:
        return {"has_image": False, "ai_image_score": None}
    probe, _, _, settings = models()
    pictures = [components.as_served_by_x(Image.open(image), settings["x_jpeg_quality"]) for image in images]
    score = float(probe.predict_proba(components.embed_images(pictures))[:, 1].max())
    return {"has_image": True, "ai_image_score": round(score, 4)}


def fusion(text: str, v: dict) -> dict:
    """Component 2: P(brainrot) from the vision score and the text, then the decision."""
    _, text_probe, fusion_model, settings = models()
    row = {
        "image_ai_score": v["ai_image_score"] or 0.0,
        "has_image": float(v["has_image"]),
        "text_bait_score": float(text_probe.predict_proba(components.embed_texts([text or " "]))[0, 1]),
        **components.text_features(text),
    }
    x = np.array([[row[f] for f in settings["fusion_features"]]])
    p = float(fusion_model.predict_proba(x)[0, 1])
    verdict = str(decision.decide([p], settings["t_low"], settings["t_high"])[0])

    # Reasons for a blur or hide: the features that pushed this post's log-odds up the most.
    scaler, logreg = fusion_model[0], fusion_model[-1]
    push = (scaler.transform(x)[0]) * logreg.coef_.ravel()
    ranked = sorted(zip(settings["fusion_features"], push), key=lambda kv: -kv[1])
    reasons = [REASONS[f] for f, value in ranked if value > 0.3 and f in REASONS][:2] if verdict != "show" else []
    return {"p_brainrot": round(p, 4), "decision": verdict, "reasons": reasons,
            "text_bait_score": round(row["text_bait_score"], 4), "features": row}


def run(raw_input: str) -> dict:
    text, image_path = read_input(raw_input)
    return score_post(text, [image_path] if image_path else [])


def score_post(text: str, images: list) -> dict:
    """One post (its text and 0+ images) in, the request record out. server.py calls this directly."""
    v = vision(images)
    f = fusion(text, v)
    words = len(text.split())
    return {
        # No post text, image or handle is ever logged: only a hash and a shape.
        "input_summary": f"post {hashlib.sha256(text.encode()).hexdigest()[:10]}: {words} words, "
                         f"{'with' if v['has_image'] else 'no'} image",
        "features": {
            "n_words": float(words),
            "share_caps": f["features"]["share_caps"],
            "has_image": float(v["has_image"]),
            "image_ai_score": f["features"]["image_ai_score"],
            "text_bait_score": f["text_bait_score"],
            "confidence": abs(f["p_brainrot"] - 0.5) * 2,
        },
        "outputs": {"vision": v, "fusion": {k: f[k] for k in ["p_brainrot", "decision", "reasons", "text_bait_score"]}},
        "versions": VERSIONS,
        "review": f["decision"] == "blur",   # unsure: blurred with its reason, the user decides
        "tokens": 0,
        "cost_eur": 0.0,                     # both components run locally on CPU
    }
