"""
The pieces train.py and pipeline.py share, so a post is turned into numbers
the same way in training and in the app.

- CLIP: one frozen encoder for both images and text. Nothing in it is trained;
  we only fit small logistic regressions on top of its embeddings.
- as_served_by_x: what X does to an uploaded image (resize, re-encode as JPEG).
- text_features: one simple feature per idea of what makes a post bait.
"""

import io
import re
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

import config


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def clip():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(device()).eval()
    return model, CLIPProcessor.from_pretrained(config.CLIP_MODEL)


def as_served_by_x(picture: Image.Image, quality: int) -> Image.Image:
    """Shrink to X's display size and re-encode as JPEG at `quality`."""
    picture = picture.convert("RGB")
    picture.thumbnail((config.MAX_IMAGE_SIDE, config.MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def _unit(x):
    """Unit-length CLIP embedding. transformers 5 wraps it in an output object (pooler_output)."""
    x = getattr(x, "pooler_output", x)
    return x / x.norm(dim=-1, keepdim=True)


@torch.no_grad()
def embed_images(pictures, batch=32) -> np.ndarray:
    model, processor = clip()
    out = []
    for i in range(0, len(pictures), batch):
        inputs = processor(images=[p.convert("RGB") for p in pictures[i:i + batch]], return_tensors="pt").to(device())
        out.append(_unit(model.get_image_features(**inputs)).cpu().numpy())
    return np.vstack(out).astype(np.float32)


@torch.no_grad()
def embed_texts(texts, batch=256) -> np.ndarray:
    model, processor = clip()
    out = []
    for i in range(0, len(texts), batch):
        inputs = processor(text=list(texts[i:i + batch]), return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to(device())
        out.append(_unit(model.get_text_features(**inputs)).cpu().numpy())
    return np.vstack(out).astype(np.float32)


EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
SECOND_PERSON = re.compile(r"\b(you|your|you're|you'll|yours)\b", re.I)
TEASER = re.compile(r"\b(this|these|here's|why|what|how|when|who)\b", re.I)

TEXT_FEATURES = ["n_words", "share_caps", "n_punct", "starts_with_number", "second_person",
                 "teaser_words", "n_hashtags", "n_emoji"]


def text_features(text: str) -> dict[str, float]:
    """One number per idea. Kept simple so each coefficient can be read as a reason."""
    words = text.split()
    letters = [c for c in text if c.isalpha()]
    return {
        "n_words": float(len(words)),                                                      # short teasers
        "share_caps": sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,  # SHOUTING
        "n_punct": float(text.count("!") + text.count("?")),                               # !!! and ???
        "starts_with_number": float(bool(re.match(r"^\s*\d+", text))),                     # "17 things..."
        "second_person": float(len(SECOND_PERSON.findall(text))),                          # "you won't believe"
        "teaser_words": float(len(TEASER.findall(text))),                                  # "this is why"
        "n_hashtags": float(text.count("#")),
        "n_emoji": float(len(EMOJI.findall(text))),
    }
