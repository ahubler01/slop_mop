"""
Every assumption the pipeline relies on, in one place.

data/fetch_data.py, train.py and pipeline.py all import from here, so the costs
used to pick the thresholds in training are exactly the costs the app uses.
Change a figure here and rerun train.py to see how the answer moves.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
IMAGE_DIR = DATA_DIR / "images"          # git-ignored, re-created by data/fetch_data.py
IMAGES_CSV = DATA_DIR / "images.csv"     # one row per kept image: file, label, generator
POSTS_CSV = DATA_DIR / "posts.csv"       # Webis-Clickbait-17 tweets and their crowd labels
ARTIFACTS = ROOT / "artifacts"
FIG_DIR = ARTIFACTS / "figures"
SEED = 42

# --- Data ---------------------------------------------------------------------
# OpenFake (ComplexDataLab/OpenFake, CC-BY-SA-4.0): real photos and images from
# 2025-26 generators. We read whole row groups of the test split (148 images
# each, generators mixed inside every group) until we have enough per generator.
OPENFAKE_FILE = "datasets/ComplexDataLab/OpenFake/core/test-00000-of-00013.parquet"
REAL_SOURCES = ["imagenet", "docci"]
GENERATORS = ["z-image-turbo", "flux.2-klein-9b", "illustrious", "gpt-image-1.5", "midjourney-7"]
MAX_PER_SOURCE = {"real": 900, "fake": 200}   # per generator for fakes, all real sources together
MAX_ROW_GROUPS = 48                           # the whole shard, ~5 GB; midjourney-7 is rare, so it ends up with fewer
# In-the-wild stress test, never used for training: real and AI images that
# people posted on Reddit (OpenFake "reddit" config), closer to a real feed.
WILD_FILE = "datasets/ComplexDataLab/OpenFake/reddit/test-00000-of-00005.parquet"
WILD_ROW_GROUPS = 14   # ~1.5 GB; only ~14% of these are real, so we need many to get ~150 real photos
WILD_CSV = ROOT / "data" / "raw" / "wild.csv"
MAX_IMAGE_SIDE = 1024                         # X serves feed images at about this size

# Webis-Clickbait-17 (Zenodo 5530410, CC-BY-4.0): ~19.5k tweets from news
# accounts, each judged by 5 annotators on a 0-1 clickbait scale.
WEBIS_URL = "https://zenodo.org/api/records/5530410/files/clickbait17-train-170630.zip/content"
CLICKBAIT_IF_MEAN_AT_LEAST = 0.5              # the corpus' own cut-off for truthClass

# --- Vision component: frozen CLIP + linear probe -----------------------------
CLIP_MODEL = "openai/clip-vit-base-patch32"   # small enough for a free CPU Space
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
# X re-encodes every uploaded image as JPEG. We test the probe on the same
# images re-encoded at falling quality, and train a second probe on a random
# mix of qualities to see whether that fixes it.
JPEG_QUALITIES = [95, 85, 70, 50, 30]
X_JPEG_QUALITY = 85          # roughly what X uses; the feed and the app score images at this quality
JPEG_TRAIN_MIX = [95, 85, 70, 50]
FALSE_ALARM_RATE = 0.05      # vision metric: share of AI images caught when 5% of real photos are flagged

# --- The simulated feed -------------------------------------------------------
# No public dataset has X posts with labelled images AND labelled text, so each
# Webis tweet gets an image from the held-out OpenFake pool. These two shares
# set how the feed looks; they are assumptions, not measurements.
SHARE_WITH_IMAGE = 0.6       # share of posts that carry an image
SHARE_AI_OF_IMAGES = 0.3     # share of those images that are AI-generated

# A post is "brainrot" (the user wants it hidden) if its text is clickbait OR
# its image is AI-generated. This is the product definition, not a finding.

# --- Costs, in seconds of the user's time per post ----------------------------
COST_MISSED = 6.0        # brainrot post shown: the user reads it, maybe clicks, ~6 s gone
COST_FALSE_HIDE = 20.0   # good post hidden: rarely noticed, and each one erodes trust in the filter
COST_BLUR = 1.5          # post blurred with a reason: the user decides in a glance, either way
COST_TN = 0.0
COST_TP = 0.0

# Hiding without a blur band would be worth it above this probability:
#   p * COST_MISSED > (1 - p) * COST_FALSE_HIDE  <=>  p > 20 / 26 = 0.77
SINGLE_THRESHOLD = COST_FALSE_HIDE / (COST_FALSE_HIDE + COST_MISSED)

# --- Value -------------------------------------------------------------------
POSTS_PER_DAY = 300          # posts an active user scrolls past in a day
EUR_PER_HOUR = 15.0          # what an hour of attention is worth to the user

# Baseline: what an X user can do today, mute words. A list a motivated user
# would plausibly type into Settings > Muted words.
MUTE_WORDS = [
    "you won't believe", "will blow your mind", "shocking", "this is why", "here's why",
    "what happened next", "omg", "viral", "must see", "insane", "unbelievable",
    "ai art", "midjourney", "#ai", "genius", "can't stop", "need to know",
]
