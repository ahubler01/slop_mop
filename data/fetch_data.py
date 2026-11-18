"""
One-off download of the raw data. NOT part of training: train.py only reads
what this script writes, so training never depends on the servers being up.

    uv run python data/fetch_data.py

1. OpenFake: reads row groups of one test shard until every generator in
   config.GENERATORS has enough images, drops every image where OpenCV finds a
   face (no personal data anywhere, including training), and saves the
   original bytes to data/raw/images/ with a table data/raw/images.csv.
2. Webis-Clickbait-17: downloads the labelled training zip from Zenodo, keeps
   the post text and the annotators' mean score, writes data/raw/posts.csv.
"""

import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

FACES = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def has_face(picture: Image.Image) -> bool:
    grey = cv2.cvtColor(np.array(picture.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return len(FACES.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))) > 0


def fetch_openfake() -> None:
    config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    wanted = {g: config.MAX_PER_SOURCE["fake"] for g in config.GENERATORS}
    wanted["real"] = config.MAX_PER_SOURCE["real"]
    rows, dropped_faces, first_group = [], 0, 0
    if config.IMAGES_CSV.exists():  # resume: keep what is already downloaded
        rows = pd.read_csv(config.IMAGES_CSV).to_dict("records")
        first_group = max(int(r.get("row_group", 15)) for r in rows) + 1
        for r in rows:
            wanted["real" if r["generator"] in config.REAL_SOURCES else r["generator"]] -= 1

    with HfFileSystem().open(config.OPENFAKE_FILE) as f:
        shard = pq.ParquetFile(f)
        for group in range(first_group, min(shard.num_row_groups, config.MAX_ROW_GROUPS)):
            if all(n <= 0 for n in wanted.values()):
                break
            table = shard.read_row_group(group, columns=["image", "label", "model"]).to_pylist()
            for row in table:
                source = "real" if row["model"] in config.REAL_SOURCES else row["model"]
                if wanted.get(source, 0) <= 0:
                    continue
                data = row["image"]["bytes"]
                picture = Image.open(io.BytesIO(data))
                if has_face(picture):
                    dropped_faces += 1
                    continue
                name = f"{len(rows):05d}.{(picture.format or 'png').lower()}"
                (config.IMAGE_DIR / name).write_bytes(data)
                rows.append({
                    "file": name, "label": row["label"], "generator": row["model"],
                    "format": picture.format, "width": picture.width, "height": picture.height,
                    "row_group": group,
                })
                wanted[source] -= 1
            print(f"  row group {group}: {len(rows)} kept, {dropped_faces} dropped for faces", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(config.IMAGES_CSV, index=False)
    print(f"Wrote {config.IMAGES_CSV} ({len(table)} images)")
    print(table.groupby(["label", "generator", "format"]).size().to_string())


def fetch_wild() -> None:
    """A few row groups of OpenFake's Reddit test set, faces dropped, as wild_*.* files."""
    config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with HfFileSystem().open(config.WILD_FILE) as f:
        shard = pq.ParquetFile(f)
        for group in range(config.WILD_ROW_GROUPS):
            for row in shard.read_row_group(group, columns=["image", "label"]).to_pylist():
                data = row["image"]["bytes"]
                picture = Image.open(io.BytesIO(data))
                if has_face(picture):
                    continue
                name = f"wild_{len(rows):05d}.{(picture.format or 'png').lower()}"
                (config.IMAGE_DIR / name).write_bytes(data)
                rows.append({"file": name, "label": row["label"], "format": picture.format})
    table = pd.DataFrame(rows)
    table.to_csv(config.WILD_CSV, index=False)
    print(f"Wrote {config.WILD_CSV} ({len(table)} images)")
    print(table.groupby(["label", "format"]).size().to_string())


def fetch_webis() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = config.DATA_DIR / "webis-clickbait-17.zip"  # git-ignored, kept so it is downloaded once
    if not zip_path.exists():
        print("Downloading Webis-Clickbait-17 (~0.9 GB, once)...", flush=True)
        zip_path.write_bytes(urlopen(config.WEBIS_URL).read())
    archive = zipfile.ZipFile(zip_path)
    read = lambda name: [json.loads(line) for line in archive.read(name).decode().splitlines()]  # noqa: E731
    folder = archive.namelist()[0].split("/")[0]
    posts = pd.DataFrame(read(f"{folder}/instances.jsonl"))
    truth = pd.DataFrame(read(f"{folder}/truth.jsonl"))
    table = posts.merge(truth, on="id")
    table = pd.DataFrame({
        "id": table["id"],
        "text": table["postText"].str[0].fillna(""),
        "clickbait_mean": table["truthMean"].round(3),
    })
    table = table[table["text"].str.strip() != ""]
    table.to_csv(config.POSTS_CSV, index=False)
    print(f"Wrote {config.POSTS_CSV} ({len(table)} posts, "
          f"{(table.clickbait_mean >= config.CLICKBAIT_IF_MEAN_AT_LEAST).mean():.0%} clickbait)")


if __name__ == "__main__":
    if not config.POSTS_CSV.exists():
        fetch_webis()
    fetch_openfake()
    if not config.WILD_CSV.exists():
        fetch_wild()
