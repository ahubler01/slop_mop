"""
Trains both components and the decision rule, and writes what pipeline.py needs.

    uv run python data/fetch_data.py   # once
    uv run python train.py

1. Vision. CLIP image embeddings + a logistic-regression probe: is this image
   AI-generated? Before it goes into the product we ask it three questions:
     a. Does it survive X re-encoding every image as JPEG?
     b. Does it catch generators it has never seen? New ones appear every month.
     c. Is its score a probability? The next component reads it as one.
     d. Does it hold on images people really post (Reddit), not just the
        benchmark? Probe C adds half of those to training; the product uses C.
2. Feed. Each Webis tweet gets an image from the held-out OpenFake pool, so we
   have posts with a labelled text AND a labelled image.
3. Fusion (tabular). A logistic regression on the image score, a CLIP text
   score and simple text features gives P(brainrot) per post.
4. Decision. Show / blur / hide thresholds chosen on validation posts by
   minimising the user's lost seconds, then priced once on test posts against
   "show everything" and X's mute words.
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import components
import config
import decision

CACHE = config.ARTIFACTS / "cache"
MODELS = config.ARTIFACTS / "models"
CONDITIONS = ["original"] + [f"jpeg{q}" for q in config.JPEG_QUALITIES]
X_LIKE = f"jpeg{config.X_JPEG_QUALITY}"
FUSION_FEATURES = ["image_ai_score", "has_image", "text_bait_score"] + components.TEXT_FEATURES
BLUE, GREY, RED = "#3b6fb6", "#b0b0b0", "#c8553d"


def save_fig(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def three_way_split(frame, stratify):
    """60 / 20 / 20, stratified, fixed seed."""
    train, rest = train_test_split(frame, test_size=0.4, stratify=frame[stratify], random_state=config.SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest[stratify], random_state=config.SEED)
    return frame.assign(split=pd.Series("train", index=frame.index).mask(frame.index.isin(val.index), "val")
                        .mask(frame.index.isin(test.index), "test"))


def fit_logreg(X, y, X_val, y_val):
    """Logistic regression with C chosen on validation log loss (it rewards calibrated probabilities)."""
    def fit(C):
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X, y)
    losses = {C: log_loss(y_val, fit(C).predict_proba(X_val)[:, 1]) for C in config.C_GRID}
    best = min(losses, key=losses.get)
    return fit(best), best


def caught_at_false_alarm_rate(p_real_val, p_real, p_fake):
    """Share of AI images caught when the threshold flags FALSE_ALARM_RATE of real validation photos."""
    threshold = np.quantile(p_real_val, 1 - config.FALSE_ALARM_RATE)
    return float(np.mean(p_fake >= threshold)), float(np.mean(p_real >= threshold))


# --- 1. Vision ---------------------------------------------------------------

def image_embeddings(images):
    """CLIP embedding of every image, as stored and as X would serve it at each JPEG quality."""
    path = CACHE / "image_embeddings.npz"
    if path.exists():
        cached = np.load(path)
        if all(c in cached and len(cached[c]) == len(images) for c in CONDITIONS):
            return {c: cached[c] for c in CONDITIONS}
    out = {c: [] for c in CONDITIONS}
    for start in range(0, len(images), 32):
        print(f"  embedding images {start}/{len(images)}", end="\r", flush=True)
        pictures = [Image.open(config.IMAGE_DIR / f).convert("RGB") for f in images.file.iloc[start:start + 32]]
        out["original"].append(components.embed_images(pictures))
        for q in config.JPEG_QUALITIES:
            out[f"jpeg{q}"].append(components.embed_images([components.as_served_by_x(p, q) for p in pictures]))
    out = {c: np.vstack(v) for c, v in out.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def wild_embeddings(wild):
    path = CACHE / "wild_embeddings.npy"
    if path.exists() and len(np.load(path)) == len(wild):
        return np.load(path)
    out = []
    for start in range(0, len(wild), 32):
        pictures = [Image.open(config.IMAGE_DIR / f) for f in wild.file.iloc[start:start + 32]]
        out.append(components.embed_images([components.as_served_by_x(p, config.X_JPEG_QUALITY) for p in pictures]))
    out = np.vstack(out)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def vision(images, emb):
    rng = np.random.default_rng(config.SEED)
    idx = {s: np.flatnonzero(images.split == s) for s in ["train", "val", "test"]}
    y = images.fake.to_numpy()

    # The format shortcut: if every real photo is a JPEG and every fake a PNG,
    # a probe can score well by spotting compression instead of generation.
    print("\nFile format by label (a probe could learn this instead of 'AI or not'):")
    print(pd.crosstab(images.label, images.format).to_string())

    # Probe A learns from images as stored. Probe B learns from each training
    # image re-encoded at a random JPEG quality, the way it will look on X.
    mix = rng.choice(config.JPEG_TRAIN_MIX, size=len(idx["train"]))
    X_mix = np.stack([emb[f"jpeg{q}"][i] for q, i in zip(mix, idx["train"])])
    probe_a, c_a = fit_logreg(emb["original"][idx["train"]], y[idx["train"]], emb["original"][idx["val"]], y[idx["val"]])
    probe_b, c_b = fit_logreg(X_mix, y[idx["train"]], emb[X_LIKE][idx["val"]], y[idx["val"]])
    print(f"\nProbe A (as stored) C = {c_a}, probe B (JPEG mix) C = {c_b}")

    # a. JPEG robustness: same test images, worse and worse compression.
    real_val, real_test, fake_test = (idx["val"][y[idx["val"]] == 0], idx["test"][y[idx["test"]] == 0],
                                      idx["test"][y[idx["test"]] == 1])
    rows = []
    for name, probe, val_condition in [("A: trained as stored", probe_a, "original"), ("B: trained on JPEG mix", probe_b, X_LIKE)]:
        p_val_real = probe.predict_proba(emb[val_condition][real_val])[:, 1]
        for c in CONDITIONS:
            p = probe.predict_proba(emb[c])[:, 1]
            caught, false_alarm = caught_at_false_alarm_rate(p_val_real, p[real_test], p[fake_test])
            rows.append({"probe": name, "condition": c, "auc": roc_auc_score(y[idx["test"]], p[idx["test"]]),
                         "caught": caught, "false_alarm": false_alarm})
    jpeg = pd.DataFrame(rows)
    print("\nJPEG robustness on the test images:")
    print(jpeg.pivot(index="condition", columns="probe", values="caught").loc[CONDITIONS].round(3).to_string())

    # b. Unseen generators: retrain probe B without one generator, test on it.
    rows = []
    for g in config.GENERATORS:
        keep = images.generator.to_numpy() != g
        tr, va = idx["train"][keep[idx["train"]]], idx["val"][keep[idx["val"]]]
        mix_g = rng.choice(config.JPEG_TRAIN_MIX, size=len(tr))
        held_out, _ = fit_logreg(np.stack([emb[f"jpeg{q}"][i] for q, i in zip(mix_g, tr)]), y[tr], emb[X_LIKE][va], y[va])
        g_test = idx["test"][images.generator.to_numpy()[idx["test"]] == g]
        for name, probe in [("seen in training", probe_b), ("never seen", held_out)]:
            p = probe.predict_proba(emb[X_LIKE])[:, 1]
            caught, _ = caught_at_false_alarm_rate(p[real_val], p[real_test], p[g_test])
            rows.append({"generator": g, "probe": name, "caught": caught, "n_test": len(g_test)})
    unseen = pd.DataFrame(rows)
    print("\nAI images caught, per generator, seen vs never seen in training:")
    print(unseen.pivot(index="generator", columns="probe", values="caught").round(3).to_string())

    # c. Calibration of probe B as X serves images: the fusion reads its score as a probability.
    p_test = probe_b.predict_proba(emb[X_LIKE][idx["test"]])[:, 1]
    bins = pd.cut(p_test, np.linspace(0, 1, 6), include_lowest=True)
    reliability = pd.DataFrame({"predicted": p_test, "actual": y[idx["test"]]}).groupby(bins, observed=True).agg(
        n=("actual", "size"), mean_predicted=("predicted", "mean"), share_fake=("actual", "mean"))
    print(f"\nProbe B calibration on test ({X_LIKE}): Brier {brier_score_loss(y[idx['test']], p_test):.3f}")
    print(reliability.round(3).to_string())

    # d. In the wild: real and AI images people posted on Reddit. Half of them
    # are only ever used for testing. Probe C is probe B plus the other half,
    # to see whether a little in-domain data fixes what the clean split hides.
    wild = pd.read_csv(config.WILD_CSV)
    wild["fake"] = (wild.label == "fake").astype(int)
    wild_emb = wild_embeddings(wild)
    w_train, w_test = train_test_split(np.arange(len(wild)), test_size=0.5, stratify=wild.fake, random_state=config.SEED)
    probe_c, c_c = fit_logreg(np.vstack([X_mix, wild_emb[w_train]]), np.concatenate([y[idx["train"]], wild.fake.to_numpy()[w_train]]),
                              emb[X_LIKE][idx["val"]], y[idx["val"]])
    yw = wild.fake.to_numpy()[w_test]

    def rates(probe, p_real, p_fake):
        """AI caught and real photos flagged, at the threshold that flags 5% of real validation photos."""
        threshold = np.quantile(probe.predict_proba(emb[X_LIKE][real_val])[:, 1], 1 - config.FALSE_ALARM_RATE)
        return np.mean(p_fake >= threshold), np.mean(p_real >= threshold), len(p_fake), len(p_real)

    p_b = {c: probe_b.predict_proba(emb[c])[:, 1] for c in CONDITIONS}
    pw_b, pw_c = probe_b.predict_proba(wild_emb[w_test])[:, 1], probe_c.predict_proba(wild_emb[w_test])[:, 1]
    never = unseen[unseen.probe == "never seen"]
    heavy = f"jpeg{min(config.JPEG_QUALITIES)}"
    stress = pd.DataFrame([
        ("clean test images", *rates(probe_b, p_b["original"][real_test], p_b["original"][fake_test])),
        (f"as X serves them (JPEG {config.X_JPEG_QUALITY})", *rates(probe_b, p_b[X_LIKE][real_test], p_b[X_LIKE][fake_test])),
        (f"heavy JPEG ({min(config.JPEG_QUALITIES)})", *rates(probe_b, p_b[heavy][real_test], p_b[heavy][fake_test])),
        ("generator never seen", float(np.average(never.caught, weights=never.n_test)), np.nan, int(never.n_test.sum()), 0),
        ("in the wild (Reddit)", *rates(probe_b, pw_b[yw == 0], pw_b[yw == 1])),
        ("in the wild, after adding\nReddit photos to training", *rates(probe_c, pw_c[yw == 0], pw_c[yw == 1])),
    ], columns=["condition", "ai_caught", "real_flagged", "n_ai", "n_real"])
    stress["real_flagged_ci"] = [decision.wilson(round(r.real_flagged * r.n_real), r.n_real) if r.n_real else (np.nan, np.nan)
                                 for r in stress.itertuples()]
    print(f"\nIn the wild: {len(wild)} Reddit images ({wild.fake.mean():.0%} AI), {len(w_test)} held out for testing. "
          f"AUC probe B {roc_auc_score(yw, pw_b):.3f}, probe C {roc_auc_score(yw, pw_c):.3f} (C = {c_c})")
    print("Stress test (threshold flags 5% of real validation photos):")
    print(stress.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(stress))
    ax.bar(x - 0.2, stress.ai_caught, 0.4, color=BLUE, label="AI images caught")
    ax.bar(x + 0.2, stress.real_flagged.fillna(0), 0.4, color=RED, label="real photos wrongly flagged as AI")
    for i, r in stress.iterrows():
        ax.text(i - 0.2, r.ai_caught + 0.02, f"{r.ai_caught:.0%}", ha="center", fontsize=8)
        if not np.isnan(r.real_flagged):
            ax.text(i + 0.2, r.real_flagged + 0.02, f"{r.real_flagged:.0%}", ha="center", fontsize=8)
    ax.axvline(3.5, color="black", lw=0.8, ls=":")
    ax.set_xticks(x, stress.condition.str.replace(" (", "\n("), fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Image detector: perfect on the benchmark, not on real feeds", pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, frameon=False)
    save_fig(fig, "1_image_detector_stress_test")

    summary = {"C_probe_a": c_a, "C_probe_b": c_b, "C_probe_c": c_c,
               "brier_probe_b": brier_score_loss(y[idx["test"]], p_test),
               "auc_probe_b_x_like": roc_auc_score(y[idx["test"]], p_test),
               "auc_wild_probe_b": roc_auc_score(yw, pw_b), "auc_wild_probe_c": roc_auc_score(yw, pw_c)}
    wild_test = {"B": wild.iloc[w_test].assign(score=pw_b), "C": wild.iloc[w_test].assign(score=pw_c)}
    return probe_c, jpeg, unseen, stress, wild_test, summary


# --- 2. Feed -----------------------------------------------------------------

def build_feed(posts, images, emb, probe, test_pool=None):
    """Give each post an image (or none) from the held-out pool of its split.

    Train and validation posts draw from the vision validation images, test
    posts from the vision test images, so the image score is always out of sample.
    """
    rng = np.random.default_rng(config.SEED)
    images = images.assign(score=probe.predict_proba(emb[X_LIKE])[:, 1])  # out of sample for val/test images
    source = {"train": "val", "val": "val", "test": "test"}
    pools = {(s, fake): images[(images.split == source[s]) & (images.fake == fake)]
             for s in source for fake in (0, 1)}
    if test_pool is not None:  # stress test: test posts carry in-the-wild images instead
        pools.update({("test", fake): test_pool[test_pool.fake == fake] for fake in (0, 1)})
    has_image = rng.random(len(posts)) < config.SHARE_WITH_IMAGE
    is_ai = has_image & (rng.random(len(posts)) < config.SHARE_AI_OF_IMAGES)
    scores, files = np.zeros(len(posts)), np.full(len(posts), "", dtype=object)
    for i, split in enumerate(posts.split):
        if has_image[i]:
            pool = pools[(split, int(is_ai[i]))]
            j = rng.integers(len(pool))
            scores[i], files[i] = pool.score.iat[j], pool.file.iat[j]
    return posts.assign(has_image=has_image.astype(float), ai_image=is_ai, image_ai_score=scores, image_file=files,
                        brainrot=posts.clickbait | is_ai)


def text_scores(posts):
    """CLIP text embedding + logistic regression: how bait-like the wording is.

    Training posts get out-of-fold scores, so the fusion model learns how much
    to trust this score from scores it would actually see on new posts.
    """
    path = CACHE / "text_embeddings.npy"
    if path.exists() and len(np.load(path)) == len(posts):
        emb = np.load(path)
    else:
        print("  embedding post texts...", flush=True)
        emb = components.embed_texts(posts.text.tolist())
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
    tr, va = (posts.split == "train").to_numpy(), (posts.split == "val").to_numpy()
    y = posts.clickbait.to_numpy()
    probe, C = fit_logreg(emb[tr], y[tr], emb[va], y[va])
    scores = probe.predict_proba(emb)[:, 1]
    scores[tr] = cross_val_predict(probe, emb[tr], y[tr], cv=5, method="predict_proba")[:, 1]
    print(f"Text probe C = {C}, test AUC for clickbait = {roc_auc_score(y[~tr & ~va], scores[~tr & ~va]):.3f}")
    return probe, scores


# --- 3 and 4. Fusion and decision -------------------------------------------

def mute_words(text):
    text = text.lower()
    return any(word in text for word in config.MUTE_WORDS)


def product(feed, wild_feeds):
    parts = {s: feed[feed.split == s] for s in ["train", "val", "test"]}
    y = {s: parts[s].brainrot.to_numpy() for s in parts}
    variants = {
        "text only": ["text_bait_score"] + components.TEXT_FEATURES,
        "image only": ["image_ai_score", "has_image"],
        "text + image (fusion)": FUSION_FEATURES,
    }
    models, rows, test_p = {}, [], {}
    test = parts["test"]
    rows.append({"design": "show everything (today)", **decision.summarise(np.full(len(test), "show"), y["test"])})
    rows.append({"design": "X mute words (baseline)",
                 **decision.summarise(np.where(test.text.map(mute_words), "hide", "show"), y["test"])})
    for name, columns in variants.items():
        model, C = fit_logreg(parts["train"][columns], y["train"], parts["val"][columns], y["val"])
        t_low, t_high = decision.best_band(model.predict_proba(parts["val"][columns])[:, 1], y["val"])
        p = model.predict_proba(test[columns])[:, 1]
        test_p[name] = p
        models[name] = {"model": model, "C": C, "t_low": t_low, "t_high": t_high}
        rows.append({"design": name, "t_low": t_low, "t_high": t_high, "auc": roc_auc_score(y["test"], p),
                     "log_loss": log_loss(y["test"], p), **decision.summarise(decision.decide(p, t_low, t_high), y["test"])})
    # Stress test: the same test posts carrying in-the-wild Reddit images, scored by
    # the detector trained without Reddit (B) and with half of Reddit (C, the product's).
    fused = models["text + image (fusion)"]
    for name, wild_feed in wild_feeds.items():
        wild_test = wild_feed[wild_feed.split == "test"]
        p_wild = fused["model"].predict_proba(wild_test[FUSION_FEATURES])[:, 1]
        rows.append({"design": name, "t_low": fused["t_low"], "t_high": fused["t_high"],
                     "auc": roc_auc_score(wild_test.brainrot, p_wild), "log_loss": log_loss(wild_test.brainrot, p_wild),
                     **decision.summarise(decision.decide(p_wild, fused["t_low"], fused["t_high"]), wild_test.brainrot)})
    results = pd.DataFrame(rows).set_index("design")
    print("\nTest posts, each design with the band chosen on validation:")
    print(results.round(3).to_string())

    # Is the fusion's gain over mute words more than luck? Paired bootstrap of lost seconds.
    lost = lambda d: decision.seconds_per_post(decision.outcome_counts(d, y["test"]))  # noqa: E731
    d_fused = decision.decide(test_p["text + image (fusion)"], fused["t_low"], fused["t_high"])
    d_mute = np.where(test.text.map(mute_words), "hide", "show")
    rng = np.random.default_rng(config.SEED)
    gains = []
    for _ in range(1000):
        i = rng.integers(len(test), size=len(test))
        gains.append(decision.seconds_per_post(decision.outcome_counts(d_mute[i], y["test"][i]))
                     - decision.seconds_per_post(decision.outcome_counts(d_fused[i], y["test"][i])))
    ci = np.percentile(gains, [2.5, 97.5]) * config.POSTS_PER_DAY / 60
    print(f"Fusion vs mute words: {(lost(d_mute) - lost(d_fused)) * config.POSTS_PER_DAY / 60:.1f} min/day saved "
          f"(95% bootstrap CI {ci[0]:.1f} to {ci[1]:.1f})")

    # Figure 3: the headline. Minutes a day lost to brainrot and to wrong hides.
    order = ["show everything (today)", "X mute words (baseline)", "image only", "text only", "text + image (fusion)",
             *wild_feeds]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    minutes = results.loc[order, "minutes_lost_per_day"]
    bars = ax.barh(order, minutes, color=[GREY, GREY, BLUE, BLUE, RED, "#e8a598", "#e8a598"])
    for bar, design in zip(bars, order):
        r = results.loc[design]
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{r.minutes_lost_per_day:.1f} min  (sees {r.missed_rate:.0%} of brainrot, hides {r.false_hide_rate:.1%} of good posts)",
                va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, minutes.max() * 1.9)
    ax.set_xlabel(f"minutes lost per day ({config.POSTS_PER_DAY} posts, test set)")
    ax.set_title("Which design wastes the least of the user's time?")
    save_fig(fig, "2_designs_compared")

    # Figure 4: why a blur band. Lost minutes if the fusion only showed or hid.
    p, grid = test_p["text + image (fusion)"], decision.GRID[1:]
    single = [decision.summarise(decision.decide(p, t, t), y["test"])["minutes_lost_per_day"] for t in grid]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(grid, single, color=BLUE, lw=2, label="show or hide only, at threshold t")
    ax.axvspan(fused["t_low"], fused["t_high"], color=RED, alpha=0.12,
               label=f"chosen blur band {fused['t_low']:.2f}-{fused['t_high']:.2f}")
    ax.axhline(results.loc["text + image (fusion)", "minutes_lost_per_day"], color=RED, lw=2, ls="--",
               label="show / blur / hide with that band")
    ax.axhline(results.loc["show everything (today)", "minutes_lost_per_day"], color=GREY, lw=1.5, ls=":",
               label="show everything (today)")
    ax.set_ylim(0, 3 * results.loc["show everything (today)", "minutes_lost_per_day"])
    ax.axvline(config.SINGLE_THRESHOLD, color="black", lw=1, ls=":")
    ax.text(config.SINGLE_THRESHOLD + 0.01, ax.get_ylim()[1] * 0.92, f"cost rule\nt = {config.SINGLE_THRESHOLD:.2f}", fontsize=8)
    ax.set_xlabel("P(brainrot) threshold t")
    ax.set_ylabel("minutes lost per day")
    ax.set_title("Hiding only when sure, blurring when unsure")
    ax.legend(fontsize=8, loc="upper center")
    save_fig(fig, "3_threshold_and_band")

    coefs = pd.Series(fused["model"][-1].coef_.ravel(), index=FUSION_FEATURES).sort_values()
    print("\nFusion coefficients (per 1 std of each feature):")
    print(coefs.round(3).to_string())
    return models, results, coefs


def main():
    config.ARTIFACTS.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("=== 1. Vision")
    images = pd.read_csv(config.IMAGES_CSV)
    images["fake"] = (images.label == "fake").astype(int)
    images = three_way_split(images, "generator").reset_index(drop=True)
    print(images.groupby(["generator", "split"]).size().unstack().to_string())
    emb = image_embeddings(images)
    probe, jpeg, unseen, stress, wild_test, vision_summary = vision(images, emb)

    print("\n=== 2. Feed")
    posts = pd.read_csv(config.POSTS_CSV)
    posts["text"] = posts.text.fillna("").astype(str)
    posts["clickbait"] = posts.clickbait_mean >= config.CLICKBAIT_IF_MEAN_AT_LEAST
    posts = three_way_split(posts, "clickbait").reset_index(drop=True)
    text_probe, posts["text_bait_score"] = text_scores(posts)
    posts = posts.join(pd.DataFrame([components.text_features(t) for t in posts.text]))
    feed = build_feed(posts, images, emb, probe)
    print(f"{len(feed)} posts: {feed.clickbait.mean():.0%} clickbait text, {feed.has_image.mean():.0%} with an image, "
          f"{feed.ai_image.mean():.0%} with an AI image, {feed.brainrot.mean():.0%} brainrot")

    wild_feeds = {
        "stress test: Reddit images,\ndetector trained without them": build_feed(posts, images, emb, probe, test_pool=wild_test["B"]),
        "stress test: Reddit images,\ndetector retrained with some": build_feed(posts, images, emb, probe, test_pool=wild_test["C"]),
    }

    print("\n=== 3-4. Fusion and decision")
    models, results, coefs = product(feed, wild_feeds)

    fused = models["text + image (fusion)"]
    joblib.dump(probe, MODELS / "vision_probe.joblib")
    joblib.dump(text_probe, MODELS / "text_probe.joblib")
    joblib.dump(fused["model"], MODELS / "fusion.joblib")
    (MODELS / "config.json").write_text(json.dumps({
        "clip_model": config.CLIP_MODEL, "x_jpeg_quality": config.X_JPEG_QUALITY,
        "fusion_features": FUSION_FEATURES, "t_low": fused["t_low"], "t_high": fused["t_high"],
        "C_fusion": fused["C"], **vision_summary,
    }, indent=2))
    jpeg.to_csv(config.ARTIFACTS / "vision_jpeg.csv", index=False)
    unseen.to_csv(config.ARTIFACTS / "vision_unseen_generators.csv", index=False)
    stress.to_csv(config.ARTIFACTS / "vision_stress_test.csv", index=False)
    results.to_csv(config.ARTIFACTS / "product_test_results.csv")
    coefs.to_csv(config.ARTIFACTS / "fusion_coefficients.csv", header=["coefficient"])
    feed.drop(columns=["text"]).to_csv(config.ARTIFACTS / "feed.csv", index=False)
    print(f"\nSaved models to {MODELS} and figures to {config.FIG_DIR}")


if __name__ == "__main__":
    main()
