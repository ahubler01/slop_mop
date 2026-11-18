# Slop Mop: model results

Written by `uv run python train.py` (after `uv run python data/fetch_data.py` once). Every number below is in a CSV in this folder.

## The chain

```
image ──► 1. VISION: frozen CLIP ViT-B/32 + logistic-regression probe ──► P(AI image) ─┐
text ───► CLIP text score + 8 simple features (caps, !?, "you", "this/why", ...) ──────┴─► 2. TABULAR fusion (logistic regression) ──► P(brainrot) ──► show / blur / hide
```

The fusion model never sees pixels, only the probe's probability. That is why the probe has to be calibrated (Brier 0.007 on the test images).

## Data

| Data | Source | Licence | Used |
|---|---|---|---|
| AI and real images | OpenFake `core` test shard 0 | CC-BY-SA-4.0 | 1,696 images (900 real, 796 from 5 generators), face photos dropped |
| Images from the wild | OpenFake `reddit` test shard 0 | CC-BY-SA-4.0 | 977 images (812 AI, 165 real), face photos dropped; half for training, half for testing only |
| Post texts | Webis-Clickbait-17 (Zenodo 5530410) | CC-BY-4.0 | 19,484 tweets, 24% clickbait (5 annotators, mean ≥ 0.5) |

**Simulated feed:** no public dataset has X posts with both a labelled image and a labelled text. Each tweet therefore gets an image from the held-out pool: 60% of posts carry an image, and 30% of those images are AI-generated (`config.py`). A post counts as brainrot if its text is clickbait OR its image is AI. That is a product definition, not a finding.

## 1. The image detector: [figure 1](figures/1_image_detector_stress_test.png)

The threshold is set so that it flags 5% of the real validation photos.

| Condition | AI caught | Real photos wrongly flagged |
|---|---|---|
| Clean test images | 100% | 7% |
| As X serves them (JPEG 85) | 100% | 5% |
| Heavy JPEG (30) | 100% | 3% |
| Generator never seen in training (leave one out, 5 generators) | 99% | – |
| **In the wild (Reddit)** | 93% | **75%** (95% CI 64–83%) |
| In the wild, after adding half the Reddit images to training | 97% | 41% (31–52%) |

- **JPEG is not the problem.** CLIP resizes every image to 224 px, which throws away compression artefacts anyway. The same thing makes the file format a non-issue: all real images are JPEG while 39% of the AI ones are PNG, and the probe does not use that.
- **The benchmark is the problem.** On a clean split the probe separates "ImageNet/DOCCI photo" from "generated look". It does not learn "was this generated". On real Reddit photos, 3 in 4 are flagged. AUC drops from 1.00 to 0.73.
- **Some in-domain data fixes most of it.** Adding ~490 Reddit images to training raises AUC to 0.94. That detector (C) is the one the product ships.

## 2. The product: [figure 2](figures/2_designs_compared.png)

All costs are in seconds of the user's time:

- a brainrot post shown costs 6 s;
- a good post hidden costs 20 s, because it erodes trust;
- a blurred post costs 1.5 s.

Test set: 3,897 posts, 300 posts a day.

| Design | Minutes lost per day | Brainrot still seen | Good posts hidden |
|---|---|---|---|
| Show everything (today) | 11.6 | 100% | 0% |
| X mute words (baseline) | 11.8 | 98% | 0.7% |
| Image only | 6.6 | 49% | 0.5% |
| Text only | 6.8 | 27% | 0.8% |
| **Text + image (fusion)** | **3.5** | **8%** | **0.9%** |
| Stress: Reddit images, detector B | 17.8 | 7% | 24.7% |
| Stress: Reddit images, detector C | 8.5 | 5% | 8.8% |

- Fusion saves 8.2 minutes a day compared with mute words (95% bootstrap CI 7.7–8.8).
- Mute words are worse than doing nothing: they catch 2% of brainrot and still hide some good posts.
- With the detector trained without Reddit images, the product is worse than no filter on real-world images. Detector C keeps it better than today, but it still hides 9% of good posts.

## 3. Why a blur band: [figure 3](figures/3_threshold_and_band.png)

- If the model only shows or hides, the cheapest threshold on the test set lands at t ≈ 0.77. That is exactly what the cost rule predicts: 20 / (20 + 6) = 0.77. This is a sign the probabilities are calibrated.
- Adding a blur band, chosen on validation posts, cuts the lost time further, from ~5.2 to 3.5 minutes a day:
  - show below 0.17;
  - blur between 0.17 and 0.91;
  - hide above 0.91.
- The cost of the band: 28% of posts get blurred. That share is set entirely by `COST_BLUR`.

## Known limits

- **Text domain.** Webis tweets are news headlines. Personal captions ("Sunset over the lake this evening") get a high bait score, so the drift check should watch `text_bait_score`.
- **Simulated feed.** The 60% / 30% shares and the OR label are assumptions. Fusion learns roughly "max of the two scores", and its gain over each single component comes partly from that construction.
- **Few real photos in the wild.** Only 165 Reddit photos are real (83 in test), hence the wide intervals.
- **Time costs are guesses.** 6 s, 20 s and 1.5 s should be checked with users. Changing them in `config.py` and rerunning re-chooses every threshold.
