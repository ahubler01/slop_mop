# Project summary: AI feed filter for X ("brainrot remover")

A browser extension that scores every post on your X feed in real time and hides it *before you see it* when it crosses a threshold. It combines an **AI-generated-image detector (vision)** with a **calibrated fusion model (tabular)** and learns your personal "not for me" taste from your hide/unhide clicks. An optional agentic LLM fact-checks posts on demand.

---

## 1. Problem and positioning

- **User:** anyone who scrolls X.
- **Decision supported:** show or hide each post.
- **Baseline to beat:** X's mute words and "Not interested" button.
- **Success metric:** share of unwanted posts the user still sees, plus the false-hide rate (how often the user clicks "unhide").
- **Value in €:** minutes per day saved × a value per hour.
- **Existing tools** (Bouncer by Imbue, GoodVibes, ToneGuard, Chirply, DetoxAI) all filter **text only**, by topic or tone.
- **Our angle:** three signals nobody combines:
  1. **Vision:** is the image AI-generated?
  2. **Content quality:** rage bait, engagement bait or clickbait, and whether the post is likely to get a Community Note.
  3. **Personal:** a "not for me" profile learned from your own clicks.

---

## 2. Data (recent, 2024–2026, small subsets are enough)

| Need | Dataset | How recent | Contents | Size to use |
|---|---|---|---|---|
| AI images (main) | **OpenFake** (`ComplexDataLab/OpenFake`, HF) | Sept 2025 | Flux 1.x, Midjourney v6/v7, DALL·E 3, Imagen 3/4, GPT Image 1, Ideogram 3, Grok-2, plus a real-world test set from the 2025 Canadian election | Stream 5–10k images (full set ≈ 1 TB). CC-BY-SA; proprietary-generator subsets are non-commercial |
| AI images (backup) | **COCO_AI / VCT²** (`NasrinImp/COCO_AI`, HF) | 2024–25 | The same caption rendered by SD 2.1, SDXL, SD 3, SD 3.5, DALL·E 3 and Midjourney 6, next to the real COCO photo | 10k captions × 7 images (≈ 6 GB) |
| Misleading posts | **COMMUNITYNOTES** (Xing et al.) | Posts from 2021–24 | 104k posts with notes and helpfulness labels, **including the post text** | 5–10k posts |
| Fact-check evaluation | **CommunityFact** (HF) | May 2026 | Claim, timestamp, language, domain, label and evidence URLs | A few hundred claims |
| Raw recent posts | **BluePrint** (Complex Data Lab) | Recent | Anonymized, PII-removed Bluesky posts | 3–5k posts to label |
| Clickbait (optional) | Webis-Clickbait-17 | 2017 (old) | 38,517 tweets with clickbait-strength scores | Optional extra |

**Gaps and fixes:**
- **There's no public English "brainrot" or rage-bait dataset.** Have an LLM label BluePrint posts against your rubric (the method used in a 2026 study of rage bait on Japanese X), and check those labels against about 200 of your own.
- **Faces:** OpenFake is political, so many images show real people. Run a face detector and drop any image with a face, from training, evaluation and demos alike.

---

## 3. Where the technical value comes from

The value is that you **train small, fast models and measure them against the zero-shot LLM**.

**A. Vision: frozen foundation model + trained linear probe**
- A frozen CLIP or DINOv3 encoder with a linear classifier on top. This is the state-of-the-art approach (UnivFD, and a 2025 DINOv3 paper that used only about 1k fake images).
- Experiments:
  - cross-generator matrix (train on Flux → test on GPT Image 1 and on Midjourney);
  - JPEG recompression robustness (X re-encodes images);
  - calibration, so that "0.8" really means 80%.

**B. Text: teacher → student distillation**
- The LLM teacher labels 3–5k posts: rage bait, engagement bait, low-effort, informative.
- Check the teacher against about 200 human labels (agreement score).
- The student is sentence embeddings plus logistic regression or LightGBM.
- **The key table:** student vs. zero-shot LLM on accuracy, latency (≈ 30 ms vs ≈ 1.5 s) and € per 1,000 posts.

**C. Tabular fusion: the required chain**
- LightGBM combines the image score, the text scores, simple features (share of capital letters, emoji, hashtags, links, length) and similarity to posts you've hidden.
- It outputs a **calibrated P(hide) per category**, followed by a threshold.

**D. Personal layer + RL bonus**
- Your hidden posts form a "not for me" profile in embedding space.
- A bandit learns each category's threshold from hide/unhide clicks.

**Optional: agentic LLM (slow path only)**
- Runs on posts in the uncertain band, or when the user clicks "check this".
- Checks claims with web search; evaluated on CommunityFact. Its free-text explanation is what the LLM judge grades.

```
X feed (extension reads each post as it appears)
  ├─ image ──► VISION: AI-image score
  └─ text ───► text scores + simple features + similarity to your hidden posts
                    ▼
        TABULAR fusion ──► calibrated P(hide) per category ──► threshold ──► blur + reason
                    │
          uncertain band ──► "Check this" ──► agentic LLM fact-check (optional)
```

---

## 4. Requirement by requirement

| Requirement | How this project meets it | Risk |
|---|---|---|
| Business process, decision, baseline, € | Show or hide each post; baseline is X's mute words; € = minutes saved | Low |
| Two modules, chained | Vision → tabular fusion → threshold (agent optional) | ✅ |
| Gradio Space with App + Monitoring tabs | App tab: paste a post or upload an image → verdict + reasons. The extension is a second client of the same API | Medium |
| Log every request, no personal data | Log a hashed post ID, `has_image`, language, length, each component's scores, model versions, latency, cost, hide/unhide. **Never text, images or handles** | Medium (needs discipline) |
| Monitoring | Quality: unhide rate, % hidden, confidence over time. Drift: embedding-distribution shift. Alert: a threshold on that shift | ✅ |
| `drift-demo` | Replay images from a generator you held out (e.g. Imagen 4), recompressed images, Spanish posts | ✅ ready-made |
| Evaluation | Per component: AUC per generator; student vs teacher vs humans. Product: 30+ cases vs mute-word baseline. LLM judge: 20+ labels on explanations. Cost per request | ✅ |
| RL bonus | Threshold bandit on real hide/unhide clicks. Reward-hacking guard: keep the % of posts hidden inside a band | ✅ natural fit |

---

## 5. Extension feasibility (under 1 second, hidden before the user sees it)

**Key trick:** X builds posts in the page before they scroll into view. Hide each post the moment it's inserted ("hidden until scored") and reveal it if it passes. The user never sees an unscored post.

| Step | In the browser (WebGPU) | Calling the HF Space (free CPU) |
|---|---|---|
| Detect the new post in the page | ~0 ms | ~0 ms |
| Text embedding + student | ~20–50 ms | ~100–300 ms + network |
| Image probe | ~50–120 ms (CLIP runs at 20+ fps on WebGPU) | ~200–500 ms |
| Fusion + threshold | <1 ms | <1 ms |
| **Total** | **~100–200 ms** | **~0.4–1 s** (batched, helped by the head start) |

**Practical notes:**
- Expect a one-time warm-up of about 300 ms.
- In a Chrome extension, WebGPU isn't available in the background service worker, so the model runs in an **offscreen document** (needs the `offscreen` permission).
- **Plan:** v1 calls the Space API (simplest, clearly "the product on the Space"). v2 adds local ONNX inference, with the Space still hosting the models, the App tab and the logs.
- **Main risk:** X changes its page structure without notice, which breaks the extension until you update it. Plan for that.
- **Effort:** the extension itself (detect posts, hide them, call the API, hide/unhide buttons) is a few hundred lines, about 1 week for one person. Models, evaluation and monitoring are the bigger share of the work.

---

## 6. Risks to discuss in REPORT.md

- **"Dumb" must not mean "opinions I disagree with."** Keep politics out of the automatic labels; viewpoint filtering is opt-in only. Discuss the filter-bubble effect.
- **Wellbeing evidence is mixed:** a 2026 Warwick field experiment suggests filtering toxicity may have unintended psychological costs.
- **Platform rules:** change only what *you* see in your own browser (like an ad blocker). Never scrape X from the server.
- **Feedback loops:** only engaged users give feedback, and thresholds can drift toward hiding too much.

---

## 7. Week-1 checklist

1. Stream a 5k-image OpenFake subset and remove the face images.
2. Get the COMMUNITYNOTES post texts from the authors' repository (fallback: COCO_AI + BluePrint).
3. Write the labeling rubric and label 200 posts by hand.
4. Build the Space skeleton: App tab, logger, Monitoring tab.
5. Build the extension prototype: hide each post until scored, call the Space API.

---

## Sources

- OpenFake: https://huggingface.co/datasets/ComplexDataLab/OpenFake · https://arxiv.org/abs/2509.09495
- COCO_AI / VCT²: https://huggingface.co/datasets/NasrinImp/COCO_AI
- COMMUNITYNOTES dataset: https://arxiv.org/abs/2510.24810
- CommunityFact: https://huggingface.co/papers/2605.30241
- BluePrint (Complex Data Lab): https://huggingface.co/ComplexDataLab
- Ragebait on Japanese X (LLM pseudo-labels): https://arxiv.org/pdf/2609.02262
- Webis-Clickbait-17: https://webis.de/data/webis-clickbait-17
- UnivFD / linear probes for AI-image detection: https://www.alphaxiv.org/abs/2608.04935 · https://arxiv.org/html/2511.22471v1
- Browser WebGPU benchmarks: https://calmops.com/ai/browser-ai-webgpu-complete-guide/
- Transformers.js + WebGPU in a Chrome extension: https://medium.com/@GenerationAI/transformers-js-onnx-runtime-webgpu-in-chrome-extension-13b563933ca9
- Existing filters: Bouncer https://feedbagel.com/post/bouncer-ai-powered-browser-extension-filters-unwanted-content-from-twitterx-feed · GoodVibes https://futuretools.io/tools/goodvibes-75564b65
- Toxicity filtering field experiment (Warwick): https://warwick.ac.uk/fac/soc/economics/research/centres/cage/manage/publications/wp817.226.pdf
