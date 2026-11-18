"""
The app: an App tab and a Monitoring tab. `uv run python app.py`

Keep the logic in pipeline.py (or your own modules); this file only wires it to
the interface and the logs. Keep `serve()` and the `/run` endpoint: check.py
calls `serve()` locally and `/run` on your Space, with one input from
eval/cases.csv (text, or a file path).
"""

import time

import gradio as gr

import env
import logbook
import monitor
import pipeline

env.load_env()
LOG_TARGET = logbook.start()

TITLE = "Slop Mop"
DESCRIPTION = "For anyone who scrolls X: scores each post (AI-generated image + bait-like text) and shows, blurs or hides it."


def serve(raw_input: str) -> dict:
    """Run one request through the pipeline, log it, and return the logged record."""
    started = time.perf_counter()
    result = pipeline.run(raw_input)
    return logbook.log_request(result | {"latency_s": round(time.perf_counter() - started, 3)})


# --- App tab -----------------------------------------------------------------

def on_run(raw_input):
    if raw_input is None or not str(raw_input).strip():
        raise gr.Error("Give me an input first.")
    record = serve(raw_input)
    outputs = record["outputs"]
    fused = outputs["fusion"]
    icon = {"show": "✅ **Show**", "blur": "🌫️ **Blur** (unsure: the user decides)", "hide": "🚫 **Hide**"}[fused["decision"]]
    reasons = "; ".join(fused["reasons"]) or "nothing stands out"
    verdict = f"{icon}: P(brainrot) = {fused['p_brainrot']:.2f}. Why: {reasons}."
    return (
        outputs["vision"],
        verdict,
        f"Latency {record['latency_s']} s, cost €{record['cost_eur']:.4f}.",
        record["id"],
    )


def on_post(text, image_path):
    """The App tab's form: text and an optional image, joined into the pipeline's one-string input."""
    if not (text or "").strip() and not image_path:
        raise gr.Error("Paste a post's text, upload its image, or both.")
    return on_run(f"{text or ''}\nimage: {image_path}" if image_path else text)


def on_feedback(request_id, value):
    if not request_id:
        gr.Warning("Run a request first.")
        return
    logbook.log_feedback(request_id, value)
    gr.Info("Thanks for the feedback.")


# --- Monitoring tab ----------------------------------------------------------

def on_refresh():
    logs = logbook.load()
    reference = monitor.load_reference()
    fired = monitor.alerts(logs, reference) if not logs.empty else []
    if reference is None:
        alert_text = "No reference set yet: run `uv run python main.py reference` and commit `monitoring/`."
    elif fired:
        alert_text = "### 🚨 Alerts\n" + "\n".join(f"- {message}" for message in fired)
    else:
        alert_text = "### ✅ No alerts"
    if reference is not None and len(logs) >= monitor.MIN_WINDOW:
        drift_table = monitor.drift(reference, monitor.live_features(logs))
    else:
        drift_table = monitor.drift(reference, reference.head(0)) if reference is not None else None
    return alert_text, monitor.summary(logs), monitor.quality(logs), drift_table


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")
    if missing := env.missing():
        gr.Markdown(f"⚠️ **Missing environment variables:** {', '.join(missing)}. See SETUP.md.")

    with gr.Tab("App"):
        request_id = gr.State("")
        with gr.Row():
            post_text = gr.Textbox(label="Post text", lines=4)
            post_image = gr.Image(label="Post image (optional)", type="filepath", height=220)
        run_button = gr.Button("Score this post", variant="primary")
        with gr.Row():
            out_a = gr.JSON(label="1. Vision: is the image AI-generated?")
            out_b = gr.Markdown(label="2. Fusion: show, blur or hide")
        # check.py and the Space API call /run with one string (see pipeline.read_input).
        raw_input = gr.Textbox(visible=False)
        api_button = gr.Button(visible=False)
        status = gr.Markdown()
        with gr.Row():
            up = gr.Button("👍")
            down = gr.Button("👎")
        run_button.click(on_post, [post_text, post_image], [out_a, out_b, status, request_id], api_name=False)
        api_button.click(on_run, raw_input, [out_a, out_b, status, request_id], api_name="run")
        up.click(lambda rid: on_feedback(rid, +1), request_id, None, api_name=False)
        down.click(lambda rid: on_feedback(rid, -1), request_id, None, api_name=False)

    with gr.Tab("Monitoring"):
        refresh = gr.Button("Refresh")
        alert_box = gr.Markdown()
        summary_table = gr.Dataframe(label="Summary", interactive=False)
        quality_plot = gr.LinePlot(x="time", y="value", color="signal", label="Quality over time", height=300)
        drift_table = gr.Dataframe(label=f"Drift: PSI of the last {monitor.WINDOW} requests against the reference set", interactive=False)
        gr.Markdown(f"<small>{LOG_TARGET}</small>")
        outputs = [alert_box, summary_table, quality_plot, drift_table]
        refresh.click(on_refresh, None, outputs, api_name=False)
        demo.load(on_refresh, None, outputs, api_name=False)


if __name__ == "__main__":
    demo.launch()
