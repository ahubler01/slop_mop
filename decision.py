"""
Turns P(brainrot) into show / blur / hide, and prices each decision in seconds
of the user's time.

Shared by train.py (to choose the band on validation posts) and pipeline.py
(to apply it live), so both always agree.

    p <  t_low            -> show
    t_low <= p < t_high   -> blur, with the reason, one click to reveal
    p >= t_high           -> hide
"""

import math

import numpy as np
import pandas as pd

import config

DECISIONS = ["show", "blur", "hide"]
GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def decide(p, t_low, t_high):
    p = np.asarray(p, float)
    return np.where(p >= t_high, "hide", np.where(p < t_low, "show", "blur"))


def outcome_counts(decisions, brainrot):
    """2x3 table: actual (brainrot / good) by decision (show / blur / hide)."""
    decisions, brainrot = np.asarray(decisions), np.asarray(brainrot, bool)
    return pd.DataFrame(
        {d: [int(((decisions == d) & brainrot).sum()), int(((decisions == d) & ~brainrot).sum())] for d in DECISIONS},
        index=["brainrot", "good"],
    )


def seconds_per_post(counts):
    """Expected seconds of the user's time lost per post scrolled past."""
    cost = (counts.at["brainrot", "show"] * config.COST_MISSED
            + counts.at["good", "hide"] * config.COST_FALSE_HIDE
            + (counts.at["brainrot", "blur"] + counts.at["good", "blur"]) * config.COST_BLUR
            + counts.at["brainrot", "hide"] * config.COST_TP
            + counts.at["good", "show"] * config.COST_TN)
    return cost / max(counts.values.sum(), 1)


def summarise(decisions, brainrot):
    counts = outcome_counts(decisions, brainrot)
    n_bad, n_good = max(counts.loc["brainrot"].sum(), 1), max(counts.loc["good"].sum(), 1)
    seconds = seconds_per_post(counts)
    return {
        "seconds_per_post": seconds,
        "missed_rate": counts.at["brainrot", "show"] / n_bad,       # brainrot the user still sees
        "false_hide_rate": counts.at["good", "hide"] / n_good,      # good posts hidden
        "blur_share": counts["blur"].sum() / counts.values.sum(),
        "minutes_lost_per_day": seconds * config.POSTS_PER_DAY / 60,
    }


def best_band(p, brainrot):
    """Cheapest (t_low, t_high) over the grid. Ties go to the narrower band."""
    best = None
    for t_low in GRID:
        for t_high in GRID[GRID >= t_low]:
            cost = seconds_per_post(outcome_counts(decide(p, t_low, t_high), brainrot))
            key = (round(cost, 9), t_high - t_low)
            if best is None or key < best[0]:
                best = (key, float(t_low), float(t_high))
    return best[1], best[2]


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)
