"""Decision rules and evaluation lookup shared by the image and video endpoints.

Keeping this here (rather than duplicated) does not couple the two endpoints: each still
validates, decodes and runs inference independently, and a failure in one has no effect on
the other. This module only holds the threshold policy both apply to a probability.

Threshold policy
-----------------
Thresholds are NOT hardcoded at 0.5. They are loaded from `thresholds.json` (see that file's
"rationale" field), which `scripts/evaluate.py` is meant to overwrite once a labelled
validation set has been scored (see that script's threshold-selection step). Until that has
been run, the checked-in file holds explicit, documented placeholder values with a declared
INCONCLUSIVE band — not a bare 0.5 cut-off presented as if it were known to be optimal.

`FAKE_THRESHOLD` / `REAL_THRESHOLD` env vars, if set, override the file for quick ops tuning
without editing the config; they take precedence over whatever `thresholds.json` holds.
"""
import json
import logging
import os

log = logging.getLogger("thresholds")

THRESHOLDS_PATH = os.getenv("THRESHOLDS_PATH", os.path.join(os.path.dirname(__file__), "thresholds.json"))
EVAL_RESULTS_PATH = os.getenv("EVAL_RESULTS_PATH", "/srv/eval_results.json")


def _load_thresholds() -> dict:
    defaults = {
        "fake_threshold": 0.6,
        "real_threshold": 0.4,
        "selection_method": "initial_manual",
        "rationale": "thresholds.json was not found or was invalid; using hardcoded fallback defaults.",
        "selected_on": None,
        "source_eval": None,
    }
    try:
        with open(THRESHOLDS_PATH) as f:
            cfg = json.load(f)
        defaults.update({k: cfg[k] for k in defaults if k in cfg})
    except (OSError, ValueError) as e:
        log.warning("Could not load %s (%s); using fallback thresholds.", THRESHOLDS_PATH, e)
    return defaults


_CONFIG = _load_thresholds()

# fake_probability >= FAKE_THRESHOLD -> FAKE; <= REAL_THRESHOLD -> REAL; otherwise INCONCLUSIVE.
FAKE_THRESHOLD = float(os.getenv("FAKE_THRESHOLD", _CONFIG["fake_threshold"]))
REAL_THRESHOLD = float(os.getenv("REAL_THRESHOLD", _CONFIG["real_threshold"]))
THRESHOLD_SELECTION_METHOD = _CONFIG["selection_method"]
THRESHOLD_RATIONALE = _CONFIG["rationale"]
THRESHOLD_SELECTED_ON = _CONFIG["selected_on"]

if FAKE_THRESHOLD <= REAL_THRESHOLD:
    raise ValueError(
        f"FAKE_THRESHOLD ({FAKE_THRESHOLD}) must be greater than REAL_THRESHOLD ({REAL_THRESHOLD}); "
        "check thresholds.json / env overrides."
    )


def decide(fake_probability: float) -> tuple[str, float | None]:
    if fake_probability >= FAKE_THRESHOLD:
        return "FAKE", fake_probability
    if fake_probability <= REAL_THRESHOLD:
        return "REAL", 1 - fake_probability
    return "INCONCLUSIVE", None


def threshold_info() -> dict:
    """Decision policy in effect, for transparency in API responses — never implies the
    probabilities themselves are calibrated, only states what cut-offs are applied and why."""
    return {
        "fake_at_or_above": FAKE_THRESHOLD,
        "real_at_or_below": REAL_THRESHOLD,
        "selection_method": THRESHOLD_SELECTION_METHOD,
        "selected_on": THRESHOLD_SELECTED_ON,
        "rationale": THRESHOLD_RATIONALE,
    }


def evaluation_accuracy() -> dict | None:
    """Measured metrics from a labelled evaluation run (see scripts/evaluate.py), or None.
    Never fabricated: if no eval has been recorded, this returns None and callers must not
    invent a number. Required keys are accuracy/dataset/samples; everything else recorded by
    the evaluation script (precision, recall, f1, confusion_matrix, threshold, speed,
    calibration note, limitations) is passed through as-is when present."""
    try:
        with open(EVAL_RESULTS_PATH) as f:
            d = json.load(f)
        result = {"accuracy": float(d["accuracy"]), "dataset": str(d["dataset"]), "samples": int(d["samples"])}
        for key in (
            "precision",
            "recall",
            "f1",
            "confusion_matrix",
            "threshold_used",
            "inference_speed_ms_per_sample",
            "split",
            "evaluated_at",
            "calibration_note",
            "limitations",
        ):
            if key in d:
                result[key] = d[key]
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None
