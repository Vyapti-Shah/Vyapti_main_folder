"""Threshold boundaries, confidence semantics, and evaluation-metrics lookup."""
import json

import pytest

from app import common


# ---- threshold boundaries ----

@pytest.mark.parametrize(
    "p,expected_label",
    [
        (0.0, "REAL"),
        (0.39999, "REAL"),
        (0.4, "REAL"),  # inclusive boundary: <= REAL_THRESHOLD -> REAL
        (0.40001, "INCONCLUSIVE"),
        (0.5, "INCONCLUSIVE"),
        (0.59999, "INCONCLUSIVE"),
        (0.6, "FAKE"),  # inclusive boundary: >= FAKE_THRESHOLD -> FAKE
        (0.60001, "FAKE"),
        (1.0, "FAKE"),
    ],
)
def test_decide_boundaries(p, expected_label):
    prediction, _ = common.decide(p)
    assert prediction == expected_label


def test_decide_respects_configured_thresholds(monkeypatch):
    monkeypatch.setattr(common, "FAKE_THRESHOLD", 0.8)
    monkeypatch.setattr(common, "REAL_THRESHOLD", 0.2)
    assert common.decide(0.75)[0] == "INCONCLUSIVE"
    assert common.decide(0.8)[0] == "FAKE"
    assert common.decide(0.2)[0] == "REAL"
    assert common.decide(0.21)[0] == "INCONCLUSIVE"


def test_thresholds_file_is_internally_consistent():
    # the module-level guard in common.py already enforces this at import time; re-assert
    # it here so a bad thresholds.json edit fails a fast, obvious test instead of just
    # crashing app startup.
    assert common.FAKE_THRESHOLD > common.REAL_THRESHOLD


def test_thresholds_not_naive_0_5_without_evidence():
    # Guards against silently "simplifying" back to an unconfigured 0.5 cut-off: a single
    # 0.5 threshold with no INCONCLUSIVE band is exactly what the initial-rules rationale
    # argues against.
    assert common.REAL_THRESHOLD < 0.5 < common.FAKE_THRESHOLD


# ---- confidence semantics ----
# confidence = the model's probability of the PREDICTED class, not always "fake probability".

def test_confidence_is_probability_of_predicted_class_for_fake():
    prediction, confidence = common.decide(0.9)
    assert prediction == "FAKE"
    assert confidence == pytest.approx(0.9)  # P(fake) when predicting FAKE


def test_confidence_is_probability_of_predicted_class_for_real():
    prediction, confidence = common.decide(0.1)
    assert prediction == "REAL"
    assert confidence == pytest.approx(0.9)  # P(real) = 1 - P(fake), when predicting REAL


def test_confidence_is_none_when_inconclusive():
    prediction, confidence = common.decide(0.5)
    assert prediction == "INCONCLUSIVE"
    assert confidence is None


def test_confidence_always_at_or_above_half_of_decided_cases():
    # Whenever a definite call is made, confidence reflects the winning side and so is
    # never below 0.5 (it would make no sense to call FAKE with <50% P(fake)).
    for p in (0.6, 0.75, 1.0, 0.4, 0.25, 0.0):
        prediction, confidence = common.decide(p)
        if prediction != "INCONCLUSIVE":
            assert confidence >= 0.5


# ---- evaluation lookup: never fabricated ----

def test_evaluation_accuracy_none_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "EVAL_RESULTS_PATH", str(tmp_path / "does-not-exist.json"))
    assert common.evaluation_accuracy() is None


def test_evaluation_accuracy_none_when_file_is_malformed_json(monkeypatch, tmp_path):
    f = tmp_path / "e.json"
    f.write_text("{not valid json")
    monkeypatch.setattr(common, "EVAL_RESULTS_PATH", str(f))
    assert common.evaluation_accuracy() is None


@pytest.mark.parametrize("missing_key", ["accuracy", "dataset", "samples"])
def test_evaluation_accuracy_none_when_required_key_missing(monkeypatch, tmp_path, missing_key):
    d = {"accuracy": 0.9, "dataset": "x", "samples": 10}
    del d[missing_key]
    f = tmp_path / "e.json"
    f.write_text(json.dumps(d))
    monkeypatch.setattr(common, "EVAL_RESULTS_PATH", str(f))
    assert common.evaluation_accuracy() is None


def test_evaluation_accuracy_passes_through_optional_fields(monkeypatch, tmp_path):
    d = {
        "accuracy": 0.9,
        "dataset": "x",
        "samples": 10,
        "precision": 0.88,
        "recall": 0.92,
        "f1": 0.9,
        "confusion_matrix": {"real": {"REAL": 4, "FAKE": 1, "INCONCLUSIVE": 0}},
        "threshold_used": {"fake_at_or_above": 0.65, "real_at_or_below": 0.35},
        "inference_speed_ms_per_sample": 120.5,
        "calibration_note": "not calibrated",
        "limitations": ["note one"],
        "unexpected_extra_field": "ignored-safely",
    }
    f = tmp_path / "e.json"
    f.write_text(json.dumps(d))
    monkeypatch.setattr(common, "EVAL_RESULTS_PATH", str(f))
    result = common.evaluation_accuracy()
    assert result["accuracy"] == 0.9
    assert result["precision"] == 0.88
    assert result["calibration_note"] == "not calibrated"
    assert "unexpected_extra_field" not in result


def test_evaluation_accuracy_minimal_file_still_works(monkeypatch, tmp_path):
    f = tmp_path / "e.json"
    f.write_text(json.dumps({"accuracy": 0.8, "dataset": "toy", "samples": 5}))
    monkeypatch.setattr(common, "EVAL_RESULTS_PATH", str(f))
    assert common.evaluation_accuracy() == {"accuracy": 0.8, "dataset": "toy", "samples": 5}


# ---- threshold_info(): transparency without implying calibration ----

def test_threshold_info_has_expected_shape():
    info = common.threshold_info()
    assert info["fake_at_or_above"] == common.FAKE_THRESHOLD
    assert info["real_at_or_below"] == common.REAL_THRESHOLD
    assert info["selection_method"] and info["rationale"]
    # the rationale must acknowledge the lack of calibration evidence, never assert the
    # opposite (a bare, unqualified "is calibrated" claim)
    rationale = info["rationale"].lower()
    assert "calibrat" in rationale
    assert "no evidence" in rationale or "not" in rationale
