import json

import pytest

from app import common, detect
from .conftest import make_image


def post(client, data, name="a.png", ctype="image/png"):
    return client.post("/detect/image", files={"file": (name, data, ctype)})


@pytest.mark.parametrize("p,label,conf", [(0.95, "FAKE", 0.95), (0.05, "REAL", 0.95), (0.5, "INCONCLUSIVE", None)])
def test_decision_rules(client, mocked, p, label, conf):
    mocked["p"] = p
    r = post(client, make_image())
    assert r.status_code == 200
    j = r.json()
    assert j["prediction"] == label
    assert j["confidence"] == (pytest.approx(conf) if conf else None)
    assert j["probabilities"]["fake"] == pytest.approx(p)
    assert j["probabilities"]["real"] == pytest.approx(1 - p)
    assert j["thresholds"]["fake_at_or_above"] == 0.6
    assert j["thresholds"]["real_at_or_below"] == 0.4
    assert j["inference_ms"] >= 0 and j["model"]["id"]


@pytest.mark.parametrize("fmt,name", [("JPEG", "a.jpg"), ("PNG", "a.png"), ("WEBP", "a.webp")])
def test_formats(client, mocked, fmt, name):
    assert post(client, make_image(fmt), name).status_code == 200


def test_accuracy_null_without_eval_file(client, mocked):
    assert post(client, make_image()).json()["evaluation"] is None


def test_accuracy_from_eval_file(client, mocked, monkeypatch, tmp_path):
    f = tmp_path / "e.json"
    f.write_text(json.dumps({"accuracy": 0.91, "dataset": "toy", "samples": 100}))
    monkeypatch.setattr(common, "EVAL_RESULTS_PATH", str(f))
    assert post(client, make_image()).json()["evaluation"] == {"accuracy": 0.91, "dataset": "toy", "samples": 100}


def test_corrupt_file(client, mocked):
    assert post(client, b"not an image at all").status_code == 400


def test_truncated_image(client, mocked):
    assert post(client, make_image("PNG", (300, 300))[:200]).status_code == 400


def test_unsupported_content_with_good_extension(client, mocked):
    assert post(client, make_image("GIF"), "a.png").status_code == 415


def test_bad_extension(client, mocked):
    assert post(client, make_image(), "a.gif").status_code == 415


def test_too_large(client, mocked, monkeypatch):
    monkeypatch.setattr(detect, "MAX_BYTES", 100)
    assert post(client, make_image("PNG", (200, 200))).status_code == 413


def test_too_many_pixels(client, mocked, monkeypatch):
    monkeypatch.setattr(detect, "MAX_PIXELS", 100)
    assert post(client, make_image()).status_code == 413


def test_empty_file(client, mocked):
    assert post(client, b"").status_code == 400


def test_missing_file(client, mocked):
    assert client.post("/detect/image").status_code == 422


def test_model_not_ready(client, monkeypatch):
    from app.model import detector
    monkeypatch.setattr(detector, "state", "failed")
    assert post(client, make_image()).status_code == 503


def test_inference_failure_returns_no_prediction(client, mocked, monkeypatch):
    from app.model import detector

    def boom(img):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(detector, "predict", boom)
    r = client.post("/detect/image", files={"file": ("a.png", make_image(), "image/png")})
    assert r.status_code == 500
    assert "prediction" not in r.json()


def test_nan_probability_rejected(client, mocked):
    mocked["p"] = float("nan")
    r = post(client, make_image())
    assert r.status_code == 500 and "prediction" not in r.json()
