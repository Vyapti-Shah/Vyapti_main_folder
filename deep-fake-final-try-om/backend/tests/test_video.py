import io

import cv2
import numpy as np
import pytest

from app import video
from .conftest import make_image


def make_mp4(num_frames=10, fps=10, size=(64, 48)) -> bytes:
    """Build a tiny valid mp4 in-memory via a temp path (cv2.VideoWriter needs a path)."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), (i * 10) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    with open(path, "rb") as f:
        data = f.read()
    import os

    os.unlink(path)
    return data


def post(client, data, name="a.mp4", ctype="video/mp4"):
    return client.post("/detect/video", files={"file": (name, data, ctype)})


# ---- aggregation (unit) ----

def test_aggregate_mean():
    assert video.aggregate([0.2, 0.4, 0.6]) == pytest.approx(0.4)


def test_aggregate_single_frame():
    assert video.aggregate([0.7]) == 0.7


# ---- endpoint ----

def test_detect_video_happy_path(client, mocked):
    mocked["p"] = 0.8
    r = post(client, make_mp4())
    assert r.status_code == 200
    j = r.json()
    assert j["prediction"] == "FAKE"
    assert j["probabilities"]["fake"] == pytest.approx(0.8)
    assert j["frames_analyzed"] > 0
    assert len(j["frame_scores"]) == j["frames_analyzed"]
    assert all(s == pytest.approx(0.8) for s in j["frame_scores"])
    assert j["aggregation"]["method"] == "mean_probability"
    assert "not temporal" in j["aggregation"]["description"]
    assert "not" in j["note"] and "temporal" in j["note"]
    assert j["model"]["id"]
    assert j["inference_ms"] >= 0


def test_frame_sampling_caps_frame_count(client, mocked, monkeypatch):
    monkeypatch.setattr(video, "MAX_FRAMES", 2)
    monkeypatch.setattr(video, "SAMPLE_INTERVAL_SECONDS", 0.0)
    r = post(client, make_mp4(num_frames=20, fps=10))
    assert r.status_code == 200
    assert r.json()["frames_analyzed"] <= 2


def test_empty_file(client, mocked):
    assert post(client, b"").status_code == 400


def test_corrupt_video(client, mocked):
    assert post(client, b"not a real video, just bytes" * 10).status_code == 400


def test_video_with_no_decodable_frames(client, mocked, monkeypatch):
    # a validly-opened capture that immediately fails to read any frame
    class FakeCap:
        def isOpened(self):
            return True

        def get(self, prop):
            return 25.0

        def set(self, prop, val):
            pass

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(video.cv2, "VideoCapture", lambda path: FakeCap())
    r = post(client, make_mp4())
    assert r.status_code == 400
    assert "no decodable frames" in r.json()["detail"].lower()


def test_unsupported_extension(client, mocked):
    assert post(client, make_mp4(), "a.txt").status_code == 415


def test_too_large(client, mocked, monkeypatch):
    monkeypatch.setattr(video, "MAX_BYTES", 100)
    assert post(client, make_mp4(num_frames=30)).status_code == 413


def test_model_not_ready(client, monkeypatch):
    from app.model import detector

    monkeypatch.setattr(detector, "state", "loading")
    assert post(client, make_mp4()).status_code == 503


def test_inference_failure_returns_no_prediction(client, mocked, monkeypatch):
    from app.model import detector

    def boom(img):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(detector, "predict", boom)
    r = post(client, make_mp4())
    assert r.status_code == 500
    assert "prediction" not in r.json()


def test_nan_probability_rejected(client, mocked):
    mocked["p"] = float("nan")
    r = post(client, make_mp4())
    assert r.status_code == 500 and "prediction" not in r.json()


def test_temp_file_cleaned_up_on_success(client, mocked, tmp_path, monkeypatch):
    import tempfile

    created = []
    real_named_tmp = tempfile.NamedTemporaryFile

    def tracking_named_tmp(*a, **kw):
        f = real_named_tmp(*a, **kw)
        created.append(f.name)
        return f

    monkeypatch.setattr(video.tempfile, "NamedTemporaryFile", tracking_named_tmp)
    assert post(client, make_mp4()).status_code == 200
    assert created and not any(__import__("os").path.exists(p) for p in created)


def test_temp_file_cleaned_up_on_inference_failure(client, mocked, monkeypatch):
    import os
    import tempfile

    created = []
    real_named_tmp = tempfile.NamedTemporaryFile

    def tracking_named_tmp(*a, **kw):
        f = real_named_tmp(*a, **kw)
        created.append(f.name)
        return f

    monkeypatch.setattr(video.tempfile, "NamedTemporaryFile", tracking_named_tmp)

    from app.model import detector

    def boom(img):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(detector, "predict", boom)
    assert post(client, make_mp4()).status_code == 500
    assert created and not any(os.path.exists(p) for p in created)


def test_temp_file_cleaned_up_on_corrupt_video(client, mocked, monkeypatch):
    import os
    import tempfile

    created = []
    real_named_tmp = tempfile.NamedTemporaryFile

    def tracking_named_tmp(*a, **kw):
        f = real_named_tmp(*a, **kw)
        created.append(f.name)
        return f

    monkeypatch.setattr(video.tempfile, "NamedTemporaryFile", tracking_named_tmp)
    assert post(client, b"garbage").status_code == 400
    assert created and not any(os.path.exists(p) for p in created)


# ---- aggregation: additional boundary/edge cases ----

def test_aggregate_all_zero():
    assert video.aggregate([0.0, 0.0, 0.0]) == 0.0


def test_aggregate_all_one():
    assert video.aggregate([1.0, 1.0]) == 1.0


def test_aggregate_is_order_independent():
    a = [0.1, 0.9, 0.3, 0.7]
    assert video.aggregate(a) == video.aggregate(list(reversed(a)))


def test_aggregate_result_lands_on_threshold_boundary(mocked):
    # mean exactly at FAKE_THRESHOLD must decide FAKE (inclusive boundary), consistent
    # with the image endpoint's decide() semantics.
    from app.common import FAKE_THRESHOLD, decide

    scores = [FAKE_THRESHOLD, FAKE_THRESHOLD, FAKE_THRESHOLD]
    assert video.aggregate(scores) == pytest.approx(FAKE_THRESHOLD)
    assert decide(video.aggregate(scores))[0] == "FAKE"


def test_video_threshold_info_present_in_response(client, mocked):
    mocked["p"] = 0.8
    r = post(client, make_mp4())
    j = r.json()
    assert "selection_method" in j["thresholds"]
    assert "rationale" in j["thresholds"]
