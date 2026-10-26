import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import common, detect
from app.main import app
from app.model import detector


def make_image(fmt="PNG", size=(64, 48), color=(120, 90, 60)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()


@pytest.fixture
def client():
    # no lifespan: the real model is not loaded for mocked tests
    return TestClient(app)


@pytest.fixture
def mocked(monkeypatch):
    """Detector marked ready with predict() returning a controllable probability."""
    state = {"p": 0.9}
    monkeypatch.setattr(detector, "state", "ready")
    monkeypatch.setattr(detector, "info", {"revision": "test"})
    monkeypatch.setattr(detector, "predict", lambda img: state["p"])
    monkeypatch.setattr(common, "EVAL_RESULTS_PATH", "/nonexistent.json")
    return state
