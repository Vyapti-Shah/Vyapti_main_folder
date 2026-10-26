"""Integration test with the real model. Skipped unless the model can be loaded (needs cache or network)."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from .conftest import make_image


@pytest.fixture(scope="module")
def live():
    with TestClient(app) as c:  # runs lifespan -> loads model
        import time
        for _ in range(120):
            s = c.get("/health").json()["model_state"]
            if s != "loading":
                break
            time.sleep(2)
        if s != "ready":
            pytest.skip(f"real model unavailable ({s})")
        yield c


def test_real_inference(live):
    r = live.post("/detect/image", files={"file": ("x.png", make_image("PNG", (500, 300)), "image/png")})
    assert r.status_code == 200
    j = r.json()
    assert j["prediction"] in {"REAL", "FAKE", "INCONCLUSIVE"}
    p = j["probabilities"]
    assert 0 <= p["fake"] <= 1 and p["fake"] + p["real"] == pytest.approx(1)
    assert j["inference_ms"] > 0


def test_real_inference_deterministic(live):
    data = make_image("JPEG", (400, 400), (10, 200, 90))
    a = live.post("/detect/image", files={"file": ("x.jpg", data, "image/jpeg")}).json()
    b = live.post("/detect/image", files={"file": ("x.jpg", data, "image/jpeg")}).json()
    assert a["probabilities"] == b["probabilities"]
