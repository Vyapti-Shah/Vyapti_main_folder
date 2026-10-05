import httpx

from config import WORKERS


def generate_object(engine, png_path, out_path, seed=1):
    url = WORKERS.get(engine)
    if not url:
        raise ValueError(f"unknown engine {engine}")
    with open(png_path, "rb") as fh:
        r = httpx.post(f"{url}/generate", files={"file": ("in.png", fh, "image/png")},
                       data={"seed": seed}, timeout=httpx.Timeout(1800, connect=10))
    r.raise_for_status()
    out_path.write_bytes(r.content)


def worker_up(engine):
    try:
        return httpx.get(f"{WORKERS[engine]}/health", timeout=2).status_code == 200
    except Exception:
        return False
