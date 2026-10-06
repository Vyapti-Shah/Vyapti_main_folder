import os
from pathlib import Path

DATA = Path(os.getenv("DATA_DIR", "/data"))
for d in ("uploads", "masks", "crops", "outputs"):
    (DATA / d).mkdir(parents=True, exist_ok=True)

WORKERS = {"trellis": os.getenv("TRELLIS_URL"), "hunyuan": os.getenv("HUNYUAN_URL")}
SMPLX_DIR = Path(os.getenv("SMPLX_DIR", "/models/smpl"))


def smplx_available() -> bool:
    return (SMPLX_DIR / "smplx" / "SMPLX_NEUTRAL.npz").exists()
