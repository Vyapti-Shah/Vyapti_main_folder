import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
CASES_DIR = DATA_DIR / "cases"
CASES_DIR.mkdir(parents=True, exist_ok=True)

FRAME_INTERVAL_SEC = float(os.environ.get("FRAME_INTERVAL_SEC", "5"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "200"))

app = FastAPI(title="Forensic Video Report")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
app.mount("/media", StaticFiles(directory=CASES_DIR), name="media")
templates = Jinja2Templates(directory=BASE / "templates")

_locks: dict[str, threading.Lock] = {}


def _lock(case_id: str) -> threading.Lock:
    return _locks.setdefault(case_id, threading.Lock())


def case_dir(case_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", case_id):
        raise HTTPException(404, "Case not found")
    d = CASES_DIR / case_id
    if not d.exists():
        raise HTTPException(404, "Case not found")
    return d


def load_case(case_id: str) -> dict:
    return json.loads((case_dir(case_id) / "case.json").read_text())


def save_case(case_id: str, data: dict) -> None:
    with _lock(case_id):
        tmp = CASES_DIR / case_id / "case.json.tmp"
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(CASES_DIR / case_id / "case.json")


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def fmt_size(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def fmt_dt(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S (UTC)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "ffprobe failed")
    return json.loads(out.stdout)


def parse_rate(r: str) -> float:
    try:
        a, b = r.split("/")
        return float(a) / float(b) if float(b) else 0.0
    except Exception:
        return 0.0


def process_video(case_id: str, idx: int) -> None:
    case = load_case(case_id)
    v = case["videos"][idx]
    src = CASES_DIR / case_id / "uploads" / v["stored_name"]
    try:
        info = probe(src)
        vs = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
        if vs is None:
            raise RuntimeError("No video stream found")
        fmt = info.get("format", {})
        duration = float(fmt.get("duration") or vs.get("duration") or 0)
        st = src.stat()
        v.update(
            size=fmt_size(st.st_size),
            created=fmt_dt(st.st_ctime),
            modified=fmt_dt(st.st_mtime),
            accessed=fmt_dt(st.st_atime),
            recorded=(fmt.get("tags") or {}).get("creation_time", "Not available"),
            duration=fmt_ts(duration),
            width=vs.get("width"),
            height=vs.get("height"),
            framerate=f"{parse_rate(vs.get('avg_frame_rate', '0/0')):.2f} fps",
            codec=vs.get("codec_name"),
            sha256=sha256(src),
            status="extracting",
        )
        save_case(case_id, {**case, "videos": _replace(case["videos"], idx, v)})

        interval = max(FRAME_INTERVAL_SEC, duration / MAX_FRAMES if duration else FRAME_INTERVAL_SEC)
        fdir = CASES_DIR / case_id / "frames" / str(idx)
        fdir.mkdir(parents=True, exist_ok=True)
        t, n = 0.0, 0
        while t < max(duration, 0.01) and n < MAX_FRAMES:
            n += 1
            name = f"{n:04d}.jpg"
            r = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(src),
                 "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "4", str(fdir / name)],
                capture_output=True, timeout=120,
            )
            if r.returncode == 0 and (fdir / name).exists():
                case = load_case(case_id)
                v = case["videos"][idx]
                v["frames"].append({"n": n, "ts": fmt_ts(t), "file": f"{case_id}/frames/{idx}/{name}"})
                save_case(case_id, {**case, "videos": _replace(case["videos"], idx, v)})
            t += interval
        v["status"] = "done"
    except Exception as e:  # noqa: BLE001
        v["status"] = "failed"
        v["error"] = str(e)
    case = load_case(case_id)
    case["videos"][idx].update({k: v[k] for k in v if k != "frames"})
    save_case(case_id, case)


def _replace(lst, idx, item):
    new = list(lst)
    new[idx] = item
    return new


def run_case(case_id: str) -> None:
    n = len(load_case(case_id)["videos"])
    for i in range(n):
        process_video(case_id, i)
    case = load_case(case_id)
    case["status"] = "done"
    save_case(case_id, case)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    cases = []
    for p in sorted(CASES_DIR.glob("*/case.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            cases.append(json.loads(p.read_text()))
        except Exception:
            pass
    return templates.TemplateResponse(request, "index.html", {"cases": cases})


@app.post("/cases")
async def create_case(
    title: str = Form("Reference"),
    reference: UploadFile | None = File(None),
    videos: list[UploadFile] = File(...),
):
    videos = [v for v in videos if v.filename]
    if not videos:
        raise HTTPException(400, "Upload at least one video")
    case_id = uuid.uuid4().hex[:12]
    d = CASES_DIR / case_id
    (d / "uploads").mkdir(parents=True)

    ref_info = None
    if reference is not None and reference.filename:
        ext = Path(reference.filename).suffix.lower() or ".jpg"
        (d / f"reference{ext}").write_bytes(await reference.read())
        ref_info = {"file": f"{case_id}/reference{ext}", "name": reference.filename}

    entries = []
    for i, up in enumerate(videos):
        stored = f"{i}_{uuid.uuid4().hex[:6]}{Path(up.filename).suffix.lower()}"
        with open(d / "uploads" / stored, "wb") as out:
            shutil.copyfileobj(up.file, out)
        entries.append({
            "filename": Path(up.filename).name,
            "stored_name": stored,
            "path": f"uploads/{Path(up.filename).name}",
            "status": "queued",
            "frames": [],
        })

    case = {
        "id": case_id,
        "title": title.strip() or "Reference",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": ref_info,
        "status": "processing",
        "videos": entries,
    }
    save_case(case_id, case)
    threading.Thread(target=run_case, args=(case_id,), daemon=True).start()
    return RedirectResponse(f"/case/{case_id}", status_code=303)


@app.get("/case/{case_id}", response_class=HTMLResponse)
def report(request: Request, case_id: str):
    return templates.TemplateResponse(request, "report.html", {"case": load_case(case_id)})


@app.get("/api/case/{case_id}")
def api_case(case_id: str):
    return JSONResponse(load_case(case_id))
