# Track & Zoom — standalone

The drag-select tracked-zoom feature, extracted from the video-indexer as its
own runnable app so it can be worked on without booting the whole pipeline
(qdrant, vLLM, whisper, the indexing stages).

Drag a box around an object on a paused frame. SAM2 follows it through the
entire clip, and a new video is rendered showing the **original footage** as the
main view — with a red outline marking the tracked region — and a **top-right
picture-in-picture of the zoomed crop** that moves with the object.

---

## Run it

### Docker (nothing to install)

```bash
docker compose up -d --build
open http://localhost:13082
```

Ports are 13082 (UI) and 18100 (API). They avoid other stacks on this host only
so everything can coexist; there is no other relationship. Override with
`TZ_FRONTEND_PORT` / `TZ_BACKEND_PORT`.

**First boot downloads the SAM2 checkpoint** (~320 MB) into `./models/sam2`,
once. It is mounted from this folder, so a rebuild does not re-fetch it. The
healthcheck allows 10 minutes for that; the API is up and serving immediately
either way.

Air-gapped, or you already have the file:

```bash
TZ_SAM2_AUTO_DOWNLOAD=false docker compose up -d --build
# and put the checkpoint here yourself:
curl -L -o models/sam2/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

### Locally, for iterating

```bash
./run.sh              # backend :18100 (reload) + vite dev server :5173
./run.sh backend      # one or the other
./run.sh frontend
```

Check what the server thinks it has:

```bash
curl -s localhost:18100/api/health | python3 -m json.tool
```

`sam2.importable` is a package check; `sam2.weights_present` is the checkpoint.
Both must be true for a render to work — the first is reported at boot, the
second only bites when a render actually loads the model, so it is surfaced
separately rather than implied.

---

## How to use it

1. **Upload a video** (mp4/mov/mkv/webm/avi/m4v).
2. **Pause** where you want tracking to start.
3. **Drag a box** around the object. That box is the zoom window *exactly* —
   no padding, no minimum size.
4. **Track & zoom.** Progress shows live; **✕ stops it mid-way** and keeps
   nothing.
5. Finished renders land in **Saved renders**. Click one to play it, ✕ to
   delete it. Renders are never overwritten.

---

## Layout

```
backend/app/
  services/follow_zoom.py     the compositor + Kalman/SMA camera path
  services/sam2_tracker.py    SAM2 bidirectional propagation
  services/render_control.py  cooperative cancellation
  routers/renders.py          the HTTP API
  config.py  models.py  store.py
frontend/src/
  App.tsx       upload / player / render / history
  BoxSelect.tsx the drag overlay (emits normalised coords)
  api.ts  styles.css
```

**This folder is self-contained.** Nothing in it reads from, mounts, imports,
or calls any other project — no shared API, no shared database, no shared
models directory. Source, uploads, renders, the JSON store and the SAM2
checkpoint all live under this directory. Copy it anywhere on its own and it
runs.

The three `services/` files began as copies of the same feature from a larger
application. They only ever imported `..config.get_settings()`, so this app
provides its own compatible one and they run unmodified — which also means a
fix made here is a file copy away from going back, if you ever want that.

---

## Settings

All are `TZ_`-prefixed env vars (or a `backend/.env`).

| Setting | Default | What it does |
|---|---|---|
| `TZ_FOLLOW_ZOOM_PADDING_SCALE` | `1.15` | Multiplier on your drag, around its centre. 1.15 leaves a thin margin outside the selection; 1.0 is a literal match. |
| `TZ_FOLLOW_ZOOM_PIP_SCALE` | `0.32` | PiP width as a fraction of the output. |
| `TZ_FOLLOW_ZOOM_MAX_PREDICT_FRAMES` | `15` | Frames the camera may coast through an occlusion before holding position. |
| `TZ_FOLLOW_ZOOM_SHOW_WINDOW_BOX` | `true` | The red outline marking the zoomed region. |
| `TZ_FOLLOW_ZOOM_SMOOTHER` | `kalman` | `kalman` or `sma`. |
| `TZ_FOLLOW_ZOOM_OUTPUT_W/H` | `1920`/`1080` | Output canvas. |
| `TZ_SAM2_MODEL` | `sam2.1_hiera_base_plus` | Checkpoint name under the weights dir. |
| `TZ_SAM2_DEVICE` | `cuda` | Falls back to CPU automatically (very slow). |
| `TZ_SAM2_WEIGHTS_DIR` | `models/sam2` | Where `<model>.pt` lives, inside this folder. |
| `TZ_SAM2_AUTO_DOWNLOAD` | `true` | Fetch the checkpoint on first boot if absent. |

---

## Design notes worth knowing

**The zoom window is fixed for the clip and slides to follow.** A window that
resized as the subject walked toward the camera would change the PiP's
magnification frame to frame, which is unwatchable. The consequence: a subject
that grows a lot can outgrow your box — drag wider, or set
`TZ_FOLLOW_ZOOM_PADDING_SCALE=1.2`.

**Occlusion coasting is bounded.** The Kalman filter extrapolates on the last
known velocity for as long as it is asked to, so a long occlusion used to send
the crop off-frame permanently. After `max_predict_frames` the path holds the
last confident position and SAM2 snaps it back on reacquisition.

**Cancellation is threaded into the services, not the progress hook.** Frame
dumping, `init_state` and both ffmpeg encodes report no progress, so a
progress-driven cancel did nothing for minutes inside them. `should_cancel` is
checked in every loop and the ffmpeg steps run killable. SAM2's `init_state`
is the one stretch that cannot be interrupted mid-way.

**Nothing here has auth.** Run it locally.
