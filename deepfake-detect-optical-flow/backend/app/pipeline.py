import os, cv2, math, base64, bisect, subprocess, numpy as np
from . import model as M

try:
    from retinaface import RetinaFace      # serengil/retinaface
    RF_OK = True
except Exception as e:                      # falls back to Haar so the app still runs
    print(f"[detector] RetinaFace unavailable ({e}); using Haar fallback")
    RF_OK = False

SAMPLE_FPS = float(os.getenv("SAMPLE_FPS", 5))
CLIP_LEN = int(os.getenv("CLIP_LEN", 8))
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", 400))
DET_MAX_SIDE = int(os.getenv("DET_MAX_SIDE", 960))
DET_THRESH = float(os.getenv("DET_THRESH", 0.8))
MIN_TRACK = int(os.getenv("MIN_TRACK", 3))
QMIN = float(os.getenv("QMIN", 0.25))
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ---------- detection / tracking ----------
def _haar(small, s):
    g = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    fs = cascade.detectMultiScale(g, 1.1, 4, minSize=(30, 30))
    return [((int(x / s), int(y / s), int((x + a) / s), int((y + b) / s)), None) for x, y, a, b in fs]


def detect_faces(frame):
    """Returns [((x1,y1,x2,y2), landmarks_or_None)] in original-frame coordinates."""
    global RF_OK
    h, w = frame.shape[:2]
    s = min(1.0, DET_MAX_SIDE / max(h, w))
    small = cv2.resize(frame, None, fx=s, fy=s) if s < 1 else frame
    if RF_OK:
        try:
            res = RetinaFace.detect_faces(small, threshold=DET_THRESH)
            out = []
            if isinstance(res, dict):
                for f in res.values():
                    x1, y1, x2, y2 = [v / s for v in f["facial_area"]]
                    lm = {k: (v[0] / s, v[1] / s) for k, v in f.get("landmarks", {}).items()}
                    out.append(((int(x1), int(y1), int(x2), int(y2)), lm or None))
            if out:
                return out
        except Exception as e:
            print(f"[detector] RetinaFace failed ({e}); switching to Haar fallback"); RF_OK = False
    return _haar(small, s)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0


class Tracker:
    """Greedy IoU tracker: keeps each face as its own temporal sequence."""
    def __init__(self):
        self.tracks, self.nid = [], 1

    def update(self, obs_idx, dets):
        used = set()
        for b, lm in dets:
            best, bi = 0.2, None
            for t in self.tracks:
                if id(t) in used or obs_idx - t["last_idx"] > 5: continue
                v = iou(b, t["box"])
                if v > best: best, bi = v, t
            if bi is None:
                bi = {"id": self.nid, "obs": [], "box": b, "last_idx": obs_idx}; self.nid += 1
                self.tracks.append(bi)
            bi["box"], bi["last_idx"] = b, obs_idx; used.add(id(bi))
            yield bi, b, lm


def crop_face(frame, box, lm=None, margin=0.2):
    """224x224 face crop; rotated so the eyes are level when RetinaFace landmarks exist."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    ang = 0.0
    if lm and "right_eye" in lm and "left_eye" in lm:
        re_, le = lm["right_eye"], lm["left_eye"]
        a = math.degrees(math.atan2(le[1] - re_[1], le[0] - re_[0]))
        if abs(a) < 45: ang = a
    Mx = cv2.getRotationMatrix2D((cx, cy), ang, 224 / side)
    Mx[0, 2] += 112 - cx; Mx[1, 2] += 112 - cy
    return cv2.warpAffine(frame, Mx, (224, 224), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def quality(crop, box):
    g = cv2.cvtColor(cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    blur = min(1.0, cv2.Laplacian(g, cv2.CV_64F).var() / 40.0)
    size = min(1.0, (box[2]-box[0]) / 80.0)
    bright = 1 - abs(float(g.mean()) - 128) / 128
    return float(0.6*blur + 0.25*size + 0.15*bright)


# ---------- optical flow + clip selection ----------
def flow_feats(a, b):
    fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.linalg.norm(fl, axis=2); H, W = mag.shape
    rv = np.array([fl[i*H//3:(i+1)*H//3, j*W//3:(j+1)*W//3].reshape(-1, 2).mean(0)
                   for i in range(3) for j in range(3)])
    reg = float(np.linalg.norm(rv - rv.mean(0), axis=1).mean())
    m = mag > 0.05
    dirc = float(1 - np.linalg.norm((fl[m] / mag[m][:, None]).mean(0))) if m.sum() > 10 else 0.0
    return dict(mean=float(mag.mean()), var=float(mag.var()), reg=reg, dir=dirc)


def flow_scores(F):
    if not F: return [], []
    g = lambda k: np.array([f[k] for f in F])
    mean = g("mean"); chg = np.abs(np.diff(mean, prepend=mean[0]))
    nz = lambda x: np.clip(x / (np.percentile(x, 95) + 1e-6), 0, 1.5)
    s = (0.35*nz(g("var")) + 0.30*nz(g("reg")) + 0.20*nz(g("dir")) + 0.15*nz(chg)) / 1.5
    return s.tolist(), mean.tolist()


def select_clips(obs, qmin=QMIN):
    usable = [i for i, o in enumerate(obs) if o["q"] >= qmin]
    if not usable: return []
    cand = [i for i in usable if obs[i]["mag"] >= 0.02]
    if len(cand) < 3: cand = usable
    sc = np.array([obs[i]["s"] for i in cand])
    thr = np.percentile(sc, 75)
    peaks = [cand[k] for k in range(len(cand))
             if sc[k] >= thr and (k == 0 or sc[k] >= sc[k-1]) and (k == len(cand)-1 or sc[k] >= sc[k+1])]
    peaks = sorted(peaks, key=lambda i: -obs[i]["s"])[:4]
    normal = cand[int(np.argmin(np.abs(sc - np.median(sc))))]
    pos = {u: k for k, u in enumerate(usable)}
    L = len(usable); cl = min(CLIP_LEN, L); clips = set()
    for c in peaks + [normal]:
        st = min(max(pos[c] - cl // 2, 0), L - cl)
        clips.add(tuple(usable[st:st + cl]))
    return [list(c) for c in sorted(clips)][:5]


def thumb(crop):
    ok, buf = cv2.imencode(".jpg", cv2.resize(crop, (96, 96)), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def verdict(p):
    if not M.MODEL_LOADED: return "UNVERIFIED"
    return "DEEPFAKE" if p >= 0.5 else "REAL"


# ---------- plain-language explanation ----------
def tfmt(t):
    m, s = divmod(int(t), 60); return f"{m}:{s:02d}"


def explain(p, segs, n_clips, relaxed, flow_anom, demo, image=False):
    if demo:
        return dict(level="demo", headline="No verdict yet: the detector has not been trained",
                    lines=["The video was processed (faces found, motion checked, clips built), but the AI model has no "
                           "trained deepfake knowledge yet, so it cannot tell real from fake.",
                           "Any percentage it produces right now is random and should be ignored.",
                           "To get real results, add a trained model file (backend/models/deepfake_model.pt)."])
    if p >= 0.75: lvl, head = "manipulated", "Likely manipulated (deepfake)"
    elif p >= 0.5: lvl, head = "suspicious", "Possibly manipulated: please double-check"
    elif p >= 0.25: lvl, head = "uncertain", "Probably authentic, but not certain"
    else: lvl, head = "authentic", "Likely authentic: no signs of manipulation found"
    lines = []
    if image:
        lines.append("The face in this image shows signs of being altered or AI-generated." if p >= 0.5
                     else "The face in this image looks natural.")
    elif segs:
        lines.append(f"{len(segs)} of {n_clips} checked clips looked altered, around "
                     + ", ".join(f"{tfmt(a)}-{tfmt(b)}" for a, b, _ in segs[:4])
                     + ". These moments are marked in red on the video.")
        if flow_anom >= 0.6:
            lines.append("Facial movement looked unnatural in places (parts of the face moving out of sync).")
    else:
        lines.append(f"None of the {n_clips} checked clips looked altered.")
    if relaxed: lines.append("The video quality is low, so treat this result with extra caution.")
    lines.append("This is an AI estimate, not proof. For important decisions, ask a human expert to review.")
    return dict(level=lvl, headline=head, lines=lines)


# ---------- watermark / annotation ----------
COL = {"fake": (60, 60, 255), "ok": (90, 190, 90), "none": (150, 150, 150), "demo": (150, 150, 150)}


def _label(state, p):
    return {"fake": f"SUSPECTED FAKE {p*100:.0f}%", "ok": "No manipulation found",
            "none": "Not scored", "demo": "DEMO - NOT SCORED"}[state]


def draw_face(fr, box, state, p):
    x1, y1, x2, y2 = [int(v) for v in box]; c = COL[state]; w = fr.shape[1]
    cv2.rectangle(fr, (x1, y1), (x2, y2), c, max(2, w // 300))
    fs = max(0.45, w / 1800); txt = _label(state, p)
    (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    y0 = max(0, y1 - tht - 12)
    cv2.rectangle(fr, (x1, y0), (x1 + tw + 10, y0 + tht + 12), c, -1)
    cv2.putText(fr, txt, (x1 + 5, y0 + tht + 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)


def watermark(fr):
    h, w = fr.shape[:2]
    cv2.rectangle(fr, (0, 0), (w - 1, h - 1), (60, 60, 255), max(4, w // 150))
    bh = max(34, h // 12); ov = fr.copy()
    cv2.rectangle(ov, (0, h - bh), (w, h), (30, 30, 200), -1)
    fr[:] = cv2.addWeighted(ov, 0.6, fr, 0.4, 0)
    txt = "POSSIBLE DEEPFAKE - AI-FLAGGED SEGMENT"; fs = max(0.5, w / 1400)
    (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    cv2.putText(fr, txt, ((w - tw) // 2, h - bh // 2 + tht // 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), 2, cv2.LINE_AA)


def _nearest(v, t):
    ts = v["times"]; i = bisect.bisect_left(ts, t)
    c = [j for j in (i - 1, i) if 0 <= j < len(ts)]
    if not c: return None
    j = min(c, key=lambda j: abs(ts[j] - t))
    return j if abs(ts[j] - t) <= 1.0 else None


def _state(v, t, demo):
    if demo: return "demo", 0.0
    for a, b, p in v["segs"]:
        if a <= t <= b: return ("fake" if p >= 0.5 else "ok"), p
    return "none", 0.0


def render_annotated(path, viz, fps, out_path, demo):
    """Re-encodes the video (H.264 via ffmpeg) with face boxes + red watermark on suspected-fake segments."""
    cap = cv2.VideoCapture(path)
    W, H = int(cap.get(3)), int(cap.get(4))
    sc = min(1.0, 1280 / max(W, H)); ow, oh = int(W * sc) // 2 * 2, int(H * sc) // 2 * 2
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
           "-r", f"{fps}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        print("[annotate] ffmpeg not installed"); cap.release(); return False
    fx, fy, fi = ow / W, oh / H, 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if (ow, oh) != (W, H): fr = cv2.resize(fr, (ow, oh))
        t, any_fake = fi / fps, False
        for v in viz:
            j = _nearest(v, t)
            if j is None: continue
            st, p = _state(v, t, demo)
            x1, y1, x2, y2 = v["boxes"][j]
            draw_face(fr, (x1 * fx, y1 * fy, x2 * fx, y2 * fy), st, p)
            any_fake |= st == "fake"
        if any_fake: watermark(fr)
        try: proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        except BrokenPipeError: break
        fi += 1
    cap.release()
    try: proc.stdin.close()
    except Exception: pass
    proc.wait()
    return proc.returncode == 0 and os.path.exists(out_path)


# ---------- image path ----------
def analyze_image(img, out_path=None):
    M.get_model()
    faces = detect_faces(img)
    if not faces: raise ValueError("No face detected in image")
    box, lm = max(faces, key=lambda d: (d[0][2]-d[0][0])*(d[0][3]-d[0][1]))
    crop = crop_face(img, box, lm); q = quality(crop, box)
    p, _ = M.predict_clip([crop])
    demo = not M.MODEL_LOADED
    annotated = False
    if out_path:
        vis = img.copy(); st = "demo" if demo else ("fake" if p >= 0.5 else "ok")
        draw_face(vis, box, st, p)
        if st == "fake": watermark(vis)
        annotated = bool(cv2.imwrite(out_path, vis))
    warn = [] if q >= QMIN else ["Face quality is low (blur/size/lighting); treat result with caution."]
    if demo: warn.append("DEMO MODE: no trained weights loaded, so scores are meaningless.")
    return dict(type="image", verdict=verdict(p), p_fake=p, confidence=max(p, 1-p), model_loaded=M.MODEL_LOADED,
                face_quality=q, thumbs=[thumb(crop)], warnings=warn, annotated=annotated, suspicious_segments=[],
                summary=explain(p, [], 1, q < QMIN, 0, demo, image=True))


# ---------- video path ----------
def analyze_video(path, out_path=None):
    M.get_model()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, round(fps / SAMPLE_FPS))
    if total: step = max(step, math.ceil(total / MAX_SAMPLES))
    info = dict(fps=round(fps, 2), total_frames=total, duration=round(total / fps, 2),
                width=int(cap.get(3)), height=int(cap.get(4)), sample_step=step)

    tracker, n_sampled, n_det, fi, oi = Tracker(), 0, 0, 0, 0
    while True:
        if not cap.grab(): break
        if fi % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                n_sampled += 1
                dets = detect_faces(frame)
                if dets: n_det += 1
                for t, b, lm in tracker.update(oi, dets):
                    c = crop_face(frame, b, lm)
                    t["obs"].append(dict(fi=fi, t=fi / fps, box=b, crop=c, q=quality(c, b),
                                         gray=cv2.cvtColor(cv2.resize(c, (112, 112)), cv2.COLOR_BGR2GRAY)))
                oi += 1
        fi += 1
    cap.release()
    total = total or fi
    longest = max([len(t["obs"]) for t in tracker.tracks], default=0)
    print(f"[video] sampled={n_sampled} frames_with_face={n_det} tracks={len(tracker.tracks)} "
          f"longest_track={longest} retinaface={RF_OK}")

    tracks = sorted([t for t in tracker.tracks if len(t["obs"]) >= MIN_TRACK], key=lambda t: -len(t["obs"]))[:3]
    if not tracks:
        raise ValueError(f"No stable face track found: faces detected in {n_det}/{n_sampled} sampled frames, "
                         f"longest track {longest} frames (need {MIN_TRACK}). "
                         f"Use a clip with a larger, clearer, mostly frontal face.")

    out_tracks, viz, all_sel, all_clips, excluded = [], [], 0, 0, 0
    best, relaxed, dt = None, False, step / fps
    for t in tracks:
        obs = t["obs"]
        F = [flow_feats(obs[i-1]["gray"], obs[i]["gray"]) for i in range(1, len(obs))]
        s, mag = flow_scores(F)
        for i, o in enumerate(obs):
            o["s"] = s[max(i-1, 0)]; o["mag"] = mag[max(i-1, 0)]
        qs = np.array([o["q"] for o in obs]); qmin = QMIN
        if (qs >= QMIN).sum() < 3:
            qmin = float(np.percentile(qs, 40)); relaxed = True
        print(f"[video] track {t['id']}: frames={len(obs)} quality min/median/max="
              f"{qs.min():.2f}/{np.median(qs):.2f}/{qs.max():.2f} threshold={qmin:.2f}")
        blur_anoms = [round(obs[i]["t"], 2) for i in range(1, len(obs)) if obs[i-1]["q"] - obs[i]["q"] > 0.4]
        excluded += int((qs < qmin).sum())

        clips = select_clips(obs, qmin); clip_res, sel_frame_p = [], {}
        for c in clips:
            p, fp = M.predict_clip([obs[i]["crop"] for i in c])
            for i, v in zip(c, fp): sel_frame_p[i] = v
            clip_res.append(dict(start=round(obs[c[0]]["t"], 2), end=round(obs[c[-1]]["t"], 2),
                                 frames=len(c), p_fake=p, weight=float(np.mean([obs[i]["q"] for i in c])) + 1e-6))
        if not clip_res: continue
        w = np.array([c["weight"] for c in clip_res]); pv = np.array([c["p_fake"] for c in clip_res])
        p_track = float((w * pv).sum() / w.sum())
        selected = sorted(sel_frame_p)
        tl = [dict(t=round(o["t"], 2), flow=round(o["s"], 3), quality=round(o["q"], 3),
                   selected=i in sel_frame_p) for i, o in enumerate(obs)]
        rec = dict(track_id=t["id"], frames=len(obs), p_fake=p_track, clips=clip_res, timeline=tl,
                   blur_transitions=blur_anoms,
                   spatial_artifact=float(np.mean(list(sel_frame_p.values()))),
                   flow_anomaly=float(np.mean([obs[i]["s"] for i in selected])),
                   temporal_consistency=float(1 - np.mean(s)) if s else 1.0)
        out_tracks.append(rec); all_sel += len(selected); all_clips += len(clip_res)
        viz.append(dict(times=[o["t"] for o in obs], boxes=[o["box"] for o in obs],
                        segs=[(c["start"], c["end"] + dt, c["p_fake"]) for c in clip_res]))
        if best is None or p_track > best[0]:
            best = (p_track, rec, [thumb(obs[i]["crop"]) for i in selected[:12]])
    if best is None: raise ValueError("No usable face frames found after quality filtering")

    p, rec, thumbs = best
    demo = not M.MODEL_LOADED
    sus = sorted([dict(start=c["start"], end=round(c["end"] + dt, 2), p_fake=c["p_fake"])
                  for r in out_tracks for c in r["clips"] if c["p_fake"] >= 0.5], key=lambda x: x["start"])
    warns = []
    if relaxed: warns.append("Low-quality video (blurry/small/dark faces): analysed the best available frames, so results are less reliable.")
    if excluded: warns.append(f"{excluded} low-quality face frames (blur/occlusion/small) were excluded from model inference.")
    if any(t["blur_transitions"] for t in out_tracks): warns.append("Sudden blur transitions detected; kept as temporal anomaly signal.")
    annotated = False
    if out_path:
        annotated = render_annotated(path, viz, fps, out_path, demo)
        if not annotated: warns.append("Could not create the annotated video (ffmpeg problem); showing the original.")
    return dict(type="video", verdict=verdict(p), p_fake=p, confidence=max(p, 1-p), model_loaded=M.MODEL_LOADED,
                video=info, annotated=annotated, suspicious_segments=sus,
                summary=explain(p, [(x["start"], x["end"], x["p_fake"]) for x in sus], all_clips, relaxed, rec["flow_anomaly"], demo),
                pipeline=dict(original_frames=total, candidate_frames=n_sampled, selected_frames=all_sel,
                              clips=all_clips, faces_tracked=len(out_tracks)),
                metrics=dict(flow_anomaly=rec["flow_anomaly"], spatial_artifact=rec["spatial_artifact"],
                             temporal_consistency=rec["temporal_consistency"]),
                timeline=rec["timeline"], clips=rec["clips"], tracks=out_tracks, thumbs=thumbs, warnings=warns)