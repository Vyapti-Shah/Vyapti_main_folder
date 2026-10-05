'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ICON = {person:'🧍', bicycle:'🚲', car:'🚗', motorcycle:'🏍', bus:'🚌', truck:'🚚', train:'🚆'};
const CLASS_IDS = {person:0, bicycle:1, car:2, motorcycle:3, bus:5, train:6, truck:7};
const cap = s => s[0].toUpperCase() + s.slice(1);
const fmtT = t => `${String(Math.floor(t / 60)).padStart(2, '0')}:${(t % 60).toFixed(3).padStart(6, '0')}`;
const S = {video: null, mode: 'known_distance', lines: {A: null, B: null}, drag: null, cal: null, settings: null, jobId: null, timer: null};

/* ------------------------------------------------------------ helpers */
async function api(url, opts) {
  const r = await fetch(url, opts);
  let d = null; try { d = await r.json(); } catch (_) {}
  if (!r.ok) throw new Error(d && d.detail ? (typeof d.detail === 'string' ? d.detail : 'Invalid input – check the values and try again.') : `Request failed (${r.status})`);
  return d;
}
const postJSON = (url, body) => api(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
function xhrUpload(url, fd, onProgress) {
  return new Promise((res, rej) => {
    const x = new XMLHttpRequest();
    x.open('POST', url);
    x.upload.onprogress = e => e.lengthComputable && onProgress(e.loaded / e.total);
    x.onload = () => { let d = null; try { d = JSON.parse(x.responseText); } catch (_) {} x.status < 300 ? res(d) : rej(new Error(d && typeof d.detail === 'string' ? d.detail : `Upload failed (${x.status})`)); };
    x.onerror = () => rej(new Error('Network error – is the backend running?'));
    x.send(fd);
  });
}
const STEP_INDEX = {upload: 0, calib: 1, confirm: 1, run: 2, results: 3, image: 3};
function go(step) {
  Object.keys(STEP_INDEX).forEach(s => $('#step-' + s).hidden = s !== step);
  $$('#stepper li').forEach((li, i) => li.className = i < STEP_INDEX[step] ? 'done' : i === STEP_INDEX[step] ? 'on' : '');
  window.scrollTo({top: 0});
}
const setErr = (id, m) => $('#err-' + id).textContent = m || '';

/* ------------------------------------------------------------ 1 · upload */
$('#file').addEventListener('change', e => e.target.files[0] && upload(e.target.files[0]));
const drop = $('#drop');
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));

async function upload(file) {
  setErr('upload'); const isImg = file.type.startsWith('image/');
  const fd = new FormData(); fd.append('file', file);
  const bar = $('#up-bar'); bar.hidden = false; $('i', bar).style.width = '0';
  $('#up-msg').textContent = isImg ? 'Detecting objects…' : 'Uploading and analysing the video…';
  try {
    const d = await xhrUpload(isImg ? '/api/images' : '/api/videos', fd, p => { $('i', bar).style.width = (p * 100) + '%'; if (p >= 1) $('#up-msg').textContent = isImg ? 'Detecting objects…' : 'Checking frame rate and camera motion…'; });
    isImg ? showImage(d) : showCalibration(file.name, d);
  } catch (e) { setErr('upload', e.message); }
  finally { bar.hidden = true; $('#up-msg').textContent = ''; $('#file').value = ''; }
}

function showImage(d) {
  $('#img-out').src = d.annotated_url + '?t=' + Date.now();
  const rows = d.detections.map(x => `<tr><td>${ICON[x.cls] || ''} ${esc(cap(x.cls))}</td><td>${(x.conf * 100).toFixed(0)}%</td></tr>`).join('');
  $('#img-table').innerHTML = `<tr><th>Object</th><th>Confidence</th></tr>${rows || '<tr><td colspan="2">No objects found.</td></tr>'}`;
  go('image');
}

/* ------------------------------------------------------------ 2 · calibration */
function showCalibration(name, v) {
  S.video = v; S.lines = {A: null, B: null};
  $('#vid-meta').textContent = `${name} – ${v.width}×${v.height}, ${v.fps} FPS${v.fps_assumed ? ' (assumed)' : ''}${v.duration_s ? ', ' + v.duration_s + ' s' : ''}`;
  $('#cam-warn').hidden = !(v.camera && v.camera.moving);
  setErr('calib'); setMode('known_distance'); loadFrame(v.frame_url); go('calib');
}
function setMode(m) {
  S.mode = m;
  $$('.mode').forEach(b => { const on = b.dataset.mode === m; b.classList.toggle('active', on); b.setAttribute('aria-checked', on); });
  $$('.panel').forEach(p => p.hidden = p.id !== 'panel-' + m);
  if (m === 'known_distance') requestAnimationFrame(draw);
}
$$('.mode').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
$$('[data-switch]').forEach(b => b.addEventListener('click', () => setMode(b.dataset.switch)));
$('#lensType').addEventListener('change', e => { const mm = e.target.value === 'mm'; $('#lens-fov').hidden = mm; $('#lens-mm').hidden = !mm; $('#lens-mm2').hidden = !mm; });

/* --- line drawing canvas: straight lines only, endpoints + body draggable --- */
const cv = $('#cv'), ctx = cv.getContext('2d'), img = new Image();
function loadFrame(url) { img.onload = () => { cv.width = img.naturalWidth; cv.height = img.naturalHeight; draw(); status(); }; img.src = url + '?t=' + Date.now(); }
const scale = () => cv.width / (cv.getBoundingClientRect().width || cv.width);
function pos(e) { const r = cv.getBoundingClientRect(); return {x: (e.clientX - r.left) * cv.width / r.width, y: (e.clientY - r.top) * cv.height / r.height}; }
const LINE = {A: {col: '#22c493', names: ['P1', 'P2']}, B: {col: '#f59a3c', names: ['P4', 'P3']}};

function draw() {
  if (!img.naturalWidth) return;
  const s = scale(), {A, B} = S.lines;
  ctx.clearRect(0, 0, cv.width, cv.height); ctx.drawImage(img, 0, 0);
  if (A && B) {
    ctx.beginPath(); [A[0], A[1], B[1], B[0]].forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.closePath();
    ctx.fillStyle = 'rgba(241,194,50,.18)'; ctx.fill();
  }
  for (const k of ['A', 'B']) {
    const L = S.lines[k]; if (!L) continue;
    ctx.strokeStyle = LINE[k].col; ctx.lineWidth = 3 * s; ctx.beginPath(); ctx.moveTo(L[0].x, L[0].y); ctx.lineTo(L[1].x, L[1].y); ctx.stroke();
    ctx.font = `600 ${15 * s}px sans-serif`; ctx.textBaseline = 'middle';
    L.forEach((p, i) => {
      ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(p.x, p.y, 8 * s, 0, 7); ctx.fill();
      ctx.fillStyle = LINE[k].col; ctx.beginPath(); ctx.arc(p.x, p.y, 5 * s, 0, 7); ctx.fill();
      ctx.lineWidth = 3 * s; ctx.strokeStyle = 'rgba(0,0,0,.7)'; ctx.strokeText(LINE[k].names[i], p.x + 12 * s, p.y - 12 * s);
      ctx.fillStyle = '#fff'; ctx.fillText(LINE[k].names[i], p.x + 12 * s, p.y - 12 * s);
    });
    const mx = (L[0].x + L[1].x) / 2, my = (L[0].y + L[1].y) / 2;
    ctx.strokeText(`Line ${k}`, mx + 8 * s, my + 16 * s); ctx.fillStyle = '#fff'; ctx.fillText(`Line ${k}`, mx + 8 * s, my + 16 * s);
  }
}
function status() {
  const {A, B} = S.lines;
  $('#draw-status').textContent = !A ? 'Click and drag to draw Line A (P1 → P2).' : !B ? 'Now draw Line B (P4 → P3) across the road, in the same direction as Line A.' : 'Both lines set. Drag an end-point to rotate or resize, or drag a line to move it.';
}
const nextToDraw = () => !S.lines.A ? 'A' : !S.lines.B ? 'B' : null;
const dSeg = (p, a, b) => { const dx = b.x - a.x, dy = b.y - a.y, l2 = dx * dx + dy * dy || 1; const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / l2)); return Math.hypot(p.x - a.x - t * dx, p.y - a.y - t * dy); };

cv.addEventListener('pointerdown', e => {
  const p = pos(e), s = scale(); let hit = null;
  for (const k of ['A', 'B']) { const L = S.lines[k]; if (!L) continue;
    L.forEach((q, i) => { if (!hit && Math.hypot(p.x - q.x, p.y - q.y) <= 16 * s) hit = {k, kind: 'end', i}; }); }
  if (!hit) for (const k of ['A', 'B']) { const L = S.lines[k]; if (L && dSeg(p, L[0], L[1]) <= 11 * s) { hit = {k, kind: 'move', last: p}; break; } }
  if (!hit && nextToDraw()) { const k = nextToDraw(); S.lines[k] = [{...p}, {...p}]; hit = {k, kind: 'end', i: 1, fresh: true}; }
  if (!hit) return;
  S.drag = hit; cv.setPointerCapture(e.pointerId); e.preventDefault();
});
cv.addEventListener('pointermove', e => {
  const d = S.drag; if (!d) return; const p = pos(e), L = S.lines[d.k];
  p.x = Math.max(0, Math.min(cv.width, p.x)); p.y = Math.max(0, Math.min(cv.height, p.y));
  if (d.kind === 'end') L[d.i] = p;
  else { const dx = p.x - d.last.x, dy = p.y - d.last.y; if (L.every(q => q.x + dx >= 0 && q.x + dx <= cv.width && q.y + dy >= 0 && q.y + dy <= cv.height)) { L.forEach(q => { q.x += dx; q.y += dy; }); d.last = p; } }
  draw();
});
const endDrag = () => {
  const d = S.drag; if (!d) return; S.drag = null;
  const L = S.lines[d.k];
  if (L && Math.hypot(L[0].x - L[1].x, L[0].y - L[1].y) < 15 * scale()) S.lines[d.k] = null;   // too short: discard
  const {A, B} = S.lines;                                                                       // keep P1/P4 on the same side
  if (A && B && (A[1].x - A[0].x) * (B[1].x - B[0].x) + (A[1].y - A[0].y) * (B[1].y - B[0].y) < 0) S.lines.B = [B[1], B[0]];
  draw(); status();
};
cv.addEventListener('pointerup', endDrag); cv.addEventListener('pointercancel', endDrag);
$('#redrawA').onclick = () => { S.lines.A = null; draw(); status(); };
$('#redrawB').onclick = () => { S.lines.B = null; draw(); status(); };
$('#clearLines').onclick = () => { S.lines = {A: null, B: null}; draw(); status(); };
window.addEventListener('resize', draw);

/* --- collect inputs --- */
function num(sel) { const v = parseFloat($(sel).value); if (!(v > 0)) throw new Error('Fill in every calibration value with a number greater than 0.'); return v; }
function collectCalibration() {
  if (S.mode === 'known_distance') {
    if (!S.lines.A || !S.lines.B) throw new Error('Draw both Line A and Line B on the road first.');
    const pts = L => L.map(p => [p.x, p.y]);
    return {mode: S.mode, line_a: pts(S.lines.A), line_b: pts(S.lines.B), len_a_m: num('#lenA'), len_b_m: num('#lenB'), sep_m: num('#sep')};
  }
  if (S.mode === 'camera_params') {
    const mm = $('#lensType').value === 'mm';
    return {mode: S.mode, cam_height_m: num('#camH'), tilt_deg: num('#tilt'), hfov_deg: mm ? null : num('#hfov'), focal_mm: mm ? num('#focalMm') : null, sensor_width_mm: mm ? num('#sensorMm') : null};
  }
  return {mode: S.mode, hfov_deg: num('#autoFov')};
}
function collectSettings() {
  const classes = $$('.cls:checked').map(c => CLASS_IDS[c.value]);
  if (!classes.length) throw new Error('Select at least one object type in Detection settings.');
  return {classes, imgsz: +$('#imgsz').value, stride: +$('#stride').value, window_s: +$('#win').value, conf: +$('#conf').value};
}

$('#btn-calibrate').onclick = async () => {
  setErr('calib'); const btn = $('#btn-calibrate');
  try {
    const cal = collectCalibration(), settings = collectSettings();
    btn.disabled = true;
    const r = await postJSON('/api/calibrate', {video_id: S.video.video_id, calibration: cal});
    S.cal = cal; S.settings = settings; showConfirm(r); go('confirm');
  } catch (e) { setErr('calib', e.message); }
  finally { btn.disabled = false; }
};

/* ------------------------------------------------------------ 3 · summary */
function showConfirm(r) {
  const checks = [...r.checks, `FPS: ${r.fps}`, `Camera: ${r.camera}${r.camera_known ? '' : ' (could not be verified)'}`];
  $('#confirm-checks').innerHTML = checks.map(c => `<li>${esc(c)}</li>`).join('');
  const b = $('#confirm-acc'); b.textContent = r.accuracy; b.className = 'badge ' + r.accuracy;
  $('#confirm-warns').innerHTML = r.warnings.map(w => `<li>${esc(w)}</li>`).join('');
  setErr('confirm');
}
$('#btn-back').onclick = () => go('calib');
$('#btn-start').onclick = async () => {
  setErr('confirm'); const btn = $('#btn-start'); btn.disabled = true;
  try {
    const j = await postJSON('/api/jobs', {video_id: S.video.video_id, calibration: S.cal, settings: S.settings});
    S.jobId = j.job_id; $('#bar').style.width = '0'; $('#dash-rows').innerHTML = '<p class="muted dim">Waiting for objects…</p>'; setErr('run');
    $('#run-status').textContent = 'Queued…'; go('run'); poll();
  } catch (e) { setErr('confirm', e.message); }
  finally { btn.disabled = false; }
};

/* ------------------------------------------------------------ 4 · live run */
function poll() {
  clearInterval(S.timer);
  S.timer = setInterval(async () => {
    try {
      const j = await api('/api/jobs/' + S.jobId); renderJob(j);
      if (j.status === 'done') { clearInterval(S.timer); renderResults(j.result); go('results'); }
      if (j.status === 'error') { clearInterval(S.timer); setErr('run', j.error || 'Processing failed.'); }
    } catch (e) { clearInterval(S.timer); setErr('run', e.message); }
  }, 700);
}
function renderJob(j) {
  $('#bar').style.width = (j.progress * 100).toFixed(1) + '%';
  $('#run-status').textContent = j.status === 'queued' ? 'Queued – another analysis is running…' : j.status === 'encoding' ? 'Encoding the annotated video…' : `Detecting and tracking… ${(j.progress * 100).toFixed(0)}% (video time ${fmtT(j.time_s)})`;
  if (!j.live.length) return;
  $('#dash-rows').innerHTML = [...j.live].sort((a, b) => a.id - b.id).map(o =>
    `<div class="dr"><span>${ICON[o.cls] || ''}</span><span>${esc(cap(o.cls))} <b>#${o.id}</b></span><span>${o.kmh == null ? '<span class="dim">measuring…</span>' : o.kmh.toFixed(1) + ' km/h'}</span></div>`).join('');
}

/* ------------------------------------------------------------ 5 · results */
function renderResults(r) {
  const v = $('#res-video'); v.src = r.video_url + '?t=' + Date.now(); v.load();
  $('#dl-tracks').href = r.tracks_csv; $('#dl-traj').href = r.trajectories_csv;
  const gated = r.calibration.mode === 'known_distance';
  const rows = r.tracks.map(t => {
    const g = t.gate;
    const gate = !gated ? '' : g ? `Line ${g.order[0]} crossed → ${fmtT(g.order[0] === 'A' ? g.t_a : g.t_b)}<br>Line ${g.order[2]} crossed → ${fmtT(g.order[2] === 'A' ? g.t_a : g.t_b)}<br>${g.distance_m} m in ${g.time_s} s` : 'did not cross both lines';
    return `<tr><td>${ICON[t.cls] || ''} ${esc(cap(t.cls))}</td><td>#${t.id}</td><td><b>${t.speed_kmh == null ? '—' : t.speed_kmh.toFixed(1) + ' km/h'}</b><div class="muted" style="font-size:.78rem">${esc(t.basis)}</div></td>`
      + `<td>${t.avg_kmh == null ? '—' : t.avg_kmh.toFixed(1)}</td><td>${t.peak_kmh == null ? '—' : t.peak_kmh.toFixed(1)}</td><td>${fmtT(t.first_s)} – ${fmtT(t.last_s)}</td>${gated ? `<td class="gate">${gate}</td>` : ''}</tr>`;
  }).join('');
  $('#res-table').innerHTML = `<tr><th>Object</th><th>ID</th><th>Speed</th><th>Avg km/h</th><th>Peak km/h</th><th>Visible</th>${gated ? '<th>Line crossings</th>' : ''}</tr>` + (rows || '<tr><td colspan="7">No tracked objects. Try a lower confidence or a higher detection resolution.</td></tr>');
}
const reset = () => { clearInterval(S.timer); S.video = null; setErr('upload'); go('upload'); };
$('#btn-new').onclick = reset; $('#btn-new2').onclick = reset;
go('upload');
