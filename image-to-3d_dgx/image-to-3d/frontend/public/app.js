import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';

const $ = s => document.querySelector(s);
async function api(path, opt) {
  const r = await fetch('/api' + path, opt);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m } catch {} throw new Error(m) }
  return r.json();
}

/* ---------- viewer ---------- */
const stage = $('#stage');
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.NoToneMapping;           // keep source colours un-shifted
stage.prepend(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x12151a);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.add(new THREE.HemisphereLight(0xffffff, 0x777777, 0.8));
const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(3, 6, 5); scene.add(sun);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.screenSpacePanning = true;
const grid = new THREE.GridHelper(40, 80, 0x3a4250, 0x252b35); scene.add(grid);
let model = null, dist = 6;
function resize() { const w = stage.clientWidth, h = stage.clientHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(stage); resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

function setView(v) {
  const d = {front: [0, 0, 1], back: [0, 0, -1], left: [-1, 0, 0], right: [1, 0, 0], top: [0, 1, .001], bottom: [0, -1, .001]}[v];
  const t = controls.target;
  camera.position.set(t.x + d[0] * dist, t.y + d[1] * dist, t.z + d[2] * dist); controls.update();
}
function loadGLB(url) {
  return new Promise((res, rej) => new GLTFLoader().load(url, g => {
    if (model) scene.remove(model);
    model = g.scene;
    model.traverse(o => { if (o.material) { o.material.envMapIntensity = .7; o.material.wireframe = $('#wire').checked; } });
    scene.add(model);
    const b = new THREE.Box3().setFromObject(model), s = b.getSize(new THREE.Vector3());
    controls.target.copy(b.getCenter(new THREE.Vector3()));
    dist = Math.max(s.length(), .5) * 1.3; camera.near = dist / 200; camera.far = dist * 200; camera.updateProjectionMatrix();
    grid.position.y = b.min.y; setView('front'); res();
  }, undefined, rej));
}
document.querySelectorAll('#views button').forEach(b => b.onclick = () => setView(b.dataset.v));
$('#wire').onchange = e => model?.traverse(o => o.material && (o.material.wireframe = e.target.checked));
$('#spin').onchange = e => controls.autoRotate = e.target.checked;
$('#grid').onchange = e => grid.visible = e.target.checked;

/* ---------- upload + detect ---------- */
let det = null;
api('/health').then(h => $('#health').textContent =
  `${h.device} · SMPL-X ${h.smplx ? 'ready' : 'weights missing'} · TRELLIS ${h.trellis ? '✓' : '✗'} · Hunyuan ${h.hunyuan ? '✓' : '✗'}`).catch(() => {});
$('#drop').onclick = () => $('#file').click();
$('#drop').ondragover = e => { e.preventDefault(); $('#drop').classList.add('on') };
$('#drop').ondragleave = () => $('#drop').classList.remove('on');
$('#drop').ondrop = e => { e.preventDefault(); upload(e.dataTransfer.files[0]); };
$('#file').onchange = e => upload(e.target.files[0]);

async function upload(f) {
  if (!f) return;
  $('#drop').firstChild.textContent = 'Detecting objects…';
  const fd = new FormData(); fd.append('file', f);
  try { det = await api('/detect', {method: 'POST', body: fd}); } catch (e) { $('#drop').firstChild.textContent = 'Error: ' + e.message; return; }
  $('#drop').firstChild.textContent = 'Drop or click to choose another image';
  $('#detSec').hidden = $('#genSec').hidden = false;
  const img = $('#img'); img.onload = drawBoxes; img.src = det.image_url;
  const hasReal = det.objects.some(o => o.id !== 'whole');
  $('#objs').innerHTML = det.objects.map(o => `<label class="o"><input type="checkbox" value="${o.id}" ${o.id === 'whole' && hasReal ? '' : 'checked'}>
    <img src="${o.crop_url}"><span>${o.label} <small>${Math.round(o.conf * 100)}%</small>${o.keypoints ? ' · pose ✓' : ''}</span></label>`).join('');
}
function drawBoxes() {
  const c = $('#ov'), img = $('#img'); c.width = img.naturalWidth; c.height = img.naturalHeight;
  const g = c.getContext('2d'); g.lineWidth = Math.max(2, c.width / 300); g.font = `${Math.max(14, c.width / 50)}px sans-serif`;
  det.objects.filter(o => o.id !== 'whole').forEach((o, i) => {
    g.strokeStyle = g.fillStyle = `hsl(${i * 67 % 360} 90% 60%)`;
    g.strokeRect(o.bbox[0], o.bbox[1], o.bbox[2] - o.bbox[0], o.bbox[3] - o.bbox[1]); g.fillText(o.label, o.bbox[0] + 4, o.bbox[1] + 18);
  });
}

/* ---------- generate ---------- */
$('#go').onclick = async () => {
  const ids = [...document.querySelectorAll('#objs input:checked')].map(i => i.value);
  if (!ids.length) return alert('Select at least one object');
  $('#go').disabled = true; $('#log').textContent = 'queued…';
  try {
    const {job_id} = await api('/generate', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image_id: det.image_id, object_ids: ids, engine: $('#engine').value, person_mode: $('#pmode').value})});
    for (;;) {
      await new Promise(r => setTimeout(r, 1500));
      const j = await api('/jobs/' + job_id); $('#log').textContent = j.log.join('\n');
      if (j.status === 'error') throw new Error(j.error);
      if (j.status === 'done') { await show(job_id, j); break; }
    }
  } catch (e) { $('#log').textContent += '\nERROR: ' + e.message; }
  $('#go').disabled = false;
};
async function show(job, j) {
  await loadGLB(j.scene_url);
  $('#dl').hidden = false;
  $('#dlglb').href = `/api/jobs/${job}/download?fmt=glb`; $('#dlobj').href = `/api/jobs/${job}/download?fmt=obj`;
  $('#parts').innerHTML = j.parts.map(p => `<label><input type="checkbox" data-id="${p.id}" checked> ${p.label}</label>`).join('');
  $('#parts').querySelectorAll('input').forEach(cb => cb.onchange = () =>
    model.traverse(o => { if (o.name.startsWith(cb.dataset.id + '_')) o.visible = cb.checked; }));
}
