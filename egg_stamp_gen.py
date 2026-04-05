import numpy as np
import cv2
import scipy.ndimage
from skimage.measure import marching_cubes
import trimesh
import io
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response

app = FastAPI()

HTML_CONTENT = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>🥚 EggStamp Gen</title>
<script src="https://cdn.tailwindcss.com"></script>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #080c14; color: #f1f5f9; font-family: 'Segoe UI', sans-serif; overflow: hidden; }

#panel {
  position: fixed; top: 0; left: 0; width: 300px; height: 100vh;
  background: #0d1220;
  border-right: 1px solid #1a2640;
  overflow-y: auto; z-index: 10;
  padding: 20px 16px 40px;
}
#canvas-wrap {
  position: fixed; top: 0; left: 300px; right: 0; bottom: 0;
}

.logo { font-size: 1.6rem; font-weight: 900; letter-spacing: -0.5px; margin-bottom: 2px; }
.logo span { color: #f59e0b; }
.sub { color: #3d5270; font-size: 0.68rem; font-family: monospace; margin-bottom: 18px; }

.sec-title {
  font-size: 0.65rem; font-weight: 800; letter-spacing: 2px; text-transform: uppercase;
  color: #f59e0b; padding-bottom: 5px; border-bottom: 1px solid #1a2640;
  margin: 18px 0 10px;
}
.row { margin-bottom: 10px; }
.lbl { display: flex; justify-content: space-between; font-size: 0.74rem; color: #7a94b8; margin-bottom: 3px; font-weight: 600; }
.val { color: #fff; font-weight: 700; }
input[type=range] { width: 100%; accent-color: #f59e0b; cursor: pointer; }

.upload-zone {
  border: 2px dashed #1e3a5f; border-radius: 10px; padding: 14px 10px;
  text-align: center; cursor: pointer; transition: all 0.2s;
  background: rgba(30,58,95,0.1);
}
.upload-zone:hover { border-color: #f59e0b; }
.upload-zone input { display: none; }
#preview-img { max-width: 100%; max-height: 100px; border-radius: 6px; margin-top: 8px; display: none; object-fit: contain; }

.toggle-group { display: flex; gap: 6px; }
.toggle-btn {
  flex: 1; padding: 8px 4px; border-radius: 7px; font-size: 0.76rem; font-weight: 700;
  border: 2px solid #1a2640; cursor: pointer; transition: all 0.2s;
  background: transparent; color: #3d5270; text-align: center;
}
.toggle-btn.active { border-color: #f59e0b; color: #f59e0b; background: rgba(245,158,11,0.07); }

.btn-gen {
  width: 100%; margin-top: 18px; padding: 14px;
  background: linear-gradient(135deg, #d97706, #f59e0b);
  color: #000; font-weight: 900; font-size: 0.95rem;
  border: none; border-radius: 10px; cursor: pointer;
  box-shadow: 0 0 20px rgba(245,158,11,0.25);
  transition: all 0.2s; letter-spacing: 0.5px;
}
.btn-gen:hover { transform: translateY(-1px); box-shadow: 0 0 30px rgba(245,158,11,0.4); }
.btn-gen:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
.btn-dl {
  width: 100%; margin-top: 8px; padding: 12px;
  background: linear-gradient(135deg, #059669, #10b981);
  color: #fff; font-weight: 900; font-size: 0.88rem;
  border: none; border-radius: 10px; cursor: pointer;
  box-shadow: 0 0 16px rgba(16,185,129,0.2);
  transition: all 0.2s; display: none;
}
.btn-dl:hover { transform: translateY(-1px); }

#status {
  margin-top: 10px; padding: 10px 14px; border-radius: 8px;
  font-size: 0.78rem; font-weight: 600; display: none; line-height: 1.4;
}
#status.loading { background: rgba(245,158,11,0.08); color: #f59e0b; border: 1px solid rgba(245,158,11,0.15); }
#status.error { background: rgba(239,68,68,0.08); color: #ef4444; border: 1px solid rgba(239,68,68,0.15); }
#status.ok { background: rgba(16,185,129,0.08); color: #10b981; border: 1px solid rgba(16,185,129,0.15); }
.hint { font-size: 0.66rem; color: #3d5270; margin-top: 3px; line-height: 1.4; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: #f59e0b33; border-radius: 3px; }
</style>
</head>
<body>
<div id="panel">
  <div class="logo">🥚 Egg<span>Stamp</span></div>
  <div class="sub">v2.0 // SDF Egg Generator</div>

  <div class="sec-title">Картинка</div>
  <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
    <input type="file" id="fileInput" accept="image/png,image/jpeg,image/webp">
    <div id="uploadHint">
      <div style="font-size:1.6rem">📁</div>
      <div style="font-size:0.76rem;color:#3d5270;margin-top:3px">Нажми или перетащи PNG</div>
      <div style="font-size:0.65rem;color:#1e3a5f;margin-top:2px">Непрозрачное = узор на яйце</div>
    </div>
    <img id="preview-img" src="" alt="preview">
  </div>

  <div class="sec-title">Режим рельефа</div>
  <div class="toggle-group">
    <button class="toggle-btn active" id="btn-emboss" onclick="setMode('emboss')">⬆️ Выпуклый<br><small style="opacity:0.6">emboss</small></button>
    <button class="toggle-btn" id="btn-deboss" onclick="setMode('deboss')">⬇️ Вдавленный<br><small style="opacity:0.6">deboss</small></button>
  </div>
  <input type="hidden" id="mode" value="emboss">

  <div class="sec-title">Форма яйца</div>
  <div class="row">
    <div class="lbl"><span>Ширина (мм)</span><span class="val" id="v-ew">50</span></div>
    <input type="range" id="egg_w" min="20" max="100" value="50" oninput="upd('v-ew',this.value)">
  </div>
  <div class="row">
    <div class="lbl"><span>Высота (мм)</span><span class="val" id="v-eh">70</span></div>
    <input type="range" id="egg_h" min="30" max="150" value="70" oninput="upd('v-eh',this.value)">
  </div>
  <div class="row">
    <div class="lbl"><span>Асимметрия верха</span><span class="val" id="v-ea">0.25</span></div>
    <input type="range" id="egg_a" min="0.0" max="0.55" step="0.01" value="0.25" oninput="upd('v-ea',this.value)">
  </div>

  <div class="sec-title">Рельеф</div>
  <div class="row">
    <div class="lbl"><span>Высота рельефа (мм)</span><span class="val" id="v-rd">1.2</span></div>
    <input type="range" id="relief_depth" min="0.2" max="4.0" step="0.1" value="1.2" oninput="upd('v-rd',this.value)">
  </div>
  <div class="row">
    <div class="lbl"><span>Толщина стенки (мм)</span><span class="val" id="v-wt">3.0</span></div>
    <input type="range" id="wall_t" min="1.0" max="8.0" step="0.1" value="3.0" oninput="upd('v-wt',this.value)">
  </div>
  <div class="row">
    <div class="lbl"><span>Сглаживание краёв</span><span class="val" id="v-sm">1.2</span></div>
    <input type="range" id="smooth" min="0.0" max="5.0" step="0.1" value="1.2" oninput="upd('v-sm',this.value)">
  </div>

  <div class="sec-title">Узор на яйце</div>
  <div class="row">
    <div class="lbl"><span>Масштаб (%)</span><span class="val" id="v-ps">80</span></div>
    <input type="range" id="pat_scale" min="10" max="130" value="80" oninput="upd('v-ps',this.value)">
  </div>
  <div class="row">
    <div class="lbl"><span>Вертикальный сдвиг (%)</span><span class="val" id="v-pv">0</span></div>
    <input type="range" id="pat_vert" min="-50" max="50" value="0" oninput="upd('v-pv',this.value)">
  </div>
  <div class="row">
    <div class="lbl"><span>Поворот (°)</span><span class="val" id="v-pr">0</span></div>
    <input type="range" id="pat_rot" min="0" max="360" value="0" oninput="upd('v-pr',this.value)">
  </div>

  <div class="sec-title">Качество</div>
  <div class="row">
    <div class="lbl"><span>Разрешение сетки</span><span class="val" id="v-res">100</span></div>
    <input type="range" id="res" min="60" max="1000" step="10" value="100" oninput="upd('v-res',this.value)">
    <div class="hint">100 — быстро ~10с. 200+ — медленно, максимум детали.</div>
  </div>

  <button class="btn-gen" id="genBtn" onclick="generate()">🚀 СГЕНЕРИРОВАТЬ STL</button>
  <button class="btn-dl" id="dlBtn" onclick="download()">📥 СКАЧАТЬ STL</button>
  <div id="status"></div>
</div>

<div id="canvas-wrap"></div>

<script type="module">
import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const wrap = document.getElementById('canvas-wrap');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x080c14);

// Освещение
scene.add(new THREE.AmbientLight(0x8090b0, 0.8));

const key = new THREE.DirectionalLight(0xfff8f0, 3.0);
key.position.set(60, 120, 80);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.near = 1;
key.shadow.camera.far = 500;
key.shadow.camera.left = -120;
key.shadow.camera.right = 120;
key.shadow.camera.top = 120;
key.shadow.camera.bottom = -120;
scene.add(key);

const fill = new THREE.DirectionalLight(0x4060a0, 0.8);
fill.position.set(-80, 40, -60);
scene.add(fill);

const W = () => window.innerWidth - 300;
const H = () => window.innerHeight;

const camera = new THREE.PerspectiveCamera(45, W() / H(), 0.5, 2000);
camera.position.set(0, 50, 160);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
wrap.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 35, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 20;
controls.maxDistance = 500;
controls.update();

// Пол
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(600, 600),
  new THREE.MeshStandardMaterial({ color: 0x0d1a2a, roughness: 0.95 })
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(300, 40, 0x1a3050, 0x0f2035);
grid.position.y = 0.05;
scene.add(grid);

// Placeholder egg wireframe
function makeEggGeo(Rx, Ry, asym, segs) {
  const geo = new THREE.SphereGeometry(1, segs, segs);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const scaleXZ = Math.max(1.0 - asym * y, 0.1);
    pos.setX(i, x * Rx * scaleXZ);
    pos.setY(i, y * Ry);
    pos.setZ(i, z * Rx * scaleXZ);
  }
  geo.computeVertexNormals();
  return geo;
}

let placeholder = null;
{
  const geo = makeEggGeo(25, 35, 0.25, 48);
  const mat = new THREE.MeshStandardMaterial({
    color: 0xd4c9b0, roughness: 0.4, metalness: 0.0,
    transparent: true, opacity: 0.15, side: THREE.DoubleSide
  });
  placeholder = new THREE.Mesh(geo, mat);
  placeholder.position.y = 35;
  placeholder.castShadow = true;
  scene.add(placeholder);

  const wf = new THREE.Mesh(
    geo.clone(),
    new THREE.MeshBasicMaterial({ color: 0x2a5080, wireframe: true, transparent: true, opacity: 0.5 })
  );
  placeholder.add(wf);
}

let eggMesh = null;
let blobUrl = null;

window.loadSTL = function(blob) {
  if (eggMesh) { scene.remove(eggMesh); eggMesh.geometry.dispose(); eggMesh.material.dispose(); eggMesh = null; }
  if (placeholder) { scene.remove(placeholder); placeholder = null; }
  if (blobUrl) URL.revokeObjectURL(blobUrl);

  blobUrl = URL.createObjectURL(blob);
  const loader = new STLLoader();
  loader.load(blobUrl, geo => {
    geo.computeVertexNormals();

    const box = new THREE.Box3().setFromBufferAttribute(geo.attributes.position);
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);

    eggMesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      color: 0xe8dcc8, roughness: 0.28, metalness: 0.04,
    }));
    eggMesh.castShadow = true;
    eggMesh.receiveShadow = true;
    // дно на Y=0
    eggMesh.position.set(-center.x, -box.min.y, -center.z);
    scene.add(eggMesh);

    // Подгонка камеры
    const H2 = size.y;
    const R2 = Math.max(size.x, size.z) / 2;
    const dist = Math.max(H2, R2 * 2) * 2.8;
    controls.target.set(0, H2 / 2, 0);
    camera.position.set(0, H2 * 0.5, dist);
    controls.update();
  });
};

window.getBlobUrl = () => blobUrl;

function resize() {
  camera.aspect = W() / H();
  camera.updateProjectionMatrix();
  renderer.setSize(W(), H());
}
resize();
window.addEventListener('resize', resize);

(function loop() {
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene, camera);
})();
</script>

<script>
let currentMode = 'emboss';

function upd(id, val) {
  const n = parseFloat(val);
  document.getElementById(id).textContent = Number.isInteger(n) ? n : n.toFixed(2);
}

function setMode(m) {
  currentMode = m;
  document.getElementById('btn-emboss').classList.toggle('active', m === 'emboss');
  document.getElementById('btn-deboss').classList.toggle('active', m === 'deboss');
}

const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('preview-img');
const uploadHint = document.getElementById('uploadHint');

fileInput.addEventListener('change', e => {
  const f = e.target.files[0]; if (!f) return;
  previewImg.src = URL.createObjectURL(f);
  previewImg.style.display = 'block';
  uploadHint.style.display = 'none';
});

const dropZone = document.querySelector('.upload-zone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = '#f59e0b'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = '#1e3a5f'; });
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.style.borderColor = '#1e3a5f';
  const f = e.dataTransfer.files[0]; if (!f) return;
  const dt = new DataTransfer(); dt.items.add(f); fileInput.files = dt.files;
  previewImg.src = URL.createObjectURL(f);
  previewImg.style.display = 'block'; uploadHint.style.display = 'none';
});

function setStatus(msg, type) {
  const s = document.getElementById('status');
  s.textContent = msg; s.className = type; s.style.display = msg ? 'block' : 'none';
}

async function generate() {
  const genBtn = document.getElementById('genBtn');
  const dlBtn = document.getElementById('dlBtn');
  genBtn.disabled = true; dlBtn.style.display = 'none';
  setStatus('⏳ Генерация яйца... (~10-40 сек)', 'loading');

  const fd = new FormData();
  const f = fileInput.files[0]; if (f) fd.append('file', f);
  fd.append('mode',         currentMode);
  fd.append('egg_w',        document.getElementById('egg_w').value);
  fd.append('egg_h',        document.getElementById('egg_h').value);
  fd.append('egg_a',        document.getElementById('egg_a').value);
  fd.append('relief_depth', document.getElementById('relief_depth').value);
  fd.append('wall_t',       document.getElementById('wall_t').value);
  fd.append('smooth',       document.getElementById('smooth').value);
  fd.append('pat_scale',    document.getElementById('pat_scale').value);
  fd.append('pat_vert',     document.getElementById('pat_vert').value);
  fd.append('pat_rot',      document.getElementById('pat_rot').value);
  fd.append('res',          document.getElementById('res').value);

  try {
    const r = await fetch('/generate', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(await r.text());
    window.loadSTL(await r.blob());
    setStatus('✅ Готово! Крути мышкой.', 'ok');
    dlBtn.style.display = 'block';
  } catch(e) {
    setStatus('❌ ' + e.message, 'error');
  } finally {
    genBtn.disabled = false;
  }
}

function download() {
  const url = window.getBlobUrl(); if (!url) return;
  Object.assign(document.createElement('a'), { href: url, download: 'egg_stamp.stl' }).click();
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTML_CONTENT


@app.post("/generate")
async def generate(
    file:         UploadFile = File(None),
    mode:         str   = Form("emboss"),
    egg_w:        float = Form(50.0),
    egg_h:        float = Form(70.0),
    egg_a:        float = Form(0.25),
    relief_depth: float = Form(1.2),
    wall_t:       float = Form(3.0),
    smooth:       float = Form(1.2),
    pat_scale:    float = Form(80.0),
    pat_vert:     float = Form(0.0),
    pat_rot:      float = Form(0.0),
    res:          int   = Form(100),
):
    # ── Маска ─────────────────────────────────────────────────────────
    if file and file.filename:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        img_bgra = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img_bgra is None:
            raise ValueError("Не удалось прочитать изображение")
            
        # Преобразуем цветную картинку в ЧБ, чтобы высота зависела от яркости цвета
        if img_bgra.ndim == 3 and img_bgra.shape[2] == 4:
            # Если есть альфа-канал: используем яркость умноженную на прозрачность
            gray = cv2.cvtColor(img_bgra[:, :, :3], cv2.COLOR_BGR2GRAY).astype(float) / 255.0
            alpha = img_bgra[:, :, 3].astype(float) / 255.0
            mask = gray * alpha
        elif img_bgra.ndim == 3:
            # Обычный RGB, просто берем яркость
            mask = cv2.cvtColor(img_bgra, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        else:
            # Если изначально одноканальное
            mask = img_bgra.astype(float) / 255.0
    else:
        # Демо: звезда
        sz = 256
        mask = np.zeros((sz, sz), float)
        cx = cy = sz // 2
        for iy in range(sz):
            for ix in range(sz):
                dx, dy = ix - cx, iy - cy
                r = np.sqrt(dx*dx + dy*dy)
                ang = np.arctan2(dy, dx)
                if r < (np.cos(5 * ang) * 0.35 + 0.65) * sz * 0.38:
                    mask[iy, ix] = 1.0

    if smooth > 0:
        mask = scipy.ndimage.gaussian_filter(mask, sigma=smooth)
    mask = np.clip(mask, 0, 1)
    img_rows, img_cols = mask.shape

    # ── Параметры яйца ─────────────────────────────────────────────────
    Rx = egg_w / 2.0
    Ry = egg_h / 2.0
    asym = float(egg_a)

    # ── Вокселная сетка — яйцо центрировано в (0,0,0) ─────────────────
    pad = Rx * 0.12 + relief_depth + 2.5
    xr = np.linspace(-(Rx + pad), Rx + pad, res)
    yr = np.linspace(-(Ry + pad), Ry + pad, res)
    zr = np.linspace(-(Rx + pad), Rx + pad, res)
    X, Y, Z = np.meshgrid(xr, yr, zr, indexing='ij')

    # ── Форма: асимметричный эллипсоид ────────────────────────────────
    # t = Y/Ry ∈ [-1,1]; +1=верх(острый), -1=низ(широкий)
    t = Y / Ry
    Rxz = Rx * np.maximum(1.0 - asym * t, 0.05)

    # Имплицитное поле: < 0 внутри яйца
    egg_f = (X / Rxz)**2 + (Y / Ry)**2 + (Z / Rxz)**2 - 1.0

    # ── Внутренняя оболочка (для толщины стенки) ──────────────────────
    Rxz_in = np.maximum(Rxz - wall_t, 0.3)
    Ry_in  = float(max(Ry - wall_t, 0.3))
    inner_f = (X / Rxz_in)**2 + (Y / Ry_in)**2 + (Z / Rxz_in)**2 - 1.0
    shell = np.maximum(egg_f, -inner_f)

    # ── UV-проекция: цилиндрическая с физическими размерами ───────────
    # Вычисляем угол так, чтобы 0 был строго спереди (X=0, Z>0), и угол рос вправо.
    # Это исправляет горизонтальное отзеркаливание.
    angle = np.arctan2(X, Z) 
    
    # Физические дистанции на поверхности по X и Y (в мм)
    arc_x = angle * Rx
    
    # Вертикальный сдвиг (в миллиметрах). pat_vert от -50 до 50%
    shift_y = (float(pat_vert) / 50.0) * (Ry * 0.5)
    arc_y = Y - shift_y

    # Вращение
    rot_rad = float(pat_rot) * np.pi / 180.0
    cos_r = np.cos(-rot_rad)
    sin_r = np.sin(-rot_rad)

    arc_x_rot = arc_x * cos_r - arc_y * sin_r
    arc_y_rot = arc_x * sin_r + arc_y * cos_r

    # Масштаб. При scale=100 физическая ширина картинки равна диаметру яйца (2*Rx).
    # Это гарантирует, что изображение не будет оборачиваться вокруг всего яйца 
    # и масштаб ~80 будет выглядеть адекватным как обычный штамп на передней части.
    scale_f = float(pat_scale) / 100.0
    if scale_f < 0.01:
        scale_f = 0.01
        
    img_width_mm = (2.0 * Rx) * scale_f
    ppm = img_cols / img_width_mm  # pixels per mm

    # Маппинг в координаты картинки (сохраняет идеальные пропорции)
    px_f = arc_x_rot * ppm + (img_cols / 2.0)
    py_f = -arc_y_rot * ppm + (img_rows / 2.0)

    pat_vol = scipy.ndimage.map_coordinates(
        mask, [py_f, px_f], order=1, mode='constant', cval=0.0
    )
    pat_vol = np.clip(pat_vol, 0, 1)

    # ── Рельеф — ТОЛЬКО вблизи внешней поверхности ────────────────────
    # egg_f ≈ 0 на поверхности, < 0 внутри, > 0 снаружи.
    # Применяем рельеф только в слое толщиной relief_depth*2 вокруг поверхности.
    
    # Толщина слоя в единицах поля (градиент ≈ 2/Rx)
    layer = relief_depth * (2.0 / Rx)
    
    # Вес: 1.0 на поверхности, плавно затухает к 0 за пределами слоя
    surface_weight = np.clip(1.0 - np.abs(egg_f) / layer, 0.0, 1.0)
    pat_val = pat_vol * surface_weight

    norm_scale = layer  # сдвиг поля = толщина слоя

    if mode == "emboss":
        # Двигаем внешнюю поверхность наружу там где маска
        egg_mod   = egg_f - pat_val * norm_scale
        shell_mod = np.maximum(egg_mod, -inner_f)
    else:
        # Вдавливаем внешнюю поверхность внутрь там где маска
        egg_mod   = egg_f + pat_val * norm_scale
        shell_mod = np.maximum(egg_mod, -inner_f)

    # Плоское дно: срезаем нижнюю часть яйца для устойчивости при 3D-печати
    flat_y = -Ry * 0.88
    bot_mask = Y - flat_y   # > 0 выше flat_y (это "не-дно"), < 0 ниже flat_y
    final_f = np.maximum(shell_mod, -bot_mask)

    # ── Marching Cubes ─────────────────────────────────────────────────
    spacing = (xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0])
    verts, faces, _, _ = marching_cubes(final_f, level=0.0, spacing=spacing)

    # Перевод в мировые координаты
    verts[:, 0] += xr[0]
    verts[:, 1] += yr[0]
    verts[:, 2] += zr[0]

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    buf = io.BytesIO()
    mesh.export(file_obj=buf, file_type='stl')
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=egg_stamp.stl"}
    )


if __name__ == "__main__":
    print("🥚 EggStamp Gen → http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)