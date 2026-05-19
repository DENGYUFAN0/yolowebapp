/**
 * YOLOv5 실시간 객체 탐지 웹앱
 * TensorFlow.js + COCO-SSD 모델 사용
 * (YOLOv5와 동일한 COCO 80개 클래스 지원)
 */

'use strict';

// ── COCO 클래스 목록 (YOLOv5 동일) ──────────────────────────────
const COCO_CLASSES = [
  'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
  'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
  'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
  'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
  'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
  'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
  'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
  'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
  'remote','keyboard','cell phone','microwave','oven','toaster','sink',
  'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
  'toothbrush'
];

// 클래스별 색상 팔레트
const CLASS_COLORS = {};
const COLOR_PALETTE = [
  '#6366f1','#22d3ee','#10b981','#f59e0b','#ef4444','#ec4899','#8b5cf6',
  '#06b6d4','#84cc16','#f97316','#14b8a6','#e879f9','#3b82f6','#a3e635',
  '#fb923c','#34d399','#a78bfa','#60a5fa','#fbbf24','#f472b6'
];

COCO_CLASSES.forEach((cls, i) => {
  CLASS_COLORS[cls] = COLOR_PALETTE[i % COLOR_PALETTE.length];
});

// ── 상태 변수 ─────────────────────────────────────────────────
let model = null;
let videoStream = null;
let isDetecting = false;
let animFrameId = null;
let frameCount = 0;
let frameSkip = 1;
let minConfidence = 0.5;
let maxDetections = 20;
let showBBox = true;
let showLabel = true;
let showConfDisplay = true;
let facingMode = 'environment';

// FPS 계산
let lastFpsTime = performance.now();
let fpsFrames = 0;
let currentFps = 0;

// ── DOM 요소 참조 ──────────────────────────────────────────────
const $id = id => document.getElementById(id);

const dom = {
  loadingScreen: $id('loading-screen'),
  loadingBar: $id('loading-bar'),
  loadingProgress: $id('loading-progress'),
  app: $id('app'),
  video: $id('video'),
  canvas: $id('canvas'),
  cameraPlaceholder: $id('camera-placeholder'),
  detectionOverlay: $id('detection-overlay'),
  detectionBadge: $id('detection-badge'),
  detectionStatusText: $id('detection-status-text'),
  fpsBadge: $id('fps-badge'),
  flipBtn: $id('btn-flip'),
  btnStart: $id('btn-start'),
  btnStop: $id('btn-stop'),
  btnCapture: $id('btn-capture'),
  statTotal: $id('stat-total'),
  statFps: $id('stat-fps'),
  statConf: $id('stat-confidence'),
  statTime: $id('stat-time'),
  resultList: $id('result-list'),
  resultCount: $id('result-count'),
  settingsPanel: $id('settings-panel'),
  settingsBackdrop: $id('settings-backdrop'),
  infoModal: $id('info-modal'),
  infoBackdrop: $id('info-backdrop'),
  captureFlash: $id('capture-flash'),
  toastContainer: $id('toast-container'),
  sliderConf: $id('slider-confidence'),
  valConf: $id('val-confidence'),
  sliderFrameSkip: $id('slider-frame-skip'),
  valFrameSkip: $id('val-frame-skip'),
  sliderMaxDet: $id('slider-max-det'),
  valMaxDet: $id('val-max-det'),
  toggleBBox: $id('toggle-bbox'),
  toggleLabel: $id('toggle-label'),
  toggleConfDisplay: $id('toggle-conf-display'),
  selectCamera: $id('select-camera'),
  classChips: $id('class-chips'),
};

// ── 모델 로드 ──────────────────────────────────────────────────
async function loadModel() {
  setLoadingProgress('TensorFlow.js 초기화 중...', 15);
  await tf.ready();

  setLoadingProgress('AI 모델 다운로드 중... (첫 실행 시 시간이 걸릴 수 있습니다)', 35);

  try {
    model = await cocoSsd.load({
      base: 'mobilenet_v2'  // 모바일 최적화 모델
    });
    setLoadingProgress('모델 워밍업 중...', 80);

    // 워밍업 추론 (첫 추론이 느리지 않도록)
    const dummyCanvas = document.createElement('canvas');
    dummyCanvas.width = 320;
    dummyCanvas.height = 240;
    const dummyCtx = dummyCanvas.getContext('2d');
    dummyCtx.fillRect(0, 0, 320, 240);
    await model.detect(dummyCanvas, 1);

    setLoadingProgress('준비 완료!', 100);
    await sleep(400);
    showApp();
  } catch (err) {
    console.error('모델 로드 실패:', err);
    setLoadingProgress('❌ 모델 로드 실패. 페이지를 새로고침하세요.', 0);
  }
}

function setLoadingProgress(text, pct) {
  dom.loadingProgress.textContent = text;
  dom.loadingBar.style.width = pct + '%';
}

function showApp() {
  dom.loadingScreen.style.opacity = '0';
  dom.loadingScreen.style.transition = 'opacity 0.4s';
  setTimeout(() => {
    dom.loadingScreen.classList.add('hidden');
    dom.app.classList.remove('hidden');
    initClassChips();
  }, 400);
}

// ── 카메라 시작/정지 ───────────────────────────────────────────
async function startCamera() {
  try {
    if (videoStream) stopCamera();

    showToast('카메라 접근 요청 중...', 'info');

    const constraints = {
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1280, max: 1920 },
        height: { ideal: 720, max: 1080 },
      },
      audio: false
    };

    videoStream = await navigator.mediaDevices.getUserMedia(constraints);
    dom.video.srcObject = videoStream;

    await new Promise(resolve => {
      dom.video.onloadedmetadata = () => {
        dom.video.play();
        resolve();
      };
    });

    // 캔버스 크기 맞추기
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    dom.cameraPlaceholder.classList.add('hidden');
    dom.flipBtn.classList.remove('hidden');
    dom.btnCapture.classList.remove('hidden');

    return true;
  } catch (err) {
    console.error('카메라 오류:', err);
    if (err.name === 'NotAllowedError') {
      showToast('카메라 권한을 허용해주세요', 'error');
    } else if (err.name === 'NotFoundError') {
      showToast('카메라를 찾을 수 없습니다', 'error');
    } else {
      showToast('카메라 오류: ' + err.message, 'error');
    }
    return false;
  }
}

function stopCamera() {
  if (videoStream) {
    videoStream.getTracks().forEach(t => t.stop());
    videoStream = null;
  }
  dom.video.srcObject = null;
}

function resizeCanvas() {
  const container = dom.canvas.parentElement;
  dom.canvas.width = container.clientWidth;
  dom.canvas.height = container.clientHeight;
}

// ── 탐지 루프 ──────────────────────────────────────────────────
async function startDetection() {
  if (!model) { showToast('모델이 아직 로드되지 않았습니다', 'error'); return; }

  const ok = await startCamera();
  if (!ok) return;

  isDetecting = true;
  frameCount = 0;

  dom.btnStart.classList.add('hidden');
  dom.btnStop.classList.remove('hidden');
  dom.detectionOverlay.style.display = 'flex';

  showToast('객체 탐지 시작!', 'success');
  detectLoop();
}

function stopDetection() {
  isDetecting = false;
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
  stopCamera();
  clearCanvas();

  dom.btnStop.classList.add('hidden');
  dom.btnStart.classList.remove('hidden');
  dom.btnCapture.classList.add('hidden');
  dom.flipBtn.classList.add('hidden');
  dom.cameraPlaceholder.classList.remove('hidden');
  dom.detectionOverlay.style.display = 'none';

  // 통계 초기화
  updateStats(0, 0, 0, 0);
  renderResultList([]);
  showToast('탐지가 정지되었습니다', 'info');
}

async function detectLoop() {
  if (!isDetecting) return;

  frameCount++;

  if (frameCount % frameSkip === 0) {
    const t0 = performance.now();

    try {
      if (dom.video.readyState >= 2 && dom.video.videoWidth > 0) {
        const predictions = await model.detect(dom.video, maxDetections);
        const filtered = predictions.filter(p => p.score >= minConfidence);

        const elapsed = Math.round(performance.now() - t0);

        // FPS 계산
        fpsFrames++;
        const now = performance.now();
        if (now - lastFpsTime >= 1000) {
          currentFps = Math.round(fpsFrames * 1000 / (now - lastFpsTime));
          fpsFrames = 0;
          lastFpsTime = now;
        }

        // 평균 신뢰도
        const avgConf = filtered.length > 0
          ? Math.round(filtered.reduce((s, p) => s + p.score, 0) / filtered.length * 100)
          : 0;

        // 캔버스 그리기
        drawDetections(filtered);

        // UI 업데이트
        updateStats(filtered.length, currentFps, avgConf, elapsed);
        renderResultList(filtered);
        dom.fpsBadge.textContent = currentFps + ' FPS';
      }
    } catch (err) {
      console.warn('탐지 오류:', err);
    }
  }

  animFrameId = requestAnimationFrame(detectLoop);
}

// ── 캔버스 그리기 ─────────────────────────────────────────────
function drawDetections(predictions) {
  const canvas = dom.canvas;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  if (!predictions || predictions.length === 0) return;

  // 비디오 실제 크기 대비 캔버스 크기 비율
  const vw = dom.video.videoWidth;
  const vh = dom.video.videoHeight;
  if (!vw || !vh) return;

  // object-fit: cover 보정
  const videoAspect = vw / vh;
  const canvasAspect = w / h;

  let scaleX, scaleY, offsetX, offsetY;

  if (canvasAspect > videoAspect) {
    // 캔버스가 더 넓음 → 좌우로 잘림
    scaleX = w / vw;
    scaleY = w / videoAspect / vh;
    offsetX = 0;
    offsetY = (h - vh * scaleY) / 2;
  } else {
    // 캔버스가 더 높음 → 상하 잘림
    scaleX = h * videoAspect / vw;
    scaleY = h / vh;
    offsetX = (w - vw * scaleX) / 2;
    offsetY = 0;
  }

  predictions.forEach(pred => {
    const [x, y, bw, bh] = pred.bbox;
    const color = CLASS_COLORS[pred.class] || '#6366f1';

    const rx = x * scaleX + offsetX;
    const ry = y * scaleY + offsetY;
    const rw = bw * scaleX;
    const rh = bh * scaleY;

    if (!showBBox && !showLabel) return;

    if (showBBox) {
      // 바운딩 박스 배경 글로우
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.shadowBlur = 0;

      // 모서리 강조
      const cornerSize = Math.min(rw, rh) * 0.15;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 3;
      drawCorners(ctx, rx, ry, rw, rh, cornerSize);
    }

    if (showLabel) {
      const confText = showConfDisplay ? ` ${Math.round(pred.score * 100)}%` : '';
      const labelText = pred.class.toUpperCase() + confText;
      const fontSize = Math.max(11, Math.min(14, rw / 8));
      ctx.font = `bold ${fontSize}px Inter, sans-serif`;
      const textW = ctx.measureText(labelText).width + 14;
      const labelH = fontSize + 12;

      // 라벨 배경
      ctx.fillStyle = color;
      const labelY = ry - labelH < 0 ? ry + 2 : ry - labelH;
      roundRect(ctx, rx - 1, labelY, textW, labelH, 4);
      ctx.fill();

      // 라벨 텍스트
      ctx.fillStyle = '#fff';
      ctx.fillText(labelText, rx + 6, labelY + fontSize + 2);
    }
  });
}

function drawCorners(ctx, x, y, w, h, size) {
  ctx.beginPath();
  // 좌상단
  ctx.moveTo(x, y + size); ctx.lineTo(x, y); ctx.lineTo(x + size, y);
  // 우상단
  ctx.moveTo(x + w - size, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + size);
  // 우하단
  ctx.moveTo(x + w, y + h - size); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - size, y + h);
  // 좌하단
  ctx.moveTo(x + size, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - size);
  ctx.stroke();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function clearCanvas() {
  const ctx = dom.canvas.getContext('2d');
  ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);
}

// ── UI 업데이트 ───────────────────────────────────────────────
function updateStats(total, fps, conf, time) {
  dom.statTotal.textContent = total;
  dom.statFps.textContent = fps;
  dom.statConf.textContent = conf + '%';
  dom.statTime.textContent = time + 'ms';
}

function renderResultList(predictions) {
  const count = predictions.length;
  dom.resultCount.textContent = count + '개';

  if (count === 0) {
    dom.resultList.innerHTML = `
      <div class="result-empty">
        <i class="fas fa-search"></i>
        <p>탐지된 객체가 없습니다</p>
      </div>`;
    return;
  }

  // 신뢰도 내림차순 정렬
  const sorted = [...predictions].sort((a, b) => b.score - a.score);

  // 중복 클래스 합산
  const grouped = {};
  sorted.forEach(p => {
    if (!grouped[p.class]) grouped[p.class] = { count: 0, maxScore: 0 };
    grouped[p.class].count++;
    grouped[p.class].maxScore = Math.max(grouped[p.class].maxScore, p.score);
  });

  const items = Object.entries(grouped)
    .sort((a, b) => b[1].maxScore - a[1].maxScore)
    .map(([cls, info]) => {
      const color = CLASS_COLORS[cls] || '#6366f1';
      const conf = Math.round(info.maxScore * 100);
      const countBadge = info.count > 1 ? ` <span style="font-size:11px;color:var(--text-dim)">×${info.count}</span>` : '';
      return `
        <div class="result-item">
          <div class="result-color" style="background:${color}"></div>
          <span class="result-name">${cls}${countBadge}</span>
          <span class="result-conf" style="color:${color};background:${color}22">${conf}%</span>
        </div>`;
    }).join('');

  dom.resultList.innerHTML = items;
}

// ── 캡처 ─────────────────────────────────────────────────────
function captureFrame() {
  if (!isDetecting) return;

  // 플래시 효과
  dom.captureFlash.classList.add('active');
  setTimeout(() => dom.captureFlash.classList.remove('active'), 150);

  // 최종 이미지 합성
  const offCanvas = document.createElement('canvas');
  offCanvas.width = dom.video.videoWidth;
  offCanvas.height = dom.video.videoHeight;
  const ctx = offCanvas.getContext('2d');

  // 비디오 프레임
  ctx.drawImage(dom.video, 0, 0);

  // 캔버스 오버레이 (비율 맞춤)
  ctx.drawImage(dom.canvas, 0, 0, offCanvas.width, offCanvas.height);

  // 다운로드
  offCanvas.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `detection_${Date.now()}.jpg`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('📸 이미지가 저장되었습니다', 'success');
  }, 'image/jpeg', 0.92);
}

// ── 카메라 전환 ───────────────────────────────────────────────
async function flipCamera() {
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  dom.selectCamera.value = facingMode;
  if (isDetecting) {
    stopCamera();
    const ok = await startCamera();
    if (!ok) {
      facingMode = facingMode === 'environment' ? 'user' : 'environment';
      await startCamera();
    }
  }
  showToast(facingMode === 'environment' ? '후면 카메라로 전환' : '전면 카메라로 전환', 'info');
}

// ── 설정 패널 ─────────────────────────────────────────────────
function openSettings() {
  dom.settingsPanel.classList.add('open');
}

function closeSettings() {
  dom.settingsPanel.classList.remove('open');
}

// ── 정보 모달 ─────────────────────────────────────────────────
function openInfo() {
  dom.infoModal.classList.remove('hidden');
}

function closeInfo() {
  dom.infoModal.classList.add('hidden');
}

// ── 클래스 칩 생성 ────────────────────────────────────────────
function initClassChips() {
  const colors = Object.values(COLOR_PALETTE);
  dom.classChips.innerHTML = COCO_CLASSES.map((cls, i) => `
    <span class="class-chip" style="background:${colors[i % colors.length]}">
      ${cls}
    </span>`
  ).join('');
}

// ── 토스트 알림 ───────────────────────────────────────────────
function showToast(msg, type = 'info', duration = 2500) {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  dom.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.transition = 'opacity 0.3s';
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── 유틸리티 ─────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── 이벤트 리스너 ────────────────────────────────────────────
function initEvents() {
  // 시작/정지
  dom.btnStart.addEventListener('click', startDetection);
  dom.btnStop.addEventListener('click', stopDetection);
  dom.btnCapture.addEventListener('click', captureFrame);
  dom.flipBtn.addEventListener('click', flipCamera);

  // 설정
  $id('btn-settings').addEventListener('click', openSettings);
  $id('btn-close-settings').addEventListener('click', closeSettings);
  dom.settingsBackdrop.addEventListener('click', closeSettings);

  // 정보
  $id('btn-info').addEventListener('click', openInfo);
  $id('btn-close-info').addEventListener('click', closeInfo);
  dom.infoBackdrop.addEventListener('click', closeInfo);

  // 슬라이더 - 신뢰도
  dom.sliderConf.addEventListener('input', e => {
    minConfidence = parseInt(e.target.value) / 100;
    dom.valConf.textContent = e.target.value + '%';
  });

  // 슬라이더 - 프레임 스킵
  dom.sliderFrameSkip.addEventListener('input', e => {
    frameSkip = parseInt(e.target.value);
    dom.valFrameSkip.textContent = e.target.value;
  });

  // 슬라이더 - 최대 탐지 수
  dom.sliderMaxDet.addEventListener('input', e => {
    maxDetections = parseInt(e.target.value);
    dom.valMaxDet.textContent = e.target.value;
  });

  // 토글
  dom.toggleBBox.addEventListener('change', e => { showBBox = e.target.checked; });
  dom.toggleLabel.addEventListener('change', e => { showLabel = e.target.checked; });
  dom.toggleConfDisplay.addEventListener('change', e => { showConfDisplay = e.target.checked; });

  // 카메라 선택
  dom.selectCamera.addEventListener('change', async e => {
    facingMode = e.target.value;
    if (isDetecting) {
      stopCamera();
      await startCamera();
    }
  });

  // 비디오 크기 변경 시 캔버스 리사이즈
  dom.video.addEventListener('loadeddata', resizeCanvas);

  // 화면 방향 변경
  window.addEventListener('orientationchange', () => {
    setTimeout(resizeCanvas, 300);
  });

  // PWA 설치 배너 처리
  let deferredPrompt;
  window.addEventListener('beforeinstallprompt', e => {
    e.preventDefault();
    deferredPrompt = e;
    setTimeout(() => {
      showToast('📱 홈 화면에 추가하면 앱처럼 사용할 수 있습니다!', 'info', 4000);
    }, 3000);
  });
}

// ── 초기화 ───────────────────────────────────────────────────
async function init() {
  initEvents();

  // 브라우저 지원 확인
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setLoadingProgress('❌ 이 브라우저는 카메라를 지원하지 않습니다.\nChrome 또는 Safari를 사용해주세요.', 0);
    return;
  }

  await loadModel();

  // Service Worker 등록 (PWA)
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch(err => {
      console.warn('SW 등록 실패:', err);
    });
  }
}

init();
