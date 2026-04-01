/**
 * waveform.js — Ring-buffer waveform renderer.
 *
 * Fixes the "too fast" problem from the original index.html by keeping a
 * fixed-size sample ring buffer and redrawing the chosen time window on every
 * animation frame — regardless of broadcast rate.
 *
 * Exported API:
 *   init(canvasIds)      — bind to DOM canvas elements
 *   push(msg)            — ingest a stream broadcast message
 *   setTimeWindow(ms)    — change the visible time window (default 100 ms)
 *   setChannel(ch)       — select which channel index to display
 */

// ── Constants ────────────────────────────────────────────────────────────────
const EFFECTIVE_FS  = 20000;          // post-decimation rate (Hz)
const MAX_RING_MS   = 2000;           // maximum ring buffer size (ms)
const MAX_RING_SIZE = Math.ceil(EFFECTIVE_FS * MAX_RING_MS / 1000);

// Canvas heights (match original index.html)
const WAVE_H = 120;
const DN_H   = 16;
const NG_H   = 16;
const CTRL_H = 60;

// ── State ────────────────────────────────────────────────────────────────────
const ring     = new Float32Array(MAX_RING_SIZE);
const dnRing   = new Uint8Array(MAX_RING_SIZE);
const ngRing   = new Float32Array(MAX_RING_SIZE).fill(1);
const ctrlRing = new Float32Array(MAX_RING_SIZE);

let ringHead     = 0;      // next write position (monotonic, wrap via %)
let waveMax      = 1e-6;   // auto-scale peak tracker
let timeWindowMs = 100;    // visible time window in ms
let activeChannel = 0;

// Pending state flags (latest broadcast values)
let pendingInhActive = false;

// Canvas contexts (filled by init())
let waveCtx, dnCtx, ngCtx, ctrlCtx;
let CVW = 900;  // canvas width (read from element)

// ── Public API ───────────────────────────────────────────────────────────────

export function init(ids) {
  const waveCv = document.getElementById(ids.wave);
  const dnCv   = document.getElementById(ids.dn);
  const ngCv   = document.getElementById(ids.ng);
  const ctrlCv = document.getElementById(ids.ctrl);

  CVW = waveCv.width;
  waveCtx = waveCv.getContext('2d');
  dnCtx   = dnCv.getContext('2d');
  ngCtx   = ngCv.getContext('2d');
  ctrlCtx = ctrlCv.getContext('2d');

  _clearAll();
  requestAnimationFrame(_render);
}

export function push(msg) {
  // Only ingest if this message is for the active channel (or channel-less)
  const ch = msg.channel ?? 0;
  if (ch !== activeChannel) return;

  const samples  = msg.samples  || [];
  const dnFlags  = msg.dn_flags || [];
  const ngVal    = msg.noise_gate ?? 1.0;
  const ctrlVal  = msg.control   ?? 0.0;
  pendingInhActive = msg.inhibition_active ?? false;

  for (let i = 0; i < samples.length; i++) {
    const pos = ringHead % MAX_RING_SIZE;
    ring[pos]   = samples[i];
    dnRing[pos] = dnFlags[i] ? 1 : 0;
    ngRing[pos] = ngVal;
    ctrlRing[pos] = ctrlVal;
    ringHead++;
  }
}

export function setTimeWindow(ms) {
  timeWindowMs = Math.max(10, Math.min(MAX_RING_MS, ms));
}

export function setChannel(ch) {
  activeChannel = ch;
}

// ── Internal render ───────────────────────────────────────────────────────────

function _clearAll() {
  [waveCtx, dnCtx, ngCtx, ctrlCtx].forEach(ctx => {
    if (!ctx) return;
    ctx.fillStyle = '#020202';
    ctx.fillRect(0, 0, CVW, ctx.canvas.height);
  });
}

function _render() {
  const windowSamples = Math.ceil(timeWindowMs * EFFECTIVE_FS / 1000);
  const pxPerSample   = CVW / windowSamples;

  // How many samples are available?
  const available = Math.min(ringHead, MAX_RING_SIZE);
  const drawCount  = Math.min(windowSamples, available);
  const startIdx   = ringHead - drawCount;  // absolute monotonic index

  if (waveCtx) _drawWaveform(startIdx, drawCount, windowSamples, pxPerSample);
  if (dnCtx)   _drawDN(startIdx, drawCount, windowSamples, pxPerSample);
  if (ngCtx)   _drawNG(startIdx, drawCount, windowSamples, pxPerSample);
  if (ctrlCtx) _drawCtrl(startIdx, drawCount, windowSamples, pxPerSample);

  requestAnimationFrame(_render);
}

function _sample(buf, absIdx) {
  return buf[((absIdx % MAX_RING_SIZE) + MAX_RING_SIZE) % MAX_RING_SIZE];
}

function _drawWaveform(startIdx, drawCount, windowSamples, pxPerSample) {
  const ctx = waveCtx;
  ctx.fillStyle = '#020202';
  ctx.fillRect(0, 0, CVW, WAVE_H);

  if (drawCount === 0) return;

  // Auto-scale: fast attack, slow decay
  for (let i = 0; i < drawCount; i++) {
    const a = Math.abs(_sample(ring, startIdx + i));
    if (a > waveMax) waveMax = a * 1.2;
  }
  waveMax *= 0.9998;
  if (waveMax < 1e-10) waveMax = 1e-10;

  // Center line
  ctx.strokeStyle = '#111';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(0, WAVE_H / 2);
  ctx.lineTo(CVW, WAVE_H / 2);
  ctx.stroke();

  // DN tint
  for (let i = 0; i < drawCount; i++) {
    if (_sample(dnRing, startIdx + i)) {
      ctx.fillStyle = 'rgba(255,102,0,0.25)';
      ctx.fillRect(i * pxPerSample, 0, Math.max(1, pxPerSample), WAVE_H);
    }
  }

  // Inhibition tint
  if (pendingInhActive) {
    ctx.fillStyle = 'rgba(255,50,50,0.1)';
    ctx.fillRect(0, 0, CVW, WAVE_H);
  }

  // Waveform line
  ctx.save();
  ctx.shadowColor = '#00ff41';
  ctx.shadowBlur  = 4;
  ctx.strokeStyle = '#00ff41';
  ctx.lineWidth   = 1.5;
  ctx.lineJoin    = 'round';
  ctx.beginPath();
  for (let i = 0; i < drawCount; i++) {
    const yNorm = _sample(ring, startIdx + i) / waveMax;
    const y = Math.max(2, Math.min(WAVE_H - 2,
      WAVE_H / 2 - yNorm * (WAVE_H / 2 - 4)));
    const x = i * pxPerSample;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

function _drawDN(startIdx, drawCount, windowSamples, pxPerSample) {
  const ctx = dnCtx;
  ctx.fillStyle = '#020202';
  ctx.fillRect(0, 0, CVW, DN_H);
  for (let i = 0; i < drawCount; i++) {
    if (_sample(dnRing, startIdx + i)) {
      ctx.fillStyle = '#ff6600';
      ctx.fillRect(i * pxPerSample, 0, Math.max(1, pxPerSample), DN_H);
    }
  }
}

function _drawNG(startIdx, drawCount, windowSamples, pxPerSample) {
  const ctx = ngCtx;
  ctx.fillStyle = '#020202';
  ctx.fillRect(0, 0, CVW, NG_H);

  // Draw suppression band across the window (use latest value for simplicity)
  if (drawCount > 0) {
    const ngVal = _sample(ngRing, startIdx + drawCount - 1);
    if (ngVal < 0.95) {
      const alpha = (1.0 - ngVal) * 0.8;
      ctx.fillStyle = `rgba(255,51,204,${alpha})`;
      ctx.fillRect(0, 0, CVW, NG_H);
    }
    // Current level dot at right edge
    const lineY = NG_H - ngVal * NG_H;
    ctx.fillStyle = '#00ff41';
    ctx.fillRect(CVW - 2, lineY, 2, 2);
  }
}

function _drawCtrl(startIdx, drawCount, windowSamples, pxPerSample) {
  const ctx = ctrlCtx;
  ctx.fillStyle = '#020202';
  ctx.fillRect(0, 0, CVW, CTRL_H);

  // Zero line
  ctx.strokeStyle = '#111';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(0, CTRL_H / 2);
  ctx.lineTo(CVW, CTRL_H / 2);
  ctx.stroke();

  // Control trace
  ctx.fillStyle = '#00ccff';
  for (let i = 0; i < drawCount; i++) {
    const cn = Math.max(-1, Math.min(1, _sample(ctrlRing, startIdx + i)));
    const cy = CTRL_H / 2 - cn * (CTRL_H / 2 - 2);
    ctx.fillRect(i * pxPerSample, cy - 1, Math.max(1, pxPerSample), 2);
  }
}
