/**
 * waveform.js — Ring-buffer waveform renderer with per-channel buffers.
 *
 * Every channel has its own independent ring buffer.  All incoming data is
 * pushed into the correct channel's buffer regardless of which channel is
 * currently selected for display.  Switching channels instantly shows the
 * current state — no data is lost or "paused".
 *
 * Exported API:
 *   init(canvasIds)      — bind to DOM canvas elements
 *   push(msg)            — ingest a stream broadcast message (any channel)
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

// ── Per-channel state (lazily created) ───────────────────────────────────────
const _channels = {};

function _ensureChannel(ch) {
  if (_channels[ch]) return _channels[ch];
  _channels[ch] = {
    ring:      new Float32Array(MAX_RING_SIZE),
    dnRing:    new Uint8Array(MAX_RING_SIZE),
    ngRing:    new Float32Array(MAX_RING_SIZE).fill(1),
    ctrlRing:  new Float32Array(MAX_RING_SIZE),
    head:      0,
    waveMax:   1e-6,
    inhActive: false,
  };
  return _channels[ch];
}

// ── Global render state ──────────────────────────────────────────────────────
let timeWindowMs  = 100;
let activeChannel = 0;

// Canvas contexts (filled by init())
let waveCtx, dnCtx, ngCtx, ctrlCtx;
let CVW = 900;

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

/**
 * Ingest a broadcast message.  Data is ALWAYS pushed into the channel's
 * own ring buffer — even if that channel is not currently displayed.
 */
export function push(msg) {
  const ch    = msg.channel ?? 0;
  const state = _ensureChannel(ch);

  const samples  = msg.samples  || [];
  const dnFlags  = msg.dn_flags || [];
  const ngVal    = msg.noise_gate ?? 1.0;
  const ctrlVal  = msg.control   ?? 0.0;
  state.inhActive = msg.inhibition_active ?? false;

  for (let i = 0; i < samples.length; i++) {
    const pos = state.head % MAX_RING_SIZE;
    state.ring[pos]     = samples[i];
    state.dnRing[pos]   = dnFlags[i] ? 1 : 0;
    state.ngRing[pos]   = ngVal;
    state.ctrlRing[pos] = ctrlVal;
    state.head++;
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
  const state = _channels[activeChannel];
  if (!state) {
    _clearAll();
    requestAnimationFrame(_render);
    return;
  }

  const windowSamples = Math.ceil(timeWindowMs * EFFECTIVE_FS / 1000);
  const pxPerSample   = CVW / windowSamples;

  const available = Math.min(state.head, MAX_RING_SIZE);
  const drawCount = Math.min(windowSamples, available);
  const startIdx  = state.head - drawCount;

  if (waveCtx) _drawWaveform(state, startIdx, drawCount, pxPerSample);
  if (dnCtx)   _drawDN(state, startIdx, drawCount, pxPerSample);
  if (ngCtx)   _drawNG(state, startIdx, drawCount, pxPerSample);
  if (ctrlCtx) _drawCtrl(state, startIdx, drawCount, pxPerSample);

  requestAnimationFrame(_render);
}

function _sample(buf, absIdx) {
  return buf[((absIdx % MAX_RING_SIZE) + MAX_RING_SIZE) % MAX_RING_SIZE];
}

function _drawWaveform(state, startIdx, drawCount, pxPerSample) {
  const ctx = waveCtx;
  ctx.fillStyle = '#020202';
  ctx.fillRect(0, 0, CVW, WAVE_H);

  if (drawCount === 0) return;

  for (let i = 0; i < drawCount; i++) {
    const a = Math.abs(_sample(state.ring, startIdx + i));
    if (a > state.waveMax) state.waveMax = a * 1.2;
  }
  state.waveMax *= 0.9998;
  if (state.waveMax < 1e-10) state.waveMax = 1e-10;

  ctx.strokeStyle = '#111';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(0, WAVE_H / 2);
  ctx.lineTo(CVW, WAVE_H / 2);
  ctx.stroke();

  for (let i = 0; i < drawCount; i++) {
    if (_sample(state.dnRing, startIdx + i)) {
      ctx.fillStyle = 'rgba(255,102,0,0.25)';
      ctx.fillRect(i * pxPerSample, 0, Math.max(1, pxPerSample), WAVE_H);
    }
  }

  if (state.inhActive) {
    ctx.fillStyle = 'rgba(255,50,50,0.1)';
    ctx.fillRect(0, 0, CVW, WAVE_H);
  }

  ctx.save();
  ctx.shadowColor = '#00ff41';
  ctx.shadowBlur  = 4;
  ctx.strokeStyle = '#00ff41';
  ctx.lineWidth   = 1.5;
  ctx.lineJoin    = 'round';
  ctx.beginPath();
  for (let i = 0; i < drawCount; i++) {
    const yNorm = _sample(state.ring, startIdx + i) / state.waveMax;
    const y = Math.max(2, Math.min(WAVE_H - 2,
      WAVE_H / 2 - yNorm * (WAVE_H / 2 - 4)));
    const x = i * pxPerSample;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

function _drawDN(state, startIdx, drawCount, pxPerSample) {
  const ctx = dnCtx;
  ctx.fillStyle = '#020202';
  ctx.fillRect(0, 0, CVW, DN_H);
  for (let i = 0; i < drawCount; i++) {
    if (_sample(state.dnRing, startIdx + i)) {
      ctx.fillStyle = '#ff6600';
      ctx.fillRect(i * pxPerSample, 0, Math.max(1, pxPerSample), DN_H);
    }
  }
}

function _drawNG(state, startIdx, drawCount, pxPerSample) {
  const ctx = ngCtx;
  ctx.fillStyle = '#020202';
  ctx.fillRect(0, 0, CVW, NG_H);

  if (drawCount > 0) {
    const ngVal = _sample(state.ngRing, startIdx + drawCount - 1);
    if (ngVal < 0.95) {
      const alpha = (1.0 - ngVal) * 0.8;
      ctx.fillStyle = `rgba(255,51,204,${alpha})`;
      ctx.fillRect(0, 0, CVW, NG_H);
    }
    const lineY = NG_H - ngVal * NG_H;
    ctx.fillStyle = '#00ff41';
    ctx.fillRect(CVW - 2, lineY, 2, 2);
  }
}

function _drawCtrl(state, startIdx, drawCount, pxPerSample) {
  const ctx = ctrlCtx;
  ctx.fillStyle = '#020202';
  ctx.fillRect(0, 0, CVW, CTRL_H);

  ctx.strokeStyle = '#111';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(0, CTRL_H / 2);
  ctx.lineTo(CVW, CTRL_H / 2);
  ctx.stroke();

  ctx.fillStyle = '#00ccff';
  for (let i = 0; i < drawCount; i++) {
    const cn = Math.max(-1, Math.min(1, _sample(state.ctrlRing, startIdx + i)));
    const cy = CTRL_H / 2 - cn * (CTRL_H / 2 - 2);
    ctx.fillRect(i * pxPerSample, cy - 1, Math.max(1, pxPerSample), 2);
  }
}
