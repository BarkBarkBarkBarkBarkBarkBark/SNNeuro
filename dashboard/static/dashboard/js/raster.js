/**
 * raster.js — L1 template raster and DEC decoder raster renderers.
 *
 * Uses the same ring-buffer approach as waveform.js:
 * spikes are stored as timestamped events and drawn within the visible window.
 *
 * Exported API:
 *   init(ids)          — bind canvas elements
 *   push(msg)          — ingest stream broadcast
 *   setTimeWindow(ms)  — sync time window with waveform
 *   setChannel(ch)     — active channel filter
 */

const N_L1  = 110;
const N_DEC = 16;

// DEC neuron colour palette: 0 = white (any-fire), 1–15 = rainbow
const DEC_COLORS = ['#ffffff'];
for (let i = 1; i < N_DEC; i++) {
  DEC_COLORS.push(`hsl(${((i - 1) / (N_DEC - 1)) * 300}, 100%, 60%)`);
}

// ── State ────────────────────────────────────────────────────────────────────
// Spike events stored as {t: absoluteSampleIndex, id: neuronIndex}
const MAX_EVENTS = 8192;
const l1Events  = new Array(MAX_EVENTS);
const decEvents = new Array(MAX_EVENTS);
let l1Head  = 0, decHead  = 0;
let sampleClock = 0;    // running count of pushed samples
let timeWindowMs = 100;
let activeChannel = 0;

// Canvas contexts
let rastCtx, decCtx;
let CVW = 900;
const RASTER_H  = 180;
const DEC_H     = 48;
const ROW_H     = RASTER_H / N_L1;
const DEC_ROW_H = DEC_H / N_DEC;

// ── Public API ────────────────────────────────────────────────────────────────

export function init(ids) {
  const rastCv = document.getElementById(ids.raster);
  const decCv  = document.getElementById(ids.dec);
  CVW = rastCv.width;
  rastCtx = rastCv.getContext('2d');
  decCtx  = decCv.getContext('2d');
  _clear(rastCtx, RASTER_H);
  _clear(decCtx,  DEC_H);
  requestAnimationFrame(_render);
}

export function push(msg) {
  const ch = msg.channel ?? 0;
  if (ch !== activeChannel) return;

  const nSamples = (msg.samples || []).length;
  const t = sampleClock;
  sampleClock += nSamples;

  for (const id of (msg.spikes || [])) {
    l1Events[l1Head % MAX_EVENTS] = { t: t + nSamples, id };
    l1Head++;
  }
  for (const id of (msg.dec_spikes || [])) {
    decEvents[decHead % MAX_EVENTS] = { t: t + nSamples, id };
    decHead++;
  }
}

export function setTimeWindow(ms) {
  timeWindowMs = ms;
}

export function setChannel(ch) {
  activeChannel = ch;
}

// ── Internal render ───────────────────────────────────────────────────────────

function _clear(ctx, h) {
  ctx.fillStyle = '#020202';
  ctx.fillRect(0, 0, CVW, h);
}

const EFFECTIVE_FS = 20000;

function _render() {
  if (rastCtx) _drawL1();
  if (decCtx)  _drawDEC();
  requestAnimationFrame(_render);
}

function _drawL1() {
  const ctx = rastCtx;
  _clear(ctx, RASTER_H);

  // Grid lines every 10 neurons
  ctx.strokeStyle = '#0d0d0d';
  ctx.lineWidth = 0.5;
  for (let k = 0; k <= N_L1; k += 10) {
    ctx.beginPath();
    ctx.moveTo(0, k * ROW_H);
    ctx.lineTo(CVW, k * ROW_H);
    ctx.stroke();
  }

  const windowSamples = Math.ceil(timeWindowMs * EFFECTIVE_FS / 1000);
  const tNow = sampleClock;
  const tStart = tNow - windowSamples;

  ctx.fillStyle = '#00ff41';
  for (let i = l1Head - 1; i >= Math.max(0, l1Head - MAX_EVENTS); i--) {
    const ev = l1Events[i % MAX_EVENTS];
    if (!ev || ev.t < tStart) break;
    if (ev.id < N_L1) {
      const x = Math.floor((ev.t - tStart) / windowSamples * CVW);
      ctx.fillRect(x, ev.id * ROW_H, 2, Math.max(1, ROW_H - 0.5));
    }
  }
}

function _drawDEC() {
  const ctx = decCtx;
  _clear(ctx, DEC_H);

  const windowSamples = Math.ceil(timeWindowMs * EFFECTIVE_FS / 1000);
  const tNow = sampleClock;
  const tStart = tNow - windowSamples;

  for (let i = decHead - 1; i >= Math.max(0, decHead - MAX_EVENTS); i--) {
    const ev = decEvents[i % MAX_EVENTS];
    if (!ev || ev.t < tStart) break;
    if (ev.id < N_DEC) {
      ctx.fillStyle = DEC_COLORS[ev.id];
      const x = Math.floor((ev.t - tStart) / windowSamples * CVW);
      ctx.fillRect(x, ev.id * DEC_ROW_H, 2, Math.max(1, DEC_ROW_H - 1));
    }
  }
}
