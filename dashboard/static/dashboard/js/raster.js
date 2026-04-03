/**
 * raster.js — L1 template raster and DEC decoder raster renderers.
 *
 * Per-channel event buffers: all incoming spike data is stored regardless
 * of which channel is currently displayed.  Switching channels instantly
 * shows the latest raster state — no data loss.
 *
 * Exported API:
 *   init(ids)          — bind canvas elements
 *   push(msg)          — ingest stream broadcast (any channel)
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

// ── Per-channel state (lazily created) ───────────────────────────────────────
const MAX_EVENTS = 8192;
const _channels = {};

function _ensureChannel(ch) {
  if (_channels[ch]) return _channels[ch];
  _channels[ch] = {
    l1Events:   new Array(MAX_EVENTS),
    decEvents:  new Array(MAX_EVENTS),
    l1Head:     0,
    decHead:    0,
    sampleClock: 0,
  };
  return _channels[ch];
}

// ── Global render state ──────────────────────────────────────────────────────
let timeWindowMs  = 100;
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

/**
 * Ingest a broadcast message.  Spike events are ALWAYS pushed into
 * the channel's own buffer — even if not currently displayed.
 */
export function push(msg) {
  const ch    = msg.channel ?? 0;
  const state = _ensureChannel(ch);

  const nSamples = (msg.samples || []).length;
  const t = state.sampleClock;
  state.sampleClock += nSamples;

  for (const id of (msg.spikes || [])) {
    state.l1Events[state.l1Head % MAX_EVENTS] = { t: t + nSamples, id };
    state.l1Head++;
  }
  for (const id of (msg.dec_spikes || [])) {
    state.decEvents[state.decHead % MAX_EVENTS] = { t: t + nSamples, id };
    state.decHead++;
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
  const state = _channels[activeChannel];
  if (rastCtx) _drawL1(state);
  if (decCtx)  _drawDEC(state);
  requestAnimationFrame(_render);
}

function _drawL1(state) {
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

  if (!state) return;

  const windowSamples = Math.ceil(timeWindowMs * EFFECTIVE_FS / 1000);
  const tNow   = state.sampleClock;
  const tStart  = tNow - windowSamples;

  ctx.fillStyle = '#00ff41';
  for (let i = state.l1Head - 1; i >= Math.max(0, state.l1Head - MAX_EVENTS); i--) {
    const ev = state.l1Events[i % MAX_EVENTS];
    if (!ev || ev.t < tStart) break;
    if (ev.id < N_L1) {
      const x = Math.floor((ev.t - tStart) / windowSamples * CVW);
      ctx.fillRect(x, ev.id * ROW_H, 2, Math.max(1, ROW_H - 0.5));
    }
  }
}

function _drawDEC(state) {
  const ctx = decCtx;
  _clear(ctx, DEC_H);

  if (!state) return;

  const windowSamples = Math.ceil(timeWindowMs * EFFECTIVE_FS / 1000);
  const tNow   = state.sampleClock;
  const tStart  = tNow - windowSamples;

  for (let i = state.decHead - 1; i >= Math.max(0, state.decHead - MAX_EVENTS); i--) {
    const ev = state.decEvents[i % MAX_EVENTS];
    if (!ev || ev.t < tStart) break;
    if (ev.id < N_DEC) {
      ctx.fillStyle = DEC_COLORS[ev.id];
      const x = Math.floor((ev.t - tStart) / windowSamples * CVW);
      ctx.fillRect(x, ev.id * DEC_ROW_H, 2, Math.max(1, DEC_ROW_H - 1));
    }
  }
}
