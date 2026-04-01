/**
 * channels.js — Channel card strip renderer with auto-detection.
 *
 * Renders a row of compact channel cards. Each card shows:
 *   - Channel number + mode dot
 *   - Mini waveform (ring buffer, sampled at low rate)
 *   - DN fire rate and L1 spikes/s
 *
 * Channel cards are created **automatically** when a WS message arrives
 * with an unseen channel index — no upfront channel count needed.
 * The strip auto-shows when more than 1 channel is detected.
 *
 * Exported API:
 *   init(onSelect)                — bind select callback, start render loop
 *   update(msg)                   — ingest stream broadcast for any channel
 *   setChannel(ch)                — programmatically select a channel
 *   getNumChannels()              — return number of channels seen so far
 */

const MINI_W = 150;
const MINI_H = 32;
const MINI_RING = 1000;

let _onSelect = null;
let _activeChannel = 0;
let _renderStarted = false;

// Per-channel state
const _channels = {};  // { [ch]: {ring, head, dnRate, l1Rate, canvas, ctx} }

/**
 * Initialise the channel system.
 * @param {function} onSelect  — called with channel index when user clicks a card
 */
export function init(onSelect) {
  _onSelect = onSelect;
  const strip = document.getElementById('channel-strip');
  if (strip) strip.innerHTML = '';
  if (!_renderStarted) {
    _renderStarted = true;
    requestAnimationFrame(_renderMini);
  }
}

/** Return the number of channels discovered so far. */
export function getNumChannels() {
  return Object.keys(_channels).length || 1;
}

export function update(msg) {
  const ch = msg.channel ?? 0;

  // Auto-create a card for any channel index we haven't seen yet
  if (_channels[ch] === undefined) {
    _addChannel(ch);
  }

  const state = _channels[ch];

  const samples = msg.samples || [];
  const dnFlags = msg.dn_flags || [];
  const spikes  = msg.spikes  || [];

  for (const s of samples) {
    state.ring[state.head % MINI_RING] = s;
    state.head++;
  }
  state.dnCount += dnFlags.reduce((a, b) => a + b, 0);
  state.l1Count += spikes.length;

  // Update rate text once per second
  const now = performance.now();
  if (now - state.lastStatTime >= 1000) {
    const dt = (now - state.lastStatTime) / 1000;
    state.dnRateText = (state.dnCount / dt).toFixed(0);
    state.l1RateText = (state.l1Count / dt).toFixed(0);
    state.dnCount = 0; state.l1Count = 0;
    state.lastStatTime = now;
    const dnEl = document.getElementById(`ch-dn-${ch}`);
    const l1El = document.getElementById(`ch-l1-${ch}`);
    if (dnEl) dnEl.textContent = `DN ${state.dnRateText}/s`;
    if (l1El) l1El.textContent = `L1 ${state.l1RateText}/s`;
  }
}

export function setChannel(ch) {
  _selectChannel(ch);
}

// ── Internals ──────────────────────────────────────────────────────────────────

/** Create a channel card and append it to the strip. */
function _addChannel(ch) {
  _channels[ch] = {
    ring: new Float32Array(MINI_RING),
    head: 0,
    waveMax: 1e-6,
    dnCount: 0,
    l1Count: 0,
    lastStatTime: performance.now(),
    dnRateText: '0',
    l1RateText: '0',
    canvas: null,
    ctx: null,
  };

  const strip = document.getElementById('channel-strip');
  if (!strip) return;

  const card = document.createElement('div');
  card.className = 'ch-card' + (ch === 0 ? ' selected' : '');
  card.dataset.ch = ch;

  const title = document.createElement('div');
  title.className = 'ch-title';
  title.textContent = `CH ${ch}`;

  const canvas = document.createElement('canvas');
  canvas.width  = MINI_W;
  canvas.height = MINI_H;
  _channels[ch].canvas = canvas;
  _channels[ch].ctx    = canvas.getContext('2d');

  const statsEl = document.createElement('div');
  statsEl.className = 'ch-stats';
  statsEl.innerHTML =
    `<span class="ch-stat-dn" id="ch-dn-${ch}">DN —</span> · ` +
    `<span class="ch-stat-l1" id="ch-l1-${ch}">L1 —</span>`;
  _channels[ch].statsEl = statsEl;

  card.appendChild(title);
  card.appendChild(canvas);
  card.appendChild(statsEl);
  card.addEventListener('click', () => _selectChannel(ch));
  strip.appendChild(card);

  // Auto-show the strip once we have more than 1 channel
  if (Object.keys(_channels).length > 1) {
    strip.style.display = '';
  }
}

function _selectChannel(ch) {
  _activeChannel = ch;
  document.querySelectorAll('.ch-card').forEach(card => {
    card.classList.toggle('selected', parseInt(card.dataset.ch) === ch);
  });
  if (_onSelect) _onSelect(ch);
}

function _renderMini(ts) {
  for (const [ch, state] of Object.entries(_channels)) {
    if (!state.ctx) continue;
    const ctx    = state.ctx;
    const head   = state.head;
    const ring   = state.ring;
    const count  = Math.min(head, MINI_RING);
    const startI = head - count;

    ctx.fillStyle = parseInt(ch) === _activeChannel ? '#0a1a0a' : '#020202';
    ctx.fillRect(0, 0, MINI_W, MINI_H);

    if (count === 0) continue;

    // Auto-scale
    for (let i = 0; i < count; i++) {
      const a = Math.abs(ring[(startI + i) % MINI_RING]);
      if (a > state.waveMax) state.waveMax = a * 1.2;
    }
    state.waveMax *= 0.9999;
    if (state.waveMax < 1e-10) state.waveMax = 1e-10;

    ctx.strokeStyle = parseInt(ch) === _activeChannel ? '#00ff41' : '#2a4a2a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < count; i++) {
      const s = ring[(startI + i) % MINI_RING];
      const x = i / count * MINI_W;
      const y = MINI_H / 2 - (s / state.waveMax) * (MINI_H / 2 - 2);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  requestAnimationFrame(_renderMini);
}
