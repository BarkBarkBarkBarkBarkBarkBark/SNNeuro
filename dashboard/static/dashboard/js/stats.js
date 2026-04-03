/**
 * stats.js — Stats bar updater (1 Hz refresh) with per-channel tracking.
 *
 * All channels accumulate stats continuously.  The display shows
 * whichever channel is active — switching channels reflects live data.
 *
 * Exported API:
 *   init()        — bind DOM elements
 *   update(msg)   — ingest stream broadcast (any channel)
 *   setChannel(ch)
 */

// ── Per-channel state (lazily created) ───────────────────────────────────────
const _channels = {};

function _ensureChannel(ch) {
  if (_channels[ch]) return _channels[ch];
  _channels[ch] = {
    totalL1: 0,
    totalDN: 0,
    lastCheck: performance.now(),
    lastMsg: null,  // latest broadcast for display snapshot
  };
  return _channels[ch];
}

let activeChannel = 0;

export function init() {
  // Elements are already in the DOM via the _stats_bar partial
}

/**
 * Ingest a broadcast message.  Stats are ALWAYS accumulated for every
 * channel — not just the active one.
 */
export function update(msg) {
  const ch    = msg.channel ?? 0;
  const state = _ensureChannel(ch);

  const spikes  = msg.spikes    || [];
  const dnFlags = msg.dn_flags  || [];
  const dnSum   = dnFlags.reduce((a, b) => a + b, 0);

  state.totalL1 += spikes.length;
  state.totalDN += dnSum;
  state.lastMsg = msg;

  // Only update the DOM if this is the active channel
  if (ch !== activeChannel) return;

  if (msg.dec_hex) {
    _set('s_hex', msg.dec_hex);
  }

  const now = performance.now();
  if (now - state.lastCheck >= 1000) {
    _flush(state, msg, now);
  }
}

export function setChannel(ch) {
  activeChannel = ch;
  // Immediately flush the new channel's latest state so the display
  // updates on switch rather than waiting up to 1 second.
  const state = _channels[ch];
  if (state && state.lastMsg) {
    _flush(state, state.lastMsg, performance.now());
  }
}

function _flush(state, msg, now) {
  const dt = (now - state.lastCheck) / 1000;
  const lastSample = (msg.samples || [0]).at(-1) || 0;
  _set('s_t',      msg.t);
  _set('s_sample', lastSample.toExponential(3));
  _set('s_dn',     `${(state.totalDN / dt).toFixed(0)} fires/s`);
  _set('s_l1hz',   (state.totalL1 / dt).toFixed(0));
  _set('s_ctrl',   (msg.control ?? 0).toFixed(4));
  _set('s_conf',   (msg.confidence ?? 0).toFixed(4));
  _set('s_ng',     (msg.noise_gate ?? 1.0).toFixed(3));
  _set('s_inh',    (msg.inhibition_active ?? false) ? 'ACTIVE' : 'idle');
  if (msg.dec_hex) _set('s_hex', msg.dec_hex);
  state.totalL1 = 0;
  state.totalDN = 0;
  state.lastCheck = now;
}

function _set(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
