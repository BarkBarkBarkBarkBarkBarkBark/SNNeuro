/**
 * stats.js — Stats bar updater (1 Hz refresh).
 *
 * Exported API:
 *   init()        — bind DOM elements
 *   update(msg)   — ingest stream broadcast
 *   setChannel(ch)
 */

let totalL1 = 0, totalDN = 0, lastCheck = performance.now();
let _pendingNG = 1.0, _pendingInh = false, _pendingHex = '0x0000';
let activeChannel = 0;

export function init() {
  // Elements are already in the DOM via the _stats_bar partial
}

export function update(msg) {
  const ch = msg.channel ?? 0;
  if (ch !== activeChannel) return;

  const spikes  = msg.spikes    || [];
  const dnFlags = msg.dn_flags  || [];
  const dnSum   = dnFlags.reduce((a, b) => a + b, 0);

  totalL1 += spikes.length;
  totalDN += dnSum;

  _pendingNG  = msg.noise_gate ?? 1.0;
  _pendingInh = msg.inhibition_active ?? false;

  if (msg.dec_hex) {
    _pendingHex = msg.dec_hex;
    _set('s_hex', msg.dec_hex);
  }

  const now = performance.now();
  if (now - lastCheck >= 1000) {
    const dt = (now - lastCheck) / 1000;
    const lastSample = (msg.samples || [0]).at(-1) || 0;
    _set('s_t',      msg.t);
    _set('s_sample', lastSample.toExponential(3));
    _set('s_dn',     `${(totalDN / dt).toFixed(0)} fires/s`);
    _set('s_l1hz',   (totalL1 / dt).toFixed(0));
    _set('s_ctrl',   (msg.control ?? 0).toFixed(4));
    _set('s_conf',   (msg.confidence ?? 0).toFixed(4));
    _set('s_ng',     _pendingNG.toFixed(3));
    _set('s_inh',    _pendingInh ? 'ACTIVE' : 'idle');
    totalL1 = 0; totalDN = 0; lastCheck = now;
  }
}

export function setChannel(ch) {
  activeChannel = ch;
  totalL1 = 0;
  totalDN = 0;
  lastCheck = performance.now();
  _pendingNG = 1.0;
  _pendingInh = false;
  _pendingHex = '0x0000';
}

function _set(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
