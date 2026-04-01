/**
 * controls.js — Parameter slider and select bindings.
 *
 * Exported API:
 *   init(sendCmd)          — bind all sliders; pass sendCmd from websocket.js
 *   initDNThreshold(val)   — set DN slider value from first broadcast
 */

let _sendCmd = null;
let _thInit = false;

export function init(sendCmd) {
  _sendCmd = sendCmd;

  _bind('dn_th_slider',    'dn_th_val',      'dn_threshold',         v => v.toFixed(0));
  _bind('ltp_slider',      'ltp_val',        'l1_stdp_ltp',          v => v.toFixed(3));
  _bind('ltd_slider',      'ltd_val',        'l1_stdp_ltd',          v => v.toFixed(4));
  _bind('inh_dur_slider',  'inh_dur_val',    'inh_duration_ms',      v => v.toFixed(1));
  _bind('inh_str_slider',  'inh_str_val',    'inh_strength_threshold', v => v.toFixed(0));
  _bind('ng_sd_slider',    'ng_sd_val',      'ng_inhibit_below_sd',  v => v.toFixed(1));
  _bind('ttl_width_slider','ttl_width_val',  'ttl_width_ms',         v => v.toFixed(1));
  _bind('ttl_high_slider', 'ttl_high_val',   'ttl_high',             v => v.toFixed(2));

  // Time window slider (client-side only — no WS command)
  _bindLocal('time_window_slider', 'time_window_val', v => v.toFixed(0) + ' ms');

  // Decoder strategy select
  const decSel = document.getElementById('decoder_sel');
  if (decSel) {
    decSel.addEventListener('change', () => {
      sendCmd('decoder_strategy', decSel.value);
      _updateTTLVisibility();
    });
    _updateTTLVisibility();
  }
}

export function initDNThreshold(val) {
  if (_thInit) return;
  const slider = document.getElementById('dn_th_slider');
  const label  = document.getElementById('dn_th_val');
  if (slider) { slider.value = val; slider.disabled = false; }
  if (label)  { label.textContent = val.toFixed(0); }
  _thInit = true;
}

/** Return the current time_window slider value in ms (used by waveform/raster). */
export function getTimeWindowMs() {
  const el = document.getElementById('time_window_slider');
  return el ? parseFloat(el.value) : 100;
}

// ── Internals ─────────────────────────────────────────────────────────────────

function _bind(sliderId, labelId, wsKey, fmt) {
  const slider = document.getElementById(sliderId);
  const label  = document.getElementById(labelId);
  if (!slider) return;
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    if (label) label.textContent = fmt(v);
    if (_sendCmd) _sendCmd(wsKey, v);
  });
}

function _bindLocal(sliderId, labelId, fmt) {
  const slider = document.getElementById(sliderId);
  const label  = document.getElementById(labelId);
  if (!slider) return;
  if (label) label.textContent = fmt(parseFloat(slider.value));
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    if (label) label.textContent = fmt(v);
    // Dispatch a custom event so waveform.js can react
    window.dispatchEvent(new CustomEvent('timeWindowChange', { detail: v }));
  });
}

function _updateTTLVisibility() {
  const decSel = document.getElementById('decoder_sel');
  const isTTL  = decSel && decSel.value === 'ttl';
  ['ttl_width_row', 'ttl_high_row'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = isTTL ? '' : 'none';
  });
}
