/**
 * launcher.js — Source launcher bar logic.
 *
 * Handles mode indicator, file load, file browse, and synthetic launch buttons.
 * Sends commands via the sendRaw() helper from websocket.js.
 *
 * Exported API:
 *   init(sendRaw)      — bind all launcher controls
 *   handleStatus(msg)  — process server status / mode_change messages
 */

let _sendRaw = null;
let _getNumChannels = () => 1;

export function init(sendRaw, getNumChannels = () => 1) {
  _sendRaw = sendRaw;
  _getNumChannels = getNumChannels;

  const btnFile     = document.getElementById('btn_file');
  const btnBrowse   = document.getElementById('btn_browse');
  const btnSynthetic = document.getElementById('btn_synthetic');

  if (btnFile) {
    btnFile.addEventListener('click', () => {
      const path = document.getElementById('file_path')?.value.trim();
      if (!path) { _msg('Enter a file path', true); return; }
      setMode('file', 'starting');
      _msg('Loading file…');
      sendRaw({ launch_file: path });
    });
  }

  if (btnBrowse) {
    btnBrowse.addEventListener('click', () => {
      sendRaw({ list_files: true });
    });
  }

  if (btnSynthetic) {
    btnSynthetic.addEventListener('click', () => {
      setMode('synthetic', 'starting');
      _msg('Generating synthetic recording…');
      sendRaw({
        launch_synthetic: {
          duration_s:   parseFloat(document.getElementById('synth_dur')?.value  || 20),
          num_units:    parseInt(document.getElementById('synth_units')?.value  || 2),
          noise_level:  parseFloat(document.getElementById('synth_noise')?.value || 8),
          num_channels: _getNumChannels(),
        },
      });
    });
  }
}

export function handleStatus(msg) {
  if (msg.mode_change) {
    setMode(msg.mode_change.mode, msg.mode_change.state);
    if (msg.mode_change.state === 'finished') _msg('Playback complete');
    return;
  }
  if (msg.status === 'ok' && msg.mode) {
    setMode(msg.mode, 'running');
    if (msg.file) _msg(`▶ ${msg.file} (${msg.duration_s}s at ${msg.sfreq} Hz)`);
    else          _msg(`▶ ${msg.mode} mode active`);
    return;
  }
  if (msg.status === 'error') {
    setMode('error', 'error');
    _msg(msg.message, true);
    return;
  }
  if (msg.files !== undefined) {
    if (msg.files.length > 0) {
      const fpEl = document.getElementById('file_path');
      if (fpEl) fpEl.value = msg.files[0];
      _msg(`Found ${msg.files.length} file(s) in ${msg.directory}`);
    } else {
      _msg('No .ncs files found in data/raw/', true);
    }
  }
}

export function setMode(mode, state) {
  const dot   = document.getElementById('mode_dot');
  const label = document.getElementById('mode_label');
  if (!dot || !label) return;
  label.textContent = mode.toUpperCase();
  dot.className = 'mode-dot';
  if (state === 'running' || state === 'finished') dot.classList.add('active');
  if (state === 'error')   dot.classList.add('error');
  if (state === 'starting') dot.classList.add('starting');
}

function _msg(text, isError = false) {
  const el = document.getElementById('launcher_msg');
  if (!el) return;
  el.textContent = text;
  el.style.color = isError ? '#ff3333' : '#00ff41';
  if (text) setTimeout(() => { el.textContent = ''; }, 10000);
}
