/**
 * network_viz.js — Network topology diagram renderer.
 * Ported directly from the inline drawNetworkViz() in index.html.
 *
 * Exported API:
 *   init(canvasId)   — bind canvas and start render loop
 *   update(msg)      — feed a stream broadcast message
 *   setChannel(ch)   — active channel filter
 */

const N_L1  = 110;
const N_DEC = 16;

const DEC_COLORS = ['#ffffff'];
for (let i = 1; i < N_DEC; i++) {
  DEC_COLORS.push(`hsl(${((i - 1) / (N_DEC - 1)) * 300}, 100%, 60%)`);
}

// ── State ────────────────────────────────────────────────────────────────────
let membrane   = new Float32Array(N_L1);
let dnActive   = false;
let ngFactor   = 1.0;
let inhActive  = false;
let l1Spikes   = new Set();
let decSpikes  = new Set();
let decHex     = '0x0000';
let activeChannel = 0;

let ctx;
let NET_W = 360, NET_H = 520;

// ── Public API ────────────────────────────────────────────────────────────────

export function init(canvasId) {
  const cv = document.getElementById(canvasId);
  NET_W = cv.width; NET_H = cv.height;
  ctx = cv.getContext('2d');
  requestAnimationFrame(_render);
}

export function update(msg) {
  const ch = msg.channel ?? 0;
  if (ch !== activeChannel) return;

  dnActive  = (msg.dn_flags || []).some(Boolean);
  ngFactor  = msg.noise_gate ?? 1.0;
  inhActive = msg.inhibition_active ?? false;
  l1Spikes  = new Set(msg.spikes || []);
  decSpikes = new Set(msg.dec_spikes || []);
  decHex    = msg.dec_hex || '0x0000';
  if (msg.l1_membrane) {
    const m = msg.l1_membrane;
    for (let i = 0; i < Math.min(m.length, N_L1); i++) membrane[i] = m[i];
  }
}

export function setChannel(ch) {
  activeChannel = ch;
}

// ── Render (capped at 15 fps via throttle) ────────────────────────────────────
let _lastDraw = 0;
function _render(ts) {
  if (ts - _lastDraw >= 66) { // ~15 fps
    _draw();
    _lastDraw = ts;
  }
  requestAnimationFrame(_render);
}

function _draw() {
  ctx.fillStyle = '#020202';
  ctx.fillRect(0, 0, NET_W, NET_H);

  const margin = 25;
  const colX = [50, 130, 210, 290];

  // ── Connections ──────────────────────────────────────────
  ctx.globalAlpha = 0.08;
  const dnY = 80;
  for (let i = 0; i < Math.min(N_L1, 40); i++) {
    const ly = 160 + i * (300 / Math.min(N_L1, 40));
    ctx.strokeStyle = dnActive ? '#ff6600' : '#222';
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(colX[1], dnY); ctx.lineTo(colX[2], ly); ctx.stroke();
  }
  ctx.globalAlpha = 1.0;

  // ── Encoder column ────────────────────────────────────────
  ctx.fillStyle = '#333';
  ctx.font = '9px Courier New';
  ctx.textAlign = 'center';
  ctx.fillText('ENC', colX[0], margin - 5);
  for (let i = 0; i < 20; i++) {
    ctx.fillStyle = '#1a3a1a';
    ctx.fillRect(colX[0] - 3, 40 + i * 22, 6, 3);
  }

  // ── DN + NG + INH column ──────────────────────────────────
  ctx.fillStyle = '#444';
  ctx.fillText('DN', colX[1], margin - 5);

  // DN node
  ctx.beginPath(); ctx.arc(colX[1], dnY, 12, 0, Math.PI * 2);
  ctx.fillStyle = dnActive ? '#ff6600' : '#2a1500';
  if (dnActive) { ctx.shadowColor = '#ff6600'; ctx.shadowBlur = 15; }
  ctx.fill(); ctx.shadowBlur = 0;
  ctx.strokeStyle = '#ff6600'; ctx.lineWidth = 1; ctx.stroke();

  // NG node
  ctx.fillStyle = '#444'; ctx.fillText('NG', colX[1], 130);
  ctx.beginPath(); ctx.arc(colX[1], 150, 10, 0, Math.PI * 2);
  ctx.fillStyle = ngFactor < 0.95 ? '#ff33cc' : '#1a0a1a';
  if (ngFactor < 0.95) { ctx.shadowColor = '#ff33cc'; ctx.shadowBlur = 10; }
  ctx.fill(); ctx.shadowBlur = 0;
  ctx.strokeStyle = '#ff33cc'; ctx.lineWidth = 1; ctx.stroke();

  ctx.fillStyle = '#111'; ctx.fillRect(colX[1] - 15, 170, 30, 6);
  ctx.fillStyle = ngFactor < 0.5 ? '#ff33cc' : '#1a3a1a';
  ctx.fillRect(colX[1] - 15, 170, 30 * ngFactor, 6);

  // INH node
  ctx.fillStyle = '#444'; ctx.fillText('INH', colX[1], 195);
  ctx.beginPath(); ctx.arc(colX[1], 210, 8, 0, Math.PI * 2);
  ctx.fillStyle = inhActive ? '#ff3333' : '#1a0a0a';
  if (inhActive) { ctx.shadowColor = '#ff3333'; ctx.shadowBlur = 8; }
  ctx.fill(); ctx.shadowBlur = 0;
  ctx.strokeStyle = '#ff3333'; ctx.lineWidth = 1; ctx.stroke();

  // ── L1 column ────────────────────────────────────────────
  ctx.fillStyle = '#444'; ctx.fillText('L1', colX[2], margin - 5);
  const nVis = Math.min(N_L1, 40);
  const l1StartY = 40;
  const l1Spacing = 300 / nVis;
  for (let i = 0; i < nVis; i++) {
    const ni = Math.floor(i * N_L1 / nVis);
    const y = l1StartY + i * l1Spacing;
    const mem = membrane[ni] || 0;
    const intensity = Math.min(1, Math.max(0, mem / 500));
    const fired = l1Spikes.has(ni);
    ctx.beginPath(); ctx.arc(colX[2], y, 3, 0, Math.PI * 2);
    if (fired) { ctx.fillStyle = '#00ff41'; ctx.shadowColor = '#00ff41'; ctx.shadowBlur = 8; }
    else { ctx.fillStyle = `rgb(0, ${Math.floor(30 + intensity * 180)}, 0)`; ctx.shadowBlur = 0; }
    ctx.fill(); ctx.shadowBlur = 0;
  }

  // ── DEC column ────────────────────────────────────────────
  ctx.fillStyle = '#444'; ctx.fillText('DEC', colX[3], margin - 5);
  for (let i = 0; i < N_DEC; i++) {
    const y = 40 + i * 28;
    const fired = decSpikes.has(i);
    const r = i === 0 ? 6 : 4;
    const col = DEC_COLORS[i];
    const dimCol = i === 0 ? '#1a1a1a' : `hsl(${((i-1)/(N_DEC-1))*300}, 30%, 12%)`;
    ctx.beginPath(); ctx.arc(colX[3], y, r, 0, Math.PI * 2);
    ctx.fillStyle = fired ? col : dimCol;
    if (fired) { ctx.shadowColor = col; ctx.shadowBlur = 10; }
    ctx.fill(); ctx.shadowBlur = 0;
    ctx.strokeStyle = col; ctx.lineWidth = i === 0 ? 1.5 : 0.5; ctx.stroke();
    if (i === 0) {
      ctx.fillStyle = '#666'; ctx.font = '7px Courier New'; ctx.textAlign = 'left';
      ctx.fillText('ANY', colX[3] + 10, y + 3); ctx.textAlign = 'center';
    }
  }
  ctx.fillStyle = '#00ccff'; ctx.font = '10px Courier New';
  ctx.fillText(decHex, colX[3], 40 + N_DEC * 28 + 15);

  // ── Legend ────────────────────────────────────────────────
  const legY = NET_H - 50;
  ctx.font = '8px Courier New'; ctx.textAlign = 'left';
  [
    ['#ff6600', 'DN active'],
    ['#ff33cc', 'Noise suppressing'],
    ['#ff3333', 'Inhibition active'],
    ['#00ff41', 'L1 spike'],
    ['#ffffff', 'DEC any-fire'],
    ['#00ccff', 'DEC unit'],
  ].forEach(([color, label], i) => {
    const x = 15, y = legY + i * 11;
    ctx.fillStyle = color; ctx.fillRect(x, y - 4, 6, 6);
    ctx.fillStyle = '#444'; ctx.fillText(label, x + 10, y + 2);
  });
}
