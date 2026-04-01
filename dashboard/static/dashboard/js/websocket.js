/**
 * websocket.js — WebSocket connection manager and message dispatcher.
 *
 * Connects to the Django Channels proxy at /ws/stream/ (which itself
 * connects to the asyncio pipeline server).
 *
 * Usage:
 *   import { connect, sendCmd, onMessage } from './websocket.js';
 *   onMessage('stream', handler);   // stream data frames
 *   onMessage('status', handler);   // server status replies
 *   connect();
 */

const WS_URL = `ws://${location.host}/ws/stream/`;
const RECONNECT_DELAY_MS = 2000;

let ws = null;
const _handlers = {};  // type → [fn, ...]

/** Register a handler for a message type. */
export function onMessage(type, fn) {
  if (!_handlers[type]) _handlers[type] = [];
  _handlers[type].push(fn);
}

/** Send a command JSON object to the pipeline. */
export function sendCmd(key, value) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ [key]: value }));
  }
}

/** Send a raw JSON object. */
export function sendRaw(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function _dispatch(msg) {
  // Route message to registered handlers
  const type = _classifyMsg(msg);
  const fns = _handlers[type] || [];
  fns.forEach(fn => fn(msg));
  // Always fire wildcard handlers
  (_handlers['*'] || []).forEach(fn => fn(msg, type));
}

function _classifyMsg(msg) {
  if (msg.samples !== undefined || msg.spikes !== undefined) return 'stream';
  if (msg.mode_change !== undefined)                          return 'mode_change';
  if (msg.files !== undefined)                               return 'files';
  if (msg.status !== undefined)                              return 'status';
  return 'unknown';
}

/** Update the nav bar WS status indicator. */
function _setNavStatus(text, cssClass) {
  const el = document.getElementById('ws-status-nav');
  if (!el) return;
  el.textContent = text;
  el.className = 'ws-status ' + cssClass;
}

export function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    _setNavStatus('● pipeline: connected', 'connected');
    (_handlers['open'] || []).forEach(fn => fn());
  };

  ws.onclose = () => {
    _setNavStatus('● pipeline: reconnecting…', '');
    setTimeout(connect, RECONNECT_DELAY_MS);
  };

  ws.onerror = () => ws.close();

  ws.onmessage = ({ data }) => {
    let msg;
    try { msg = JSON.parse(data); }
    catch { return; }
    _dispatch(msg);
  };
}
