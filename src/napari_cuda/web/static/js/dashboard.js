(() => {
  'use strict';

  const REFRESH_MS = (() => {
    const s = document.currentScript?.getAttribute('data-refresh-ms');
    const v = parseInt(s || '1000', 10);
    return Number.isFinite(v) && v > 0 ? v : 1000;
  })();

  const $ = (id) => document.getElementById(id);
  const updatedEl = $('updated');
  const refreshEl = $('refreshMs');
  refreshEl.textContent = String(REFRESH_MS);

  // rolling arrays for charting
  const MAX_POINTS = 180; // ~3 minutes at 1 Hz
  const series = {
    total: [],
    encode: [],
    map: [],
  };

  function pushSeries(arr, val) {
    arr.push(val);
    if (arr.length > MAX_POINTS) arr.shift();
  }

  function drawSpark(canvasId, data, color) {
    const canvas = $(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    if (!data || data.length === 0) return;
    const n = data.length;
    const max = Math.max(...data, 1);
    const min = Math.min(...data, 0);
    const span = Math.max(max - min, 1e-6);
    const dx = W / Math.max(n - 1, 1);
    if (n === 1) {
      const y = H - ((data[0] - min) / span) * (H - 6) - 3;
      ctx.beginPath();
      ctx.arc(0, y, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    } else {
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = i * dx;
        const y = H - ((data[i] - min) / span) * (H - 6) - 3; // padding
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }

  function fmtInt(v) {
    return Number.isFinite(v) ? Math.round(v).toLocaleString() : '—';
  }
  function fmtFloat(v, d=1) {
    return Number.isFinite(v) ? v.toFixed(d) : '—';
  }

  function setText(id, text) {
    const el = $(id);
    if (el) el.textContent = text;
  }

  async function fetchMetrics() {
    const res = await fetch('/metrics.json', { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    return res.json();
  }

  function updateTiles(m) {
    // derived
    const fps = m?.derived?.fps;
    setText('fps', fmtFloat(fps, 1));

    // gauges and counters with known names
    const qd = m?.gauges?.napari_cuda_frame_queue_depth ?? m?.gauges?.napari_cuda_capture_queue_depth;
    setText('queueDepth', fmtInt(qd));

    const frames = m?.counters?.napari_cuda_frames_total;
    setText('frames', fmtInt(frames));

    const dropped = m?.counters?.napari_cuda_frames_dropped;
    setText('dropped', fmtInt(dropped));

    const errors = m?.counters?.napari_cuda_encode_errors;
    setText('errors', fmtInt(errors));

    const clients = m?.gauges?.napari_cuda_pixel_clients;
    setText('pixelClients', fmtInt(clients));

    const bytes = m?.counters?.napari_cuda_bytes_total;
    const mb = Number.isFinite(bytes) ? (bytes / (1024 * 1024)) : NaN;
    setText('dataMB', fmtFloat(mb, 1));

    setText('updated', new Date((m?.ts ?? Date.now()) * 1000).toLocaleTimeString());
  }

  function updateCharts(m) {
    const h = m?.histograms || {};
    const total = h['napari_cuda_total_ms']?.mean_ms ?? h['napari_cuda_end_to_end_ms']?.mean_ms;
    const encode = h['napari_cuda_encode_ms']?.mean_ms;
    const map = h['napari_cuda_map_ms']?.mean_ms;

    pushSeries(series.total, Number.isFinite(total) ? total : 0);
    pushSeries(series.encode, Number.isFinite(encode) ? encode : 0);
    pushSeries(series.map, Number.isFinite(map) ? map : 0);

    drawSpark('chart_total', series.total, '#4f46e5');
    drawSpark('chart_encode', series.encode, '#16a34a');
    drawSpark('chart_map', series.map, '#dc2626');
  }

  let lastErr = 0;
  async function tick() {
    try {
      const m = await fetchMetrics();
      updateTiles(m);
      updateCharts(m);
    } catch (e) {
      const now = Date.now();
      if (now - lastErr > 5000) {
        // Log to browser console only, never to server
        console.warn('dashboard update error', e);
        lastErr = now;
      }
    } finally {
      setTimeout(tick, REFRESH_MS);
    }
  }

  tick();
})();
