// Visual slice editor for the "Compose Reference Audio (slices)" node
// (LTXLoadComposeReferenceAudio). Draws the input clip's waveform (emitted by the Python node via
// the UI channel after a Queue) and lets you add / drag / resize / delete slice regions; serializes
// them into the node's hidden-ish `segments` String widget as [{start_sec,end_sec,gain}].
//
// Pattern cloned from ComfyUI-LTXVideo's sparse_track_editor.js (addDOMWidget canvas + onExecuted
// data channel + serialize via a String widget). v1: SELECTION only (gain fixed 1.0, the
// in-distribution part); per-slice gain editing is a follow-up.
import { app } from "../../../scripts/app.js";

const NODE = "LTXLoadComposeReferenceAudio";
const MAX_TOTAL_SEC = 5.0; // soft budget; over this the readout warns (off-distribution territory)
const EDGE_PX = 6; // grab tolerance for slice edges
const log = (...a) => console.log("[LTXCompose]", ...a);

function segmentsWidget(node) {
  return node.widgets?.find((w) => w.name === "segments");
}
function readSlices(node) {
  const w = segmentsWidget(node);
  try {
    const v = JSON.parse(w?.value || "[]");
    if (!Array.isArray(v)) return [];
    return v.map((s) => ({ start: +s.start_sec || 0, end: +s.end_sec || 0, gain: s.gain ?? 1.0 }));
  } catch {
    return [];
  }
}
function writeSlices(node, slices) {
  const w = segmentsWidget(node);
  if (!w) return;
  const segs = slices
    .filter((s) => s.end > s.start)
    .slice()
    .sort((a, b) => a.start - b.start)
    .map((s) => ({
      start_sec: Math.round(s.start * 1000) / 1000,
      end_sec: Math.round(s.end * 1000) / 1000,
      gain: s.gain ?? 1.0,
    }));
  w.value = JSON.stringify(segs);
  node.setDirtyCanvas?.(true, true);
}

function initEditor(node) {
  const st = (node._compose = { peaks: [], duration: 0, sr: 0, slices: readSlices(node), drag: null });

  const container = document.createElement("div");
  container.style.cssText = "width:100%;height:170px;position:relative;box-sizing:border-box;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText =
    "width:100%;height:100%;display:block;border-radius:6px;background:#17171c;cursor:crosshair;touch-action:none;";
  container.appendChild(canvas);
  node._composeCanvas = canvas;

  const widget = node.addDOMWidget("compose_editor", "ComposeEditor", container, { serialize: false });
  widget.computeSize = () => [node.size?.[0] ?? 320, 180];

  const PAD = 8;
  const cw = () => canvas.clientWidth || 320;
  const ch = () => canvas.clientHeight || 170;
  const xToSec = (x) => (st.duration > 0 ? Math.max(0, Math.min(st.duration, ((x - PAD) / (cw() - 2 * PAD)) * st.duration)) : 0);
  const secToX = (s) => (st.duration > 0 ? PAD + (s / st.duration) * (cw() - 2 * PAD) : PAD);

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = cw(), h = ch();
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(h * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const mid = h * 0.5;

    if (st.peaks.length) {
      ctx.fillStyle = "#4a5568";
      const n = st.peaks.length;
      const bw = (w - 2 * PAD) / n;
      for (let i = 0; i < n; i++) {
        const x = PAD + i * bw;
        const a = st.peaks[i] * (h * 0.42);
        ctx.fillRect(x, mid - a, Math.max(1, bw), a * 2 || 1);
      }
    } else {
      ctx.fillStyle = "#888";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Queue once to load the waveform, then click/drag to add slices", w / 2, mid);
    }

    let total = 0;
    for (const s of st.slices) {
      total += Math.max(0, s.end - s.start);
      const x0 = secToX(s.start), x1 = secToX(s.end);
      ctx.fillStyle = "rgba(46,125,50,0.28)";
      ctx.fillRect(x0, 2, x1 - x0, h - 4);
      ctx.strokeStyle = "#43a047";
      ctx.lineWidth = 2;
      ctx.strokeRect(x0 + 1, 3, x1 - x0 - 2, h - 6);
    }

    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    ctx.fillStyle = total > MAX_TOTAL_SEC ? "#ef5350" : "#9e9e9e";
    const over = total > MAX_TOTAL_SEC ? "  (over ~5s budget: off-distribution)" : "";
    ctx.fillText(`${st.slices.length} slice(s), ${total.toFixed(2)}s total${over}`, PAD, h - 5);
  }
  node._composeDraw = draw;

  function hit(px) {
    for (let i = st.slices.length - 1; i >= 0; i--) {
      const s = st.slices[i], x0 = secToX(s.start), x1 = secToX(s.end);
      if (px >= x0 - EDGE_PX && px <= x1 + EDGE_PX) {
        if (Math.abs(px - x0) <= EDGE_PX) return { idx: i, edge: "l" };
        if (Math.abs(px - x1) <= EDGE_PX) return { idx: i, edge: "r" };
        if (px > x0 && px < x1) return { idx: i, edge: "body" };
      }
    }
    return null;
  }

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    if (st.duration <= 0) return; // nothing loaded yet
    const r = canvas.getBoundingClientRect();
    const px = e.clientX - r.left;
    const h = hit(px);
    if (e.button === 2) {
      if (h) { st.slices.splice(h.idx, 1); writeSlices(node, st.slices); draw(); }
      e.preventDefault();
      return;
    }
    if (h) {
      st.drag = { idx: h.idx, edge: h.edge, px0: px, s0: { ...st.slices[h.idx] } };
    } else {
      const start = xToSec(px);
      const end = Math.min(st.duration, start + 2.0); // default ~2s, then drag the right edge
      st.slices.push({ start, end, gain: 1.0 });
      st.drag = { idx: st.slices.length - 1, edge: "r", px0: px, s0: { start, end } };
    }
    canvas.setPointerCapture(e.pointerId);
    draw();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!st.drag) return;
    const r = canvas.getBoundingClientRect();
    const dSec = xToSec(e.clientX - r.left) - xToSec(st.drag.px0);
    const s = st.slices[st.drag.idx], s0 = st.drag.s0;
    if (st.drag.edge === "l") s.start = Math.min(s.end - 0.05, Math.max(0, s0.start + dSec));
    else if (st.drag.edge === "r") s.end = Math.max(s.start + 0.05, Math.min(st.duration, s0.end + dSec));
    else {
      const len = s0.end - s0.start;
      let ns = Math.max(0, Math.min(st.duration - len, s0.start + dSec));
      s.start = ns; s.end = ns + len;
    }
    draw();
  });
  const endDrag = () => { if (st.drag) { st.drag = null; writeSlices(node, st.slices); draw(); } };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  try { new ResizeObserver(() => draw()).observe(container); } catch {}
  if (node.size) node.size[1] = Math.max(node.size[1] || 0, 400);
  requestAnimationFrame(() => { draw(); app.graph?.setDirtyCanvas(true, true); });
  log("editor attached to node", node.id, "existing slices:", st.slices.length);
}

app.registerExtension({
  name: "AudioLoopHelper.ComposeReferenceAudio",
  async nodeCreated(node) {
    if (node.comfyClass !== NODE) return;
    initEditor(node);
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;
    const origExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (data) {
      origExecuted?.apply(this, arguments);
      const raw = data?.ltxcompose?.[0];
      if (!raw) return;
      try {
        const p = JSON.parse(raw);
        const st = this._compose;
        if (st) {
          st.peaks = p.peaks || [];
          st.duration = p.duration || 0;
          st.sr = p.sr || 0;
          this._composeDraw?.();
          log("waveform loaded:", st.duration.toFixed(2) + "s", st.peaks.length, "pts");
        }
      } catch (err) {
        log("onExecuted parse error", err);
      }
    };
    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      origConfigure?.apply(this, arguments);
      const st = this._compose;
      if (st) { st.slices = readSlices(this); this._composeDraw?.(); }
    };
  },
});
