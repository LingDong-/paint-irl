// App shell: input, HUD, frame loop, replay, save/load/export, and scripted test renders (?test=name).
import { initGPU, G, Engine, PPM, warmCodec } from './engine.js'
import { Scene, hitPlane, TABLE_Z, BRUSH_ORDER } from './scene.js'

const $ = id => document.getElementById(id)
const params = new URLSearchParams(location.search)
const TEST = params.get('test')
if (TEST) {
  // forward logs to the dev server (tool/serve.js)
  for (const k of ['log', 'error', 'warn']) { const f = console[k]; console[k] = (...a) => { f(...a); fetch('/log', { method: 'POST', body: a.map(String).join(' ') }).catch(() => { }) } }
  addEventListener('error', e => console.error('error:', e.message, e.filename + ':' + e.lineno))
  addEventListener('unhandledrejection', e => console.error('rejection:', e.reason?.stack || e.reason))
}

try { await initGPU() } catch (e) {
  $('err').style.display = 'grid'; $('hud').style.display = 'none'
  $('err').textContent = e.message + ' Try a recent Chrome or Edge.'
  throw e
}
const eng = new Engine(), sc = new Scene(eng)
const cv = $('cv'), ctx = cv.getContext('webgpu')
ctx.configure({ device: G.dev, format: G.format, viewFormats: [sc.colorFormat], alphaMode: 'opaque' })
window.eng = eng; window.sc = sc

// ---------------------------------------------------------------- state
const t0 = performance.now()
const now = () => (performance.now() - t0) / 1000
const S = {
  mouseDown: false, ending: false, panning: false, spaceDown: false, palDrag: null, last: [0, 0],
  pressure: 0.5, tilt: [0.32, -0.32], roll: 0, rollMode: 0, lastMove: null, mousePressure: 0.5, tabletSeen: false,
  curSample: null, lastAdd: 0, redraw: true, downTime: 0, dryHours: 24, dryPalette: false, previewDirty: false, dryPreview: false,
  replay: null, cropMode: false, cropStart: null, cropSaved: null, fitted: false, edited: false, file: null,
}
const replaySpeeds = [1, 2, 4, 8, 16, 32, Infinity]

// ---------------------------------------------------------------- coordinates
const W = () => cv.clientWidth, H = () => cv.clientHeight
const aspect = () => W() / Math.max(H(), 1)
const ray = p => sc.cam.ray([p[0] / W() * 2 - 1, 1 - p[1] / H() * 2], aspect())
function locate(p, margin = 0) {
  const r = ray(p), pal = eng.palette, can = eng.canvas
  const pq = hitPlane(r, pal.z), c = hitPlane(r, can.z)
  if (pal.z > can.z && pal.contains(pq)) return [pq, 1, pal.z]     // held above the canvas
  if (can.contains(c)) return [c, 0, can.z]
  if (pal.contains(pq)) return [pq, 1, pal.z]
  // just off an edge: stay on that surface while the bristles still reach it
  if (pal.z > can.z && pal.near(pq, margin)) return [pq, 1, pal.z]
  if (can.near(c, margin)) return [c, 0, can.z]
  if (pal.near(pq, margin)) return [pq, 1, pal.z]
  return [hitPlane(r, TABLE_Z), -1, TABLE_Z]
}
function reach() {     // how far off an edge the held brush still paints
  if (sc.held.brush === undefined) return 0
  const b = eng.brushes[sc.held.brush]
  return b.pencil ? 0.3 : 0.45 * Math.max(b.width, b.thick) + 0.2 * b.length
}
function sample(p, t = now()) {
  const [w, s, z] = locate(p, reach())
  sc.cursor = [w[0], w[1], z]
  return { x: w[0], y: w[1], p: S.pressure, tx: S.tilt[0], ty: S.tilt[1], roll: S.roll, t, s }
}

// ---------------------------------------------------------------- input
function readPen(ev, down) {
  if (ev.pointerType === 'pen') {
    S.tabletSeen = true
    S.pressure = ev.pressure
    // pen axis (tip -> top) projected on the tablet; screen y points down, world y up
    const ax = Math.tan(Math.min(Math.abs(ev.tiltX), 89) * Math.PI / 180) * Math.sign(ev.tiltX), ay = Math.tan(Math.min(Math.abs(ev.tiltY), 89) * Math.PI / 180) * Math.sign(ev.tiltY)
    const l = Math.hypot(ax, ay, 1), tl = [ax / l, -ay / l]
    S.tilt = Math.hypot(...tl) > 0.05 ? [tl[0] * 0.9, tl[1] * 0.9] : [0.32, -0.32]
    if (ev.twist) S.roll = -ev.twist * Math.PI / 180
  } else {
    // mouse: ease pressure in over the first ~60 ms like a hand settling the brush
    S.pressure = down ? S.mousePressure * Math.min(1, 0.35 + (now() - S.downTime) / 0.06) : 0
  }
}
const pos = ev => [ev.clientX, ev.clientY]
cv.addEventListener('contextmenu', e => e.preventDefault())
cv.addEventListener('pointerdown', ev => {
  try { cv.setPointerCapture(ev.pointerId) } catch { }
  const p = pos(ev)
  S.last = p
  closeMenu()
  if (ev.button === 2 || ev.button === 1) { S.panning = true; return }
  if (ev.button !== 0) return
  if (S.spaceDown || S.replay || S.loading) { S.panning = S.spaceDown; return }
  if (S.cropMode) { S.cropStart = eng.canvas.toTexel(hitPlane(ray(p), eng.canvas.z)); S.cropSaved = [...eng.canvas.mask]; return }
  const pal = eng.palette, pq = hitPlane(ray(p), pal.z), ht = pal.toTexel(pq)
  if (Math.hypot(ht[0] - pal.hole[0], ht[1] - pal.hole[1]) < pal.hole[2]) { S.palDrag = { start: pq, origin: [...pal.origin], z: pal.z }; cv.style.cursor = 'grabbing'; return }
  if (!pal.contains(pq)) { const h = sc.pick(ray(p)); if (h) { choose(h); return } }
  if (sc.held.brush === undefined && sc.held.tube === undefined) return
  if (eng.cur) { eng.end(); S.ending = false }      // the last stroke's tail is still being simulated
  S.downTime = now()
  readPen(ev, true)
  const s = sample(p)
  S.mouseDown = true
  if (sc.held.brush !== undefined) eng.begin({ t: 'stroke', b: sc.held.brush, ss: [] })
  else eng.begin({ t: 'squeeze', b: sc.held.tube, ss: [] })
  eng.add(s); S.lastAdd = s.t
  S.lastMove = [s.x, s.y]
})
cv.addEventListener('pointermove', ev => {
  const p = pos(ev)
  if (S.panning) { pan(p); return }
  if (S.palDrag) { const d = S.palDrag, q = hitPlane(ray(p), d.z); eng.placePalette([d.origin[0] + q[0] - d.start[0], d.origin[1] + q[1] - d.start[1]]); return }
  if (S.cropMode && S.cropStart) { eng.canvas.mask = cropRect(S.cropStart, eng.canvas.toTexel(hitPlane(ray(p), eng.canvas.z))); eng.canvas.allDirty(); return }
  if (!S.mouseDown) {
    if (ev.pointerType === 'pen') readPen(ev, false)
    sample(p); S.last = p
    return
  }
  // full-rate samples from pens and fast mice
  const co = ev.getCoalescedEvents?.()
  for (const e of co?.length ? co : [ev]) {
    readPen(e, true)
    const s = sample(pos(e), (e.timeStamp - t0) / 1000)
    if (S.lastMove && S.rollMode < 2 && sc.held.brush !== undefined && !eng.brushes[sc.held.brush].round) {
      const d = [s.x - S.lastMove[0], s.y - S.lastMove[1]]
      if (Math.hypot(...d) > 0.05) { S.roll = Engine.followRoll(S.roll, d, S.rollMode === 1); s.roll = S.roll }
    }
    S.lastMove = [s.x, s.y]
    eng.add(s); S.lastAdd = s.t; S.curSample = s
  }
  S.last = p
})
const cropRect = (a, b) => {
  const c = eng.canvas, cl = (v, m) => Math.round(Math.min(Math.max(v, 0), m))
  return [cl(Math.min(a[0], b[0]), c.w), cl(Math.min(a[1], b[1]), c.h), cl(Math.max(a[0], b[0]), c.w), cl(Math.max(a[1], b[1]), c.h)]
}
cv.addEventListener('pointerup', ev => {
  if (S.panning) { S.panning = false; return }
  if (S.palDrag) {
    const o = eng.palette.origin, d = S.palDrag
    eng.placePalette(d.origin)
    if (o[0] !== d.origin[0] || o[1] !== d.origin[1]) eng.movePalette(o)
    S.palDrag = null; cv.style.cursor = ''; return
  }
  if (S.cropMode && S.cropStart) {
    let r = cropRect(S.cropStart, eng.canvas.toTexel(hitPlane(ray(pos(ev)), eng.canvas.z)))
    eng.canvas.mask = S.cropSaved
    if (r[2] - r[0] < 50 || r[3] - r[1] < 50) r = [0, 0, eng.canvas.w, eng.canvas.h]     // a click resets the crop
    if (r.some((v, i) => v !== S.cropSaved[i])) eng.crop(r)
    eng.canvas.allDirty()
    S.cropStart = null; S.cropMode = false; status()
    return
  }
  if (S.mouseDown) {
    S.mouseDown = false; S.ending = true
    const c = eng.cur
    if (!S.tabletSeen && c && c.op.t === 'stroke') for (const m of Engine.liftSamples(c.op.ss)) eng.add(m)
  }
})
cv.addEventListener('pointercancel', () => { if (S.mouseDown) { S.mouseDown = false; S.ending = true } S.panning = false })
function pan(p) {
  S.userView = true
  const a = hitPlane(ray(S.last), 0), b = hitPlane(ray(p), 0)
  sc.cam.center = [sc.cam.center[0] - (b[0] - a[0]), sc.cam.center[1] - (b[1] - a[1])]
  S.last = p
}
cv.addEventListener('wheel', ev => {
  ev.preventDefault()
  const p = pos(ev)
  // pinch arrives as ctrl+wheel; a mouse wheel moves in coarse steps; trackpad scrolling pans
  const mouseWheel = ev.deltaMode === 1 || (ev.deltaX === 0 && Number.isInteger(ev.deltaY) && Math.abs(ev.deltaY) >= 50)
  if (ev.ctrlKey) zoom(Math.exp(ev.deltaY * 0.01), p)
  else if (mouseWheel) zoom(Math.exp(Math.sign(ev.deltaY) * 0.12), p)
  else { const k = sc.cam.dist / H() * 0.45; sc.cam.center = [sc.cam.center[0] + ev.deltaX * k, sc.cam.center[1] - ev.deltaY * k] }
}, { passive: false })
function zoom(f, p) {
  S.userView = true
  const before = hitPlane(ray(p), 0)
  sc.cam.dist = Math.min(Math.max(sc.cam.dist * f, 40), 2200)
  const after = hitPlane(ray(p), 0)
  sc.cam.center = [sc.cam.center[0] + before[0] - after[0], sc.cam.center[1] + before[1] - after[1]]
}
const fit = () => sc.fitDesk(aspect()), fitCanvas = () => { S.userView = true; sc.fitCanvas(aspect()) }

addEventListener('keydown', ev => {
  if (ev.target.tagName === 'INPUT' && ev.target.type !== 'checkbox' && ev.target.type !== 'range') return
  const cmd = ev.metaKey || ev.ctrlKey, k = ev.key
  if (k === ' ') { S.spaceDown = true; ev.preventDefault(); return }
  if (cmd) {
    const lk = k.toLowerCase()
    if (lk === 'z') ev.shiftKey ? eng.redo() : eng.undo()
    else if (lk === 's') save(ev.shiftKey)
    else if (lk === 'o') open()
    else if (lk === 'e') exportPNG()
    else return
    ev.preventDefault(); status(); return
  }
  if (S.replay) {
    if (k === 'Escape') stopReplay()
    else if (k === 'ArrowLeft' || k === 'ArrowDown' || k === '-') replaySpeed(-1)
    else if (k === 'ArrowRight' || k === 'ArrowUp' || k === '=') replaySpeed(1)
    else if (k === 'f') fit(); else if (k === 'g') fitCanvas()
    return
  }
  if (k >= '0' && k <= '9') choose({ brush: BRUSH_ORDER[(+k + 9) % 10] })
  else if (k === '[') { S.roll += Math.PI / 12; S.rollMode = 2 }
  else if (k === ']') { S.roll -= Math.PI / 12; S.rollMode = 2 }
  else if (k === '\\') S.rollMode = (S.rollMode + 1) % 3
  else if (k === '-') S.mousePressure = Math.max(0.1, +(S.mousePressure - 0.1).toFixed(1))
  else if (k === '=') S.mousePressure = Math.min(1, +(S.mousePressure + 0.1).toFixed(1))
  else if (k === 'h') sc.hideMode = (sc.hideMode + 1) % 3
  else if (k === 'c') eng.cleanBrush(eng.active)
  else if (k === 'f') fit()
  else if (k === 'g') fitCanvas()
  else if (k === 'd') toggleDry()
  else if (k === 'Escape') { if (!S.dryPreview && !S.cropMode) choose({}); setDry(false); S.cropMode = false; closeMenu() }
  else return
  status()
})
addEventListener('keyup', ev => { if (ev.key === ' ') S.spaceDown = false })
for (const t of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'keydown', 'keyup', 'click', 'input', 'change']) addEventListener(t, () => { S.redraw = true }, true)
addEventListener('beforeunload', e => { if (S.edited && !TEST) e.preventDefault() })
eng.onChange = () => { S.edited = true }

function choose(h) {
  sc.setHeld(h)
  if (h.brush !== undefined) eng.active = h.brush
  status()
}
function status() {
  let s
  if (sc.held.brush !== undefined) {
    const b = eng.brushes[sc.held.brush]
    s = b.name + (S.tabletSeen ? '' : `   ·   mouse pressure ${S.mousePressure.toFixed(1)} (−/=)`)
    if (!b.round) s += '   ·   ' + ['follows the stroke', 'edgewise', 'fixed angle'][S.rollMode] + ' (\\)'
  } else if (sc.held.tube !== undefined) s = eng.tubes[sc.held.tube].name + ' — drag on the palette to squeeze; click its slot or Esc to put it back'
  else s = 'Pick up a brush, the knife, the pencil or a tube'
  if (sc.hideMode > 0) s += sc.hideMode === 1 ? '   ·   handle hidden (H)' : '   ·   brush hidden (H)'
  if (S.dryPreview) s = 'Drying preview — blue: still wet, yellow: tacky, red: dry'
  if (S.cropMode) s = 'Drag a rectangle on the canvas to crop (click to reset), Esc to cancel'
  if (S.replay) { const v = replaySpeeds[S.replay.speed]; s = 'Replaying ' + (v === Infinity ? 'at full speed' : '×' + v) + '   ·   ←/→ speed, Esc to stop' }
  $('status').textContent = s
  document.querySelector('[data-a=dry]').classList.toggle('on', S.dryPreview)
  document.querySelector('[data-a=crop]').classList.toggle('on', S.cropMode)
  document.querySelector('[data-a=replay]').classList.toggle('on', !!S.replay)
}

// ---------------------------------------------------------------- HUD
const hours = v => Math.pow(v, 2.2) * 240
function dryChanged() {
  const h = hours(+$('dryHours').value)
  $('dryVal').textContent = h < 1 ? `${Math.round(h * 60)} min` : h < 48 ? `${Math.round(h)} h` : `${(h / 24).toFixed(1)} days`
  S.dryHours = h; S.previewDirty = true
}
function setDry(on) { S.dryPreview = on; sc.heat = on; S.previewDirty = true; $('dry').classList.toggle('open', on); status() }
const toggleDry = () => setDry(!S.dryPreview)
$('dryHours').oninput = dryChanged
$('dryPal').onchange = e => { S.dryPalette = e.target.checked; S.previewDirty = true }
$('dryDull').onchange = e => { eng.dullDry = e.target.checked }
$('dryGo').onclick = () => { eng.dry(S.dryHours, S.dryPalette); setDry(false) }
dryChanged()
const actions = {
  undo: () => eng.undo(), redo: () => eng.redo(), dry: toggleDry, clean: () => eng.cleanBrush(eng.active), scrape: () => eng.clearSurface(true),
  handle: () => { sc.hideMode = (sc.hideMode + 1) % 3 }, crop: () => { S.cropMode = !S.cropMode }, new: e => openMenu(e.currentTarget),
  replay: startReplay, save: () => save(false), open, export: exportPNG,
}
for (const b of document.querySelectorAll('[data-a]')) b.onclick = e => { actions[b.dataset.a](e); status(); b.blur() }

let portrait = false, texture = 0
function openMenu(anchor) {
  const m = $('menu'), r = anchor.getBoundingClientRect()
  const item = (label, on, f) => { const d = document.createElement('div'); d.textContent = label; if (on) d.className = 'on'; d.onclick = f; m.append(d) }
  m.innerHTML = ''
  for (const [w, h] of [[24, 18], [30, 24], [40, 30], [50, 40]]) {
    const [a, b] = portrait ? [h, w] : [w, h]
    item(`${a} × ${b} cm`, false, () => { closeMenu(); newCanvas(a, b) })
  }
  m.append(document.createElement('hr'))
  item('Portrait', portrait, () => { portrait = !portrait; openMenu(anchor) })
  for (const [i, name] of [[0, 'Cotton canvas'], [2, 'Fine linen'], [3, 'Smooth gessoed panel']]) item(name, texture === i, () => { texture = i; openMenu(anchor) })
  m.append(document.createElement('hr'))
  item('Wipe this canvas', false, () => { closeMenu(); eng.clearSurface(false) })
  m.style.left = r.left + 'px'; m.style.top = r.bottom + 6 + 'px'
  m.classList.add('open')
}
function closeMenu() { $('menu').classList.remove('open') }
function newCanvas(w, h) {
  if (eng.history.length && !confirm('Start a new painting? The current painting and its history will be discarded.')) return
  eng.newCanvas(w * 10, h * 10, texture)
  fit(); S.file = null; S.edited = false; document.title = 'PaintIRL'
}

// ---------------------------------------------------------------- files
async function pickSave(name, ext, desc) {
  if (window.showSaveFilePicker) return await showSaveFilePicker({ suggestedName: name, types: [{ description: desc, accept: { 'application/octet-stream': ['.' + ext] } }] })
  return null
}
function download(blob, name) { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000) }
async function save(as) {
  if (S.replay) return
  try {
    let h = S.file
    if (as || !h) h = await pickSave((S.file?.name || 'Painting.pirl'), 'pirl', 'Painting')
    const blob = await eng.encode()
    if (h) { const w = await h.createWritable(); await w.write(blob); await w.close(); S.file = h; document.title = h.name.replace(/\.pirl$/, '') }
    else download(blob, 'Painting.pirl')
    S.edited = false
  } catch (e) { if (e.name !== 'AbortError') alert('Could not save: ' + e.message) }
}
async function open() {
  if (S.replay) return
  if (S.edited && !confirm('Discard unsaved changes?')) return
  try {
    if (window.showOpenFilePicker) {
      const [h] = await showOpenFilePicker({ types: [{ description: 'Painting', accept: { 'application/octet-stream': ['.pirl'] } }] })
      await load(await (await h.getFile()).arrayBuffer()); S.file = h; document.title = h.name.replace(/\.pirl$/, '')
    } else $('file').click()
  } catch (e) { if (e.name !== 'AbortError') alert('Could not open: ' + e.message) }
}
$('file').onchange = async e => { const f = e.target.files[0]; if (f) { try { await load(await f.arrayBuffer()) } catch (err) { alert('Could not open: ' + err.message) } } e.target.value = '' }
async function load(ab) {
  S.loading = true; $('status').textContent = 'Opening…'
  try { await eng.decode(ab) } finally { S.loading = false }
  S.edited = false; S.fitted = false; choose({ brush: eng.active })
}
async function exportPNG() {
  const img = await sc.exportImage(1)
  if (!img) return
  const c = new OffscreenCanvas(img.width, img.height); c.getContext('2d').putImageData(img, 0, 0)
  const blob = await c.convertToBlob({ type: 'image/png' })
  try {
    const h = await pickSave('Painting.png', 'png', 'PNG image')
    if (h) { const w = await h.createWritable(); await w.write(blob); await w.close() } else download(blob, 'Painting.png')
  } catch (e) { if (e.name !== 'AbortError') alert(e.message) }
}

// ---------------------------------------------------------------- replay
async function startReplay() {
  if (S.replay) { stopReplay(); return }
  const ops = eng.history
  if (!ops.length || S.cropMode) return
  const snap = await eng.encode()        // the finished painting, restored when the replay is stopped
  eng.resetAll(); eng.flushAll(); eng.history = []; eng.redoOps = []
  S.replay = { ops, i: 0, k: 0, started: false, t0: now(), speed: 0, snapshot: snap, edited: S.edited }
  status()
}
async function stopReplay() {
  const r = S.replay
  if (!r) return
  S.replay = null
  eng.cur = null; eng.undoRec = null
  S.loading = true
  try { await eng.decode(await r.snapshot.arrayBuffer()) } finally { S.loading = false }
  S.edited = r.edited
  sc.setHeld({ brush: eng.active }); status()
}
function replaySpeed(d) { if (S.replay) { S.replay.speed = Math.min(Math.max(S.replay.speed + d, 0), replaySpeeds.length - 1); status() } }
function stepReplay() {
  const r = S.replay
  if (!r) return
  const speed = replaySpeeds[r.speed], fast = speed === Infinity, ts = performance.now()
  const next = () => { r.i++; r.k = 0; r.started = false; r.t0 = now() }
  // at full speed, run as many ops as fit in ~30 ms per frame
  while (r.i < r.ops.length) {
    const op = r.ops[r.i]
    if (op.ss) {
      if (!r.started) {
        eng.begin({ t: op.t, b: op.b, ss: [] })
        if (op.t === 'stroke') { sc.setHeld({ brush: op.b }); eng.active = op.b } else sc.setHeld({ tube: op.b })
        r.started = true; r.t0 = now()
      }
      const base = op.ss[0]?.t ?? 0, clock = fast ? Infinity : (now() - r.t0) * speed
      while (r.k < op.ss.length && op.ss[r.k].t - base <= clock) {
        const m = op.ss[r.k++]
        eng.add(m); sc.cursor = [m.x, m.y, m.s === 1 ? eng.palette.z : 0]
      }
      if (fast) eng.process()
      if (r.k >= op.ss.length && eng.backlog === 0) { eng.end(); next() }
    } else {
      if (op.t === 'dry') eng.dry(op.hours, op.palette)
      else if (op.t === 'clean') eng.cleanBrush(op.b)
      else if (op.t === 'scrape') eng.clearSurface(true)
      else if (op.t === 'wipe') eng.clearSurface(false)
      else if (op.t === 'crop') eng.crop(op.rect)
      else if (op.t === 'move') eng.movePalette([op.x, op.y])
      next()
    }
    if (!fast || eng.backlog > 0 || performance.now() - ts > 30) break
  }
  if (r.i >= r.ops.length) {
    S.replay = null; S.edited = r.edited
    if (sc.held.tube !== undefined) sc.setHeld({ brush: eng.active })
    status()
  }
}

// ---------------------------------------------------------------- frame loop
let inflight = 0
function frame() {
  requestAnimationFrame(frame)
  if (S.loading) return      // nothing to draw while a file is opening (and it would slow the opening down)
  if (inflight >= 2) return
  const dpr = devicePixelRatio, w = Math.max(1, Math.round(W() * dpr)), h = Math.max(1, Math.round(H() * dpr))
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; if (!S.userView) S.fitted = false; S.redraw = true }
  if (!S.fitted && W() > 0) { fit(); S.fitted = true }
  // draw only when something may have changed: input, painting, replay, or surfaces to redraw
  const busy = S.redraw || S.mouseDown || S.ending || eng.cur || S.replay || (S.dryPreview && S.previewDirty) || eng._enc ||
    eng.surfaces.some(s => s.dirty.x1 > s.dirty.x0 && s.dirty.y1 > s.dirty.y0)
  if (!busy) return
  S.redraw = false
  sc.time = now()
  stepReplay()
  if (S.mouseDown && now() - S.lastAdd > 0.012) {
    // keep the brush working while held still
    const s = { ...(S.curSample || sample(S.last)) }
    s.t = now(); s.p = S.tabletSeen ? S.pressure : S.mousePressure * Math.min(1, 0.35 + (now() - S.downTime) / 0.06)
    eng.add(s); S.lastAdd = s.t
  }
  eng.process(14)
  if (S.ending && eng.backlog === 0) { eng.end(); S.ending = false; S.curSample = null }
  if (!eng.cur && sc.held.brush !== undefined) {
    const b = eng.brushes[sc.held.brush]
    b.baseZ = sc.cursor[2]
    b.pose([sc.cursor[0], sc.cursor[1]], 0, S.tilt, S.roll)
    b.simulate(null, 0, null)
  }
  if (S.dryPreview && S.previewDirty) { eng.preview(S.dryHours, S.dryPalette); S.previewDirty = false }
  eng.shade()
  const tex = ctx.getCurrentTexture()
  sc.encode(eng.enc(), sc.target(w, h, tex.createView({ format: sc.colorFormat })), aspect())
  eng.submit()
  inflight++
  G.q.onSubmittedWorkDone().then(() => inflight--)
}

status()
// file loading/saving workers, warmed up in the background while not painting
if (!TEST) setTimeout(() => warmCodec(() => !S.mouseDown && !eng.cur && !S.replay), 2000)
if (!TEST) requestAnimationFrame(frame)
else runTest(TEST).catch(e => console.error('test failed:', e.stack || e)).finally(() => fetch('/done', { method: 'POST' }))

// ---------------------------------------------------------------- scripted tests
async function runTest(name) {
  let t = 0
  const run = async f => { f(); eng.submit(); await G.q.onSubmittedWorkDone() }
  const path = (pts, speed) => {     // [x, y, pressure] polyline -> samples at 120 Hz
    const out = [], dt = 1 / 120
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1], n = Math.max(1, Math.floor(Math.hypot(b[0] - a[0], b[1] - a[1]) / (speed * dt)))
      for (let k = 0; k < n; k++) { const f = k / n; out.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]) }
    }
    out.push(pts[pts.length - 1])
    return out
  }
  const roll = new Array(12).fill(0)
  const stroke = async (bi, s, pts, speed = 100, tilt = [0.3, -0.3]) => {
    eng.begin({ t: 'stroke', b: bi, ss: [] })
    const ps = path(pts, speed)
    for (let i = 0; i < ps.length;) {
      await run(() => {
        for (let k = 0; k < 8 && i < ps.length; k++, i++) {
          t += 1 / 120
          if (i > 0 && !eng.brushes[bi].round) { const d = [ps[i][0] - ps[i - 1][0], ps[i][1] - ps[i - 1][1]]; if (Math.hypot(...d) > 0.05) roll[bi] = Engine.followRoll(roll[bi], d) }
          eng.add({ x: ps[i][0], y: ps[i][1], p: ps[i][2], tx: tilt[0], ty: tilt[1], roll: roll[bi], t, s })
        }
        eng.process()
      })
    }
    for (const m of Engine.liftSamples(eng.cur.op.ss)) eng.add(m)
    await run(() => eng.end())
    t += 0.3
  }
  const squeeze = async (tube, pts, speed = 25) => {
    eng.begin({ t: 'squeeze', b: tube, ss: [] })
    await run(() => { for (const p of path(pts, speed)) { t += 1 / 120; eng.add({ x: p[0], y: p[1], p: p[2], tx: 0, ty: 0, roll: 0, t, s: 1 }) } eng.process() })
    await run(() => eng.end())
  }
  const pal = eng.palette.origin, P = (x, y, p = 1) => [pal[0] + x, pal[1] + y, p]
  const post = async (img, file) => {
    const c = new OffscreenCanvas(img.width, img.height); c.getContext('2d').putImageData(img, 0, 0)
    await fetch('/out/' + file, { method: 'POST', body: await c.convertToBlob({ type: 'image/png' }) })
  }
  const render = async (center, dist, w, h, file) => { sc.cam.center = center; sc.cam.dist = dist; await post(await sc.snapshot(w, h), file) }
  const T0 = performance.now()
  if (name === 'prof') {
    await (await import('./tool/prof.js')).run({ eng, sc, G, Engine, P, pal, params, log: console.log, post })
  } else if (name === 'scene') {
    await squeeze(0, [P(30, 150), P(55, 150)])
    await render([90, -45], 980, 1800, 1200, 'scene.png')
    await render([-185, -158], 40, 1400, 1000, 'scene_close.png')        // canvas corner on the table, closest zoom
  } else if (name === 'strokes') {
    await squeeze(0, [P(30, 150), P(55, 150)]); await squeeze(7, [P(30, 120), P(50, 118)]); await squeeze(2, [P(30, 90), P(48, 92)]); await squeeze(6, [P(30, 60), P(46, 60)])
    // load round 12 with ultramarine
    await stroke(2, 1, [P(36, 112, 0.8), P(40, 125, 0.9), P(46, 112, 0.9), P(52, 124, 0.8)], 40)
    await stroke(2, 0, [[-100, 60, 0.5], [-20, 70, 0.5], [60, 55, 0.5]], 120)
    await stroke(2, 0, [[-100, 30, 0.2], [-20, 38, 0.25], [60, 25, 0.2]], 140)
    await stroke(2, 0, [[-100, 0, 0.9], [-20, 8, 0.9], [60, -5, 0.9]], 100)
    // flat 8: white with a touch of yellow, across the wet blue
    await stroke(4, 1, [P(30, 152, 0.8), P(55, 149, 0.8), P(30, 151, 0.8)], 40)
    await stroke(4, 1, [P(36, 95, 0.5), P(40, 90, 0.5)], 30)
    await stroke(4, 0, [[-60, 80, 0.5], [-55, -30, 0.5]], 90)
    await stroke(4, 0, [[20, 80, 0.3], [25, -30, 0.3]], 120)
    await stroke(4, 0, [[-90, -50, 0.6], [80, -55, 0.6]], 120)
    console.log(`strokes simulated in ${((performance.now() - T0) / 1000).toFixed(2)} s, subgroups ${G.subgroups}`)
    await render([-20, 15], 260, 1600, 1200, 'strokes.png')
    await render([pal[0] + 45, pal[1] + 105], 160, 1200, 1200, 'strokes_pal.png')
    await render([-55, 40], 90, 1400, 1000, 'strokes_zoom.png')
  } else if (name === 'dry') {
    await squeeze(7, [P(30, 120), P(50, 118)]); await squeeze(11, [P(30, 90), P(48, 92)]); await squeeze(0, [P(30, 150), P(55, 150)])
    await stroke(2, 1, [P(36, 112, 0.8), P(40, 125, 0.9), P(46, 112, 0.9)], 40)
    await stroke(2, 0, [[-80, 30, 0.3], [40, 35, 0.3]], 80)          // thin blue
    await stroke(5, 1, [P(34, 96, 0.9), P(44, 86, 0.9), P(36, 92, 0.9)], 40)
    await stroke(5, 0, [[-80, -10, 0.25], [40, -5, 0.25]], 70)       // thick umber (light touch)
    await run(() => eng.preview(20, false))
    sc.heat = true
    await render([-20, 10], 220, 1400, 1000, 'dry_heat.png')
    sc.heat = false
    await run(() => eng.dry(20, false))
    // white strokes across both: blue (dried thin) should resist, umber (still wet) should smear
    await run(() => eng.brushes[4].clean(eng))
    await stroke(4, 1, [P(30, 152, 0.8), P(55, 149, 0.8), P(30, 151, 0.8)], 40)
    await stroke(4, 0, [[-60, 50, 0.5], [-58, -30, 0.5]], 80)
    await render([-20, 10], 220, 1400, 1000, 'dry.png')
    await run(() => eng.undo())
    await render([-20, 10], 220, 1400, 1000, 'dry_undo1.png')
    await run(() => { eng.undo(); eng.undo() })
    await render([-20, 10], 220, 1400, 1000, 'dry_undo3.png')
    await run(() => { eng.redo(); eng.redo(); eng.redo() })
    await render([-20, 10], 220, 1400, 1000, 'dry_redo.png')
  } else if (name === 'paint') {
    let seed = 42
    const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 }
    const gauss = () => Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-7))) * Math.cos(2 * Math.PI * rnd())
    const hand = (a, b, p, n = 6) => Array.from({ length: n + 1 }, (_, k) => { const f = k / n; return [a[0] + (b[0] - a[0]) * f + gauss() * 0.6, a[1] + (b[1] - a[1]) * f + gauss() * 0.6, p * (0.85 + 0.3 * Math.sin(f * 3.1))] })
    await squeeze(0, [P(25, 160), P(60, 160)]); await squeeze(7, [P(25, 135), P(42, 135)]); await squeeze(4, [P(25, 110), P(42, 110)])
    await squeeze(10, [P(25, 85), P(42, 85)]); await squeeze(5, [P(25, 60), P(38, 60)]); await squeeze(2, [P(25, 40), P(38, 40)])
    await stroke(5, 1, [P(40, 160, 0.9), P(52, 160, 0.9), P(40, 161, 0.9)], 40)
    await stroke(5, 1, [P(32, 136, 0.6), P(36, 135, 0.6)], 30)
    for (let k = 0; k < 4; k++) await stroke(5, 1, [P(110, 150 + k, 0.9), P(150, 140, 0.9), P(110, 135 - k, 0.9)], 60)
    for (let row = 0; row < 9; row++) {
      const y = 80 - row * 7
      if (row % 3 === 2) await stroke(5, 1, [P(120, 145, 0.8), P(145, 140, 0.8), P(120, 136, 0.8)], 50)
      if (row === 5) await stroke(5, 1, [P(40, 160, 0.7), P(55, 158, 0.7)], 40)
      await stroke(5, 0, hand([-112 + rnd() * 10, y], [112, y + gauss() * 2], 0.45), 140)
    }
    await stroke(4, 1, [P(32, 112, 0.8), P(38, 108, 0.8), P(30, 86, 0.8), P(38, 84, 0.8)], 40)
    for (let k = 0; k < 4; k++) await stroke(4, 1, [P(110, 100 + k, 0.9), P(140, 90, 0.9), P(110, 85, 0.9)], 60)
    for (let row = 0; row < 7; row++) {
      const y = 14 - row * 13
      if (row % 2 === 1) await stroke(4, 1, [P(32, 110, 0.8), P(36, 104, 0.8), P(32, 88, 0.6)], 40)
      await stroke(4, 0, hand([-115, y], [115, y - 3 + gauss() * 3], 0.55), 120)
    }
    await stroke(1, 1, [P(30, 60, 0.8), P(34, 62, 0.8)], 30)
    for (let k = 0; k < 6; k++) { const c = [-60 + k * 18, -25 + gauss() * 5]; await stroke(1, 0, [[c[0], c[1], 0.6], [c[0] + 3, c[1] + 1.5, 0.7], [c[0] + 5, c[1], 0.4]], 40) }
    await stroke(1, 1, [P(30, 40, 0.8), P(34, 42, 0.8)], 30)
    for (let k = 0; k < 6; k++) { const c = [-50 + k * 18, -45 + gauss() * 5]; await stroke(1, 0, [[c[0], c[1], 0.6], [c[0] + 3, c[1] + 1.5, 0.7], [c[0] + 5, c[1], 0.4]], 40) }
    console.log(`painting simulated in ${((performance.now() - T0) / 1000).toFixed(2)} s`)
    await post(await sc.exportImage(0.6), 'paint.png')
    await render([-40, 10], 120, 1400, 1000, 'paint_zoom.png')
    await render([pal[0] + 70, pal[1] + 110], 330, 1400, 1000, 'paint_pal.png')
    // save, reload into the same engine, compare
    const a = await sc.exportImage(0.25)
    const blob = await eng.encode()
    await run(() => eng.clearSurface(false))
    await eng.decode(await blob.arrayBuffer())
    const b = await sc.exportImage(0.25)
    let bad = 0; for (let i = 0; i < a.data.length; i++) if (Math.abs(a.data[i] - b.data[i]) > 2) bad++
    console.log(`saved ${(blob.size / 1e6).toFixed(1)} MB, reloaded: ${bad} bytes differ, history ${eng.history.length} ops`)
  } else if (name === 'brush') {
    await squeeze(7, [P(30, 120), P(50, 118)]); await squeeze(0, [P(30, 150), P(55, 150)])
    await stroke(2, 1, [P(36, 112, 0.8), P(40, 125, 0.9), P(46, 112, 0.9)], 40)
    await stroke(2, 0, [[-60, 20, 0.5], [0, 25, 0.5]], 100)
    sc.setHeld({ brush: 2 })
    eng.begin({ t: 'stroke', b: 2, ss: [] })
    await run(() => { for (let k = 0; k < 40; k++) eng.add({ x: 10 + k * 0.8, y: 0, p: 0.6, tx: 0.3, ty: -0.3, roll: 0, t: t + k / 120, s: 0 }); eng.process() })
    await render([30, 5], 140, 1400, 1000, 'brush.png')
    await render([pal[0] + 45, pal[1] + 130], 110, 1400, 1000, 'brush_pal.png')
  } else if (name === 'selftest') {
    // drives the real input handlers and frame loop with synthesized pointer events
    requestAnimationFrame(frame)
    const frames = n => new Promise(r => { let k = 0; const f = () => ++k >= n ? r() : requestAnimationFrame(f); requestAnimationFrame(f) })
    await frames(3)
    const ev = (type, w) => { const [x, y] = sc.project(w, [W(), H()]); cv.dispatchEvent(new PointerEvent(type, { clientX: x, clientY: y, button: 0, buttons: type === 'pointerup' ? 0 : 1, pointerId: 1, pointerType: 'mouse', bubbles: true })) }
    const click = async w => { ev('pointerdown', w); ev('pointerup', w); await frames(2) }
    const drag = async (pts, n) => {
      ev('pointerdown', pts[0]); await frames(1)
      for (let k = 1; k <= n; k++) {
        const f = k / n * (pts.length - 1), i = Math.min(Math.floor(f), pts.length - 2), u = f - i
        ev('pointermove', pts[i].map((v, c) => v + (pts[i + 1][c] - v) * u)); await frames(1)
      }
      ev('pointerup', pts[pts.length - 1]); await frames(10)
    }
    const pz = eng.palette.z, Q = (x, y) => [pal[0] + x, pal[1] + y, pz]
    const tube = sc.tubeRest(7).slice(12, 15)
    await click([tube[0], tube[1] - 40, TABLE_Z + 10])
    console.log('held after clicking a tube:', JSON.stringify(sc.held))
    await drag([Q(30, 120), Q(52, 118)], 50)
    choose({ brush: 4 })
    await drag([Q(36, 112), Q(40, 125), Q(46, 112), Q(50, 124)], 60)
    await drag([[-80, 40, 0], [-20, 50, 0], [40, 30, 0]], 50)
    await drag([[-60, -20, 0], [0, 10, 0], [50, -30, 0]], 50)
    choose({ brush: 9 })
    await drag([[-150, 100, 0], [-60, 110, 0], [-40, 60, 0]], 40)
    await drag([Q(38, 222), Q(-212, 232)], 30)        // carry the palette by its thumb hole over the canvas
    console.log('palette moved to', eng.palette.origin.map(v => v.toFixed(1)), 'z', eng.palette.z, 'last op', eng.history.at(-1)?.t)
    const n0 = eng.history.length
    await startReplay(); replaySpeed(9)
    for (let k = 0; k < 600 && S.replay; k++) await frames(1)
    console.log('replay at full speed finished:', !S.replay, 'history', eng.history.length, 'of', n0)
    await frames(5)
    await post(await sc.snapshot(1600, 1000), 'selftest.png')
  } else if (name === 'pirl') {
    // load a .pirl written by the Mac app, render it, save it again and compare the decompressed tiles
    const src = await (await fetch('/' + (params.get('file') || 'pirlv3test.pirl'))).arrayBuffer()
    let t1 = performance.now()
    await warmCodec()
    console.log(`codec warm-up ${((performance.now() - t1) / 1000).toFixed(2)} s (in the app this happens in the background at startup)`)
    // open it as the app does, with the frame loop running (it keeps drawing while the tiles arrive)
    requestAnimationFrame(frame)
    t1 = performance.now()
    await load(src)
    await new Promise(r => setTimeout(r, 500))
    console.log(`loaded in ${((performance.now() - t1) / 1000).toFixed(2)} s: history ${eng.history.length} ops, canvas tiles ${eng.canvas.used}, palette tiles ${eng.palette.used}`)
    await post(await sc.exportImage(0.5), 'pirl.png')
    sc.fitDesk(1.6)
    await render(sc.cam.center, sc.cam.dist, 1600, 1000, 'pirl_desk.png')
    for (const s of eng.surfaces) s.packed.clear()       // time a full encode
    t1 = performance.now()
    const out = await (await eng.encode()).arrayBuffer()
    console.log(`saved in ${((performance.now() - t1) / 1000).toFixed(2)} s: ${(out.byteLength / 1e6).toFixed(1)} MB (original ${(src.byteLength / 1e6).toFixed(1)} MB)`)
    const blocks = ab => { const d = new Uint8Array(ab), dv = new DataView(ab), r = []; for (let o = 8; o + 8 <= d.length;) { const n = Number(dv.getBigUint64(o, true)); r.push(d.subarray(o + 8, o + 8 + n)); o += 8 + n } return r }
    const inflate = async u8 => new Uint8Array(await new Response(new Blob([u8]).stream().pipeThrough(new DecompressionStream('deflate-raw'))).arrayBuffer())
    const tiles = async b => { const dv = new DataView(b.buffer, b.byteOffset), m = new Map(); for (let k = 0, o = 4; k < dv.getUint32(0, true); k++) { const f = dv.getUint32(o, true), l = dv.getUint32(o + 4, true); m.set(f, await inflate(b.subarray(o + 8, o + 8 + l))); o += 8 + l } return m }
    const A = blocks(src), B = blocks(out)
    for (const k of [1, 2]) {
      const ta = await tiles(A[k]), tb = await tiles(B[k])
      let state = 0, height = 0, missing = 0
      for (const [f, a] of ta) {
        const b = tb.get(f)
        if (!b) { missing++; continue }
        for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { if (i < 88 * 16384) state++; else height++ }
      }
      console.log(`${k === 1 ? 'canvas' : 'palette'}: ${ta.size} tiles vs ${tb.size}, missing ${missing}, differing state bytes ${state}, height bytes ${height}`)
    }
    const same = async k => { const a = await inflate(A[k]), b = await inflate(B[k]); return a.length === b.length && a.every((v, i) => v === b[i]) }
    console.log('brushes identical:', await same(3), ' history identical:', await same(4), ' header:', new TextDecoder().decode(B[0]))
  } else if (name === 'perf') {
    await squeeze(0, [P(30, 150), P(55, 150)])
    await stroke(5, 1, [P(30, 152, 0.8), P(55, 149, 0.8)], 40)
    for (const [bi, spd] of [[5, 200], [2, 200], [0, 100], [5, 1500], [2, 1500], [8, 1500]]) {
      const t1 = performance.now()
      await stroke(bi, 0, [[-100, -60, 0.6], [100, 60, 0.6]], spd)
      console.log(`brush ${eng.brushes[bi].name} ${eng.brushes[bi].n} tufts, 233mm at ${spd}mm/s: sim ${((performance.now() - t1) / 1000).toFixed(2)}s for ${(233 / spd).toFixed(2)}s of painting`)
    }
  }
}
