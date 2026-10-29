// Paint engine: pigments, paint surfaces, bristle brushes, tubes, undo, drying, save format. WebGPU port.
export const PPM = 10            // texels per mm
export const UNIT = 0.01         // paint thickness unit (mm)
export const TILE = 64           // paint state is kept in tiles of TILE x TILE texels, allocated where painted
export const NPL = 23            // u32 planes per texel (see shaders.wgsl)
export const TB = TILE * TILE * NPL * 4
export const NCH = 16            // paint channels per texel (pigments + medium)
const FPMAX = 640

const BU = GPUBufferUsage, TU = GPUTextureUsage, SS = GPUShaderStage

// ---------------------------------------------------------------- GPU
export const G = {}
export async function initGPU() {
  const ad = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' })
  if (!ad) throw new Error('WebGPU is not available in this browser.')
  const req = {}
  for (const k of ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeInvocationsPerWorkgroup', 'maxStorageBuffersPerShaderStage',
    'maxStorageBuffersInFragmentStage', 'maxStorageBuffersInVertexStage', 'maxTextureDimension2D']) if (k in ad.limits) req[k] = ad.limits[k]
  const q = new URLSearchParams(location.search)      // ?test=prof: timestamps; &nosub: without subgroups
  let sub = ad.features.has('subgroups') && !q.has('nosub')
  const feats = [...(sub ? ['subgroups'] : []), ...(q.get('test') === 'prof' && ad.features.has('timestamp-query') ? ['timestamp-query'] : [])]
  const dev = await ad.requestDevice({ requiredFeatures: feats, requiredLimits: req })
  dev.lost.then(i => console.error('GPU device lost', i.message))
  dev.addEventListener('uncapturederror', e => console.error('WebGPU:', e.error.message))
  G.dev = dev; G.q = dev.queue; G.limits = dev.limits
  G.format = navigator.gpu.getPreferredCanvasFormat()
  // 1x1 stand-in for textures that are only made when first needed
  G.dummy = dev.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING })

  // shader sections: "//// name deps..."
  const src = await (await fetch('shaders.wgsl')).text()
  const secs = {}
  let cur = null
  for (const line of src.split('\n')) {
    if (line.startsWith('//// ')) { const [n, ...d] = line.slice(5).trim().split(/\s+/); secs[n] = { deps: d, code: '' }; cur = secs[n] }
    else if (cur) cur.code += line + '\n'
  }
  G.code = (name, repl = {}) => {
    const seen = new Set(['common']), out = [secs.common.code]
    const add = n => { if (seen.has(n)) return; seen.add(n); for (const d of secs[n].deps) add(d); out.push(secs[n].code) }
    add(name)
    let c = out.join('\n')
    for (const k in repl) c = c.replaceAll(k, repl[k])
    return c
  }
  const modules = {}
  G.module = (name, repl) => {
    const key = name + JSON.stringify(repl || {})
    if (!modules[key]) modules[key] = dev.createShaderModule({ code: G.code(name, repl), label: name })
    return modules[key]
  }
  // the reservoir reduction uses subgroups when available (falls back to plain atomics)
  const tsrc = G.code('kTransfer')
  const variant = s => s ? 'enable subgroups;\ndiagnostic(off, subgroup_uniformity);\n' + tsrc.replace(/\/\/#else[\s\S]*?\/\/#endif/, '')
    : tsrc.replace(/\/\/#if SUBGROUPS[\s\S]*?\/\/#else/, '').replace('//#endif', '')
  let tm = dev.createShaderModule({ code: variant(sub), label: 'kTransfer' })
  if (sub && (await tm.getCompilationInfo()).messages.some(m => m.type === 'error')) { sub = false; tm = dev.createShaderModule({ code: variant(false), label: 'kTransfer' }) }
  G.subgroups = sub
  G.transferModule = tm
  for (const n of Object.keys(secs)) if (/^k|^fp$|^surf$|^obj$|^shadow$|^hair$/.test(n) && n !== 'kTransfer' && n !== 'kDown') {
    const info = await G.module(n).getCompilationInfo()
    for (const m of info.messages) if (m.type === 'error') console.error(`${n}:${m.lineNum}: ${m.message}`)
  }
  buildLayouts()
}

// bind group layouts from short specs: u uniform, ud dynamic uniform, r/rd read-only storage, s storage,
// f float texture, uf unfilterable texture, d depth texture, w8/w16 storage texture, sl sampler, sc comparison sampler
function bgl(spec, vis) {
  return G.dev.createBindGroupLayout({
    entries: spec.map((s, i) => {
      const e = { binding: i, visibility: vis }
      if (s === 'u' || s === 'ud') e.buffer = { type: 'uniform', hasDynamicOffset: s === 'ud' }
      else if (s === 'r' || s === 'rd') e.buffer = { type: 'read-only-storage', hasDynamicOffset: s === 'rd' }
      else if (s === 's') e.buffer = { type: 'storage' }
      else if (s === 'f') e.texture = { sampleType: 'float' }
      else if (s === 'uf') e.texture = { sampleType: 'unfilterable-float' }
      else if (s === 'd') e.texture = { sampleType: 'depth' }
      else if (s === 'w8') e.storageTexture = { access: 'write-only', format: 'rgba8unorm' }
      else if (s === 'w16') e.storageTexture = { access: 'write-only', format: 'rgba16float' }
      else if (s === 'sl') e.sampler = { type: 'filtering' }
      else if (s === 'sc') e.sampler = { type: 'comparison' }
      return e
    }),
  })
}
const L = {}
function buildLayouts() {
  const C = SS.COMPUTE, VF = SS.VERTEX | SS.FRAGMENT
  Object.assign(L, {
    surf: bgl(['s', 'r', 'u', 'uf'], C | SS.FRAGMENT),
    ground: bgl(['u', 'w16'], C), tiles: bgl(['rd'], C), fp: bgl(['rd', 'ud'], VF),
    transfer: bgl(['uf', 'r', 'r', 's', 'ud'], C), apply: bgl(['s', 's', 'r', 'ud', 'r', 's'], C), dyn: bgl(['ud'], C),
    dry: bgl(['u', 'ud', 'rd', 'w8'], C), shade: bgl(['u', 'ud', 'w8', 'w16'], C),
    down8: bgl(['uf', 'w8', 'ud'], C), down16: bgl(['uf', 'w16', 'ud'], C), brushColor: bgl(['u', 'r', 'r', 's', 'u'], C),
    frameOnly: bgl(['u'], VF), frameFull: bgl(['u', 'd', 'd', 'sc', 'sl', 'f'], VF), surfDraw: bgl(['u', 'f', 'f', 'f'], VF),
    mesh: bgl(['r'], VF), objDraw: bgl(['ud'], VF), hair: bgl(['r', 'r'], VF),
  })
  G.L = L
  const cp = (name, layouts, module) => G.dev.createComputePipeline({ layout: G.dev.createPipelineLayout({ bindGroupLayouts: layouts }), compute: { module: module || G.module(name), entryPoint: 'main' } })
  G.P = {
    ground: cp('kGround', [L.ground]), initTile: cp('kInitTile', [L.tiles, L.surf]),
    transfer: cp('kTransfer', [L.transfer, L.surf], G.transferModule), apply: cp('kApply', [L.apply]),
    squeeze: cp('kSqueeze', [L.dyn, L.surf]), pencil: cp('kPencil', [L.dyn, L.surf]),
    dry: cp('kDry', [L.dry, L.surf]), shade: cp('kShade', [L.shade, L.surf]), brushColor: cp('kBrushColor', [L.brushColor]),
    down8: cp('kDown', [L.down8], G.module('kDown', { FORMAT: 'rgba8unorm' })), down16: cp('kDown', [L.down16], G.module('kDown', { FORMAT: 'rgba16float' })),
  }
  G.P.fp = G.dev.createRenderPipeline({
    layout: G.dev.createPipelineLayout({ bindGroupLayouts: [L.fp, L.surf] }),
    vertex: { module: G.module('fp'), entryPoint: 'vs' },
    fragment: { module: G.module('fp'), entryPoint: 'fs', targets: [{ format: 'rgba32float' }] },
    primitive: { topology: 'triangle-strip' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
  })
}

export function buf(size, usage = BU.STORAGE | BU.COPY_DST | BU.COPY_SRC, data) {
  const b = G.dev.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage, mappedAtCreation: !!data })
  if (data) { new data.constructor(b.getMappedRange()).set(data); b.unmap() }
  return b
}
export function bg(layout, res) {
  return G.dev.createBindGroup({
    layout, entries: res.map((r, i) => ({
      binding: i,
      resource: r instanceof GPUBuffer ? { buffer: r } : r,
    })),
  })
}

// ---------------------------------------------------------------- pigments (8 spectral bands, 410..690 nm)
const P = (name, code, rinf, s, dry) => ({ name, code, rinf, s, dry })
export const PIGMENTS = [
  P('Titanium White', 'PW6', [0.86, 0.91, 0.93, 0.94, 0.94, 0.94, 0.935, 0.93], 4.0, 0.012),
  P('Ivory Black', 'PBk9', [0.034, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040], 0.6, 0.010),
  P('Cadmium Yellow Light', 'PY35', [0.06, 0.07, 0.13, 0.58, 0.82, 0.85, 0.86, 0.86], 2.0, 0.014),
  P('Cadmium Orange', 'PO20', [0.06, 0.06, 0.06, 0.08, 0.26, 0.76, 0.83, 0.85], 2.0, 0.014),
  P('Yellow Ochre', 'PY43', [0.09, 0.11, 0.16, 0.27, 0.44, 0.53, 0.57, 0.59], 1.6, 0.022),
  P('Cadmium Red Light', 'PR108', [0.07, 0.05, 0.045, 0.045, 0.07, 0.56, 0.82, 0.86], 2.0, 0.014),
  P('Alizarin Crimson', 'PR83', [0.11, 0.06, 0.03, 0.018, 0.022, 0.19, 0.50, 0.62], 0.25, 0.012),
  P('Ultramarine Blue', 'PB29', [0.30, 0.35, 0.16, 0.045, 0.022, 0.022, 0.035, 0.09], 0.6, 0.016),
  P('Phthalo Blue', 'PB15', [0.04, 0.08, 0.06, 0.02, 0.006, 0.003, 0.003, 0.004], 0.35, 0.02),
  P('Viridian', 'PG18', [0.06, 0.12, 0.30, 0.25, 0.08, 0.03, 0.02, 0.02], 0.4, 0.016),
  P('Burnt Sienna', 'PBr7', [0.05, 0.05, 0.055, 0.07, 0.12, 0.30, 0.42, 0.48], 0.9, 0.03),
  P('Burnt Umber', 'PBr7', [0.04, 0.04, 0.045, 0.05, 0.07, 0.11, 0.15, 0.18], 1.0, 0.04),
  P('Glazing Medium', 'alkyd', [0.25, 0.4, 0.6, 0.8, 0.9, 0.9, 0.9, 0.9], 0.0008, 0.05),
  P('Lemon Yellow', 'PY175', [0.05, 0.07, 0.32, 0.80, 0.87, 0.88, 0.88, 0.88], 1.2, 0.016),
  P('Quinacridone Magenta', 'PR122', [0.30, 0.24, 0.07, 0.018, 0.025, 0.28, 0.66, 0.76], 0.3, 0.016),
  P('Cerulean Blue', 'PB35', [0.38, 0.55, 0.50, 0.28, 0.11, 0.06, 0.05, 0.06], 1.6, 0.016),
]
export const TUBE_ORDER = [0, 13, 2, 3, 4, 5, 6, 14, 7, 15, 8, 9, 10, 11, 1, 12]     // order of the tubes in the tray
export const NPIG = PIGMENTS.length

// band weights: CIE 1931 (Wyman et al. fit) x D65, then to linear sRGB, rows normalised so a flat R=1 is white
function spectralMatrix() {
  const g = (x, mu, s1, s2) => { const t = (x - mu) / (x < mu ? s1 : s2); return Math.exp(-0.5 * t * t) }
  const d65 = [54.6, 82.75, 91.49, 93.43, 86.68, 104.86, 117.01, 117.81, 114.86, 115.92, 108.81, 109.35, 107.80, 104.79, 107.69, 104.41,
    104.05, 100.0, 96.33, 95.79, 88.69, 90.01, 89.60, 87.70, 83.29, 83.70, 80.03, 80.21, 82.28, 78.28, 69.72, 71.61, 74.35]
  const W = [0, 1, 2].map(() => new Array(8).fill(0))
  for (let l = 390; l < 710; l++) {
    const fi = (l - 390) / 10, i0 = Math.floor(fi), fr = fi - i0
    const e = d65[i0] * (1 - fr) + d65[Math.min(i0 + 1, d65.length - 1)] * fr
    const xb = 1.056 * g(l, 599.8, 37.9, 31.0) + 0.362 * g(l, 442.0, 16.0, 26.7) - 0.065 * g(l, 501.1, 20.4, 26.2)
    const yb = 0.821 * g(l, 568.8, 46.9, 40.5) + 0.286 * g(l, 530.9, 16.3, 31.1)
    const zb = 1.217 * g(l, 437.0, 11.8, 36.0) + 0.681 * g(l, 459.0, 26.0, 13.8)
    const b = Math.floor((l - 390) / 40)
    W[0][b] += e * xb; W[1][b] += e * yb; W[2][b] += e * zb
  }
  const ysum = W[1].reduce((a, b) => a + b)
  const X2R = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]
  return [0, 1, 2].map(r => {
    const row = new Array(8).fill(0)
    for (let b = 0; b < 8; b++) for (let c = 0; c < 3; c++) row[b] += X2R[r][c] * W[c][b] / ysum
    const s = row.reduce((a, b) => a + b)
    return row.map(v => v / s)
  })
}
export const SPECM = spectralMatrix()
export const rgb2spec = c => [c[2], c[2], 0.7 * c[2] + 0.3 * c[1], 0.15 * c[2] + 0.85 * c[1], c[1], 0.35 * c[1] + 0.65 * c[0], c[0], c[0]]

function pigmentTable() {
  const K = [], S = []
  for (const p of PIGMENTS) for (const r of p.rinf) { const rr = Math.max(r, 0.001); K.push(p.s * (1 - rr) * (1 - rr) / (2 * rr)); S.push(p.s) }
  while (K.length < NCH * 8) { K.push(0); S.push(0) }
  const M = [...SPECM[0], ...SPECM[1], ...SPECM[2]]
  const dry = PIGMENTS.map(p => p.dry); while (dry.length < NCH) dry.push(0)
  return new Float32Array([...K, ...S, ...M, ...dry, ...rgb2spec([0.80, 0.68, 0.48])])
}
// CPU KM: masstone of a pigment mix (fractions) over white at given thickness, for UI swatches
export function mixColor(v, thick = 40, ground = 0.9) {
  const R = []
  for (let b = 0; b < 8; b++) {
    let kx = 0, sx = 0
    PIGMENTS.forEach((p, i) => { if (i < v.length) { const rr = Math.max(p.rinf[b], 0.001); kx += v[i] * thick * p.s * (1 - rr) * (1 - rr) / (2 * rr); sx += v[i] * thick * p.s } })
    const s = Math.max(sx, 1e-4), a = 1 + kx / s, bb = Math.sqrt(Math.max(a * a - 1, 1e-8)), x = Math.min(Math.max(bb * s, 1e-5), 30)
    const cth = (1 + Math.exp(-2 * x)) / (1 - Math.exp(-2 * x))
    const r = (1 - ground * (a - bb * cth)) / (a - ground + bb * cth)
    const k2 = 0.45
    R.push(0.96 * (1 - k2) * r / (1 - k2 * r))
  }
  return [0, 1, 2].map(c => Math.min(Math.max(SPECM[c].reduce((s, m, b) => s + m * R[b], 0), 0), 1))
}
export const linToSRGB = x => x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055

// ---------------------------------------------------------------- rects
export const rect = (x0, y0, x1, y1) => ({ x0, y0, x1, y1 })
export const NONE = rect(0, 0, 0, 0)
const empty = r => r.x1 <= r.x0 || r.y1 <= r.y0
const union = (a, b) => empty(a) ? b : empty(b) ? a : rect(Math.min(a.x0, b.x0), Math.min(a.y0, b.y0), Math.max(a.x1, b.x1), Math.max(a.y1, b.y1))
const clip = (r, w, h) => rect(Math.max(r.x0, 0), Math.max(r.y0, 0), Math.min(r.x1, w), Math.min(r.y1, h))

// ---------------------------------------------------------------- paint surface
// Dense: ground (tooth + colour), display caches with mips, drying preview. Sparse: paint state tiles in a pool.
export class Surface {
  constructor(id, kind, wmm, hmm, origin, z, cornerMM, texture = 0) {
    Object.assign(this, { id, kind, origin, z, texture, grainOrg: [...origin] })
    this.w = Math.round(wmm * PPM); this.h = Math.round(hmm * PPM)
    this.mask = [0, 0, this.w, this.h]; this.maskR = cornerMM * PPM
    this.hole = [0, 0, 0, 0]
    this.toothAmp = kind === 1 ? 1.5 : [11, 0, 8, 3][texture]
    this.tilesX = Math.ceil(this.w / TILE); this.tilesY = Math.ceil(this.h / TILE)
    this.slots = new Int32Array(this.tilesX * this.tilesY).fill(-1)     // tile -> pool slot
    this.packed = new Map()                                             // file tile -> its encoding as last saved / loaded
    this.tileOf = []                                                    // slot -> tile
    this.pt = buf(this.slots.byteLength, BU.STORAGE | BU.COPY_DST, this.slots)
    this.cap = 0; this.pool = null; this.grow(32)
    const mips = Math.floor(Math.log2(Math.max(this.w, this.h))) + 1
    const tex = (format, m, extra = 0) => G.dev.createTexture({ size: [this.w, this.h], format, mipLevelCount: m, usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING | extra })
    this.ground = tex('rgba16float', 1, TU.COPY_SRC)
    this.color = tex('rgba8unorm', mips); this.nrm = tex('rgba16float', mips)
    this.prev = null       // drying preview, made on first use
    this.mips = mips
    const gesso = [0.80, 0.85, 0.87, 0.88, 0.88, 0.88, 0.875, 0.87]
    this.su = buf(64, BU.UNIFORM | BU.COPY_DST, new Float32Array([this.w, this.h, this.tilesX, this.tilesY, this.toothAmp, kind, 0, 0, ...gesso]))
    this.dirty = NONE
    this.bgs = {}
  }
  get used() { return this.tileOf.length }
  // the pool grows by copying into a bigger buffer (in command order)
  grow(n, enc) {
    const max = Math.floor(G.limits.maxStorageBufferBindingSize / TB)
    if (n > max) throw new Error('Out of GPU memory for this canvas')
    const cap = Math.min(max, Math.max(n, this.cap * 2))
    const nb = buf(cap * TB)
    if (this.pool) { if (enc && this.used) enc.copyBufferToBuffer(this.pool, 0, nb, 0, this.used * TB); Engine.trash.push(this.pool) }
    this.pool = nb; this.cap = cap; this.bgs = {}
  }
  ensurePrev() {
    if (this.prev) return
    this.prev = G.dev.createTexture({ size: [this.w, this.h], format: 'rgba8unorm', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING | TU.RENDER_ATTACHMENT })
    this.drawBG = null; this.bgs.dry = null
  }
  bgSurf() { return this.bgs.surf ||= bg(L.surf, [this.pool, this.pt, this.su, this.ground.createView()]) }
  toTexel(p) { return [(p[0] - this.origin[0]) * PPM, (p[1] - this.origin[1]) * PPM] }
  toWorld(t) { return [this.origin[0] + t[0] / PPM, this.origin[1] + t[1] / PPM] }
  sdf(world) {
    const t = this.toTexel(world), m = this.mask, r = this.maskR
    const qx = Math.abs(t[0] - (m[0] + m[2]) / 2) - (m[2] - m[0]) / 2 + r, qy = Math.abs(t[1] - (m[1] + m[3]) / 2) - (m[3] - m[1]) / 2 + r
    return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r
  }
  contains(world) {
    const t = this.toTexel(world)
    if (this.hole[2] > 0 && Math.hypot(t[0] - this.hole[0], t[1] - this.hole[1]) < this.hole[2]) return false
    return this.sdf(world) <= 0
  }
  // within `mm` of the paintable area (a brush centred just off the edge still reaches it)
  near(world, mm) { return this.sdf(world) <= mm * PPM }
  allDirty() { this.dirty = rect(0, 0, this.w, this.h) }
}

// ---------------------------------------------------------------- bristle brush
function noise1(x, seed) {
  const h = n => { const s = Math.sin(n * 127.1 + seed * 311.7) * 43758.5453; return s - Math.floor(s) }
  const i = Math.floor(x), f = x - i, u = f * f * (3 - 2 * f)
  return h(i) * (1 - u) + h(i + 1) * u
}
export class Rng {
  constructor(seed) { this.s = (seed >>> 0) || 1 }
  next() { let t = (this.s = (this.s + 0x6D2B79F5) >>> 0); t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296 }
  gauss() { const u = Math.max(this.next(), 1e-7), v = this.next(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) }
}
const len2 = (x, y) => Math.hypot(x, y)

export class Brush {
  constructor({ name, round, width, thick, length, n, tuftR, handle, seed, knife = false, pencil = false }) {
    Object.assign(this, { name, round, width, thick, length, n, tuftR, knife, pencil })
    this.handleColor = handle; this.handleLen = knife ? 120 : (pencil ? 160 : 150 + width * 3)
    this.stiffness = knife ? 20 : 1
    this.pressDepth = 0.62       // how far full pressure pushes the bristles down, as a fraction of their length
    this.travel = 0; this.foot = 0; this.slant = 1
    this.ferrule = [0, 0, 30]; this.axis = [0, 0, -1]; this.e1 = [1, 0, 0]; this.e2 = [0, 1, 0]; this.baseZ = 0
    const rng = new Rng(seed)
    this.offs = new Float32Array(2 * n); this.lens = new Float32Array(n); this.dirj = new Float32Array(2 * n); this.rads = new Float32Array(n)
    for (let i = 0; i < n; i++) {
      let ox, oy
      if (round) {
        const a = i * 2.39996323, r = Math.sqrt((i + 0.5) / n)
        ox = Math.cos(a) * r * width * 0.5 + rng.gauss() * 0.04 * width; oy = Math.sin(a) * r * width * 0.5 + rng.gauss() * 0.04 * width
        const rf = len2(ox, oy) / (width * 0.5)
        this.lens[i] = length * (1 - 0.32 * rf * rf) * (1 + 0.035 * rng.gauss())
      } else if (knife) {
        ox = (rng.next() - 0.5) * width; oy = (rng.next() - 0.5) * thick
        const ex = Math.abs(ox) / (width * 0.5)
        this.lens[i] = length * (1 - 0.55 * ex * ex)       // trowel blade: pointed tip
      } else {
        ox = (rng.next() - 0.5) * width; oy = (rng.next() - 0.5) * thick
        const ex = Math.abs(ox) / (width * 0.5)
        this.lens[i] = length * (1 - 0.06 * Math.pow(ex, 6)) * (1 + 0.03 * rng.gauss())
      }
      this.offs[2 * i] = ox; this.offs[2 * i + 1] = oy
      if (!knife) { this.dirj[2 * i] = rng.gauss() * 0.006; this.dirj[2 * i + 1] = rng.gauss() * 0.006 }
      this.rads[i] = tuftR * (0.85 + 0.3 * rng.next())
    }
    this.tip = new Float32Array(2 * n); this.tipZ = new Float32Array(n).fill(10); this.touching = new Uint8Array(n)
    this.bendPt = new Float32Array(2 * n); this.eq = new Float32Array(2 * n); this.lastTip = new Float32Array(2 * n)
    this.lie = new Float32Array(n); this.pressJ = new Float32Array(n); this.root = new Float32Array(3 * n)
    // zones: rounds = core + 6 sectors, flats = 10 strips across the width
    const zones = new Int32Array(n)
    for (let i = 0; i < n; i++) {
      const ox = this.offs[2 * i], oy = this.offs[2 * i + 1]
      if (round) { const r = len2(ox, oy) / (width * 0.5); zones[i] = r < 0.35 ? 0 : 1 + Math.floor((Math.atan2(oy, ox) + Math.PI) / (2 * Math.PI) * 6) % 6 }
      else zones[i] = Math.min(Math.max(Math.floor((ox / width + 0.5) * 10), 0), 9)
    }
    this.nz = round ? 7 : 10
    // paint the bundle can hold: ~30% of its volume (mm^3 -> unit*texel^2 is x1e4); a quarter sits on the tips
    const vol = round ? Math.PI * width * width / 4 * length * 0.55 : width * thick * length * 0.7
    this.capTuft = vol * 0.45 * 1e4 * 0.25 / n
    const cap = new Float32Array(n + this.nz)
    for (let i = 0; i < n; i++) { cap[i] = this.capTuft; cap[n + zones[i]] += this.capTuft * 3 }
    for (let z = 0; z < this.nz; z++) cap[n + z] = Math.max(cap[n + z], 1)
    this.resBytes = (n + this.nz) * NCH * 4
    this.res = buf(this.resBytes); this.dres = buf((n + 1) * NCH * 4); this.capBuf = buf(cap.byteLength, BU.STORAGE, cap)
    this.colors = buf((n + 1) * 16); this.flow = buf(n * NCH * 4); this.zoneBuf = buf(n * 4, BU.STORAGE, zones)
    this.nbBuf = buf(16, BU.UNIFORM, new Float32Array([n, 0, 0, 0]))
    // per-hair random offsets for rendering (3 hairs per tuft)
    const hr = new Rng(7)
    this.hairR = new Float32Array(n * 3 * 9).map(() => 0)
    for (let k = 0; k < n * 3; k++) { for (let c = 0; c < 8; c++) this.hairR[k * 9 + c] = hr.gauss(); this.hairR[k * 9 + 8] = hr.next() }
    this.reclump(seed)
  }

  // bristles part into a new set of clumps on every stroke
  reclump(seed) {
    const rng = new Rng(Math.imul(seed >>> 0, 2654435761) + 99), n = this.n, w = this.width
    const k = this.round ? Math.max(4, Math.floor(w * 2.4)) : Math.max(4, Math.floor(w * 1.7))
    const cs = []
    for (let i = 0; i < k; i++) {
      if (this.round) { const a = rng.next() * 2 * Math.PI, r = Math.sqrt(rng.next()) * w * 0.5; cs.push([Math.cos(a) * r, Math.sin(a) * r]) }
      else cs.push([(rng.next() - 0.5) * w, (rng.next() - 0.5) * this.thick * 0.6])
    }
    this.clump = new Float32Array(2 * n); this.clumpOf = new Int32Array(n)
    for (let j = 0; j < n; j++) {
      const ox = this.offs[2 * j], oy = this.offs[2 * j + 1]
      let best = 0, bd = Infinity
      cs.forEach((c, i) => { const d = this.round ? (ox - c[0]) ** 2 + (oy - c[1]) ** 2 : (ox - c[0]) ** 2 + 0.15 * (oy - c[1]) ** 2; if (d < bd) { bd = d; best = i } })
      this.clump[2 * j] = cs[best][0]; this.clump[2 * j + 1] = cs[best][1]; this.clumpOf[j] = best
    }
    this.nclump = k
    this.clumpJit = new Float32Array(2 * k)
    if (!this.knife) for (let i = 0; i < 2 * k; i++) this.clumpJit[i] = rng.gauss() * 0.03
    this.clumpiness = this.knife ? 0 : 0.45 + 0.3 * rng.next()
  }

  // place the brush: tips' rest centroid over `target` (surface-local mm), pressed by `pressure`
  pose(target, pressure, tilt, roll) {
    let tx = tilt[0], ty = tilt[1]
    const tl = len2(tx, ty); if (tl > 0.85) { tx *= 0.85 / tl; ty *= 0.85 / tl }
    let ax = [-tx, -ty, -Math.sqrt(Math.max(1 - tx * tx - ty * ty, 0.05))]
    const al = Math.hypot(...ax); ax = ax.map(v => v / al)
    let r = [Math.cos(roll), Math.sin(roll), 0]
    const d = r[0] * ax[0] + r[1] * ax[1]
    r = [r[0] - ax[0] * d, r[1] - ax[1] * d, r[2] - ax[2] * d]
    const rl = Math.hypot(...r); r = r.map(v => v / rl)
    this.axis = ax; this.e1 = r
    this.e2 = [ax[1] * r[2] - ax[2] * r[1], ax[2] * r[0] - ax[0] * r[2], ax[0] * r[1] - ax[1] * r[0]]
    const zt = this.pencil ? 0.3 * (1 - Math.min(pressure * 4, 1)) : 0.35 - pressure * this.pressDepth * this.length * Math.abs(ax[2])
    this.ferrule = [target[0] - ax[0] * this.length, target[1] - ax[1] * this.length, zt - ax[2] * this.length]
  }

  // advance bristle physics one substep; writes GPU bristle records (surface texel coords) into `out`, returns count + bbox
  simulate(s, moved, out) {
    const n = this.n, A = this.axis, E1 = this.e1, E2 = this.e2, Fr = this.ferrule, LEN = this.length, W = this.width, knife = this.knife
    const tip = this.tip, eq = this.eq, root = this.root, touching = this.touching
    this.moved = true
    let mlx = 0, mly = 0, nt = 0
    // free bristles converge to the tip; pressing flattens the belly and fans them out
    const pressFan = Math.min(Math.max((0.35 - (Fr[2] + A[2] * LEN)) / (LEN * Math.abs(A[2]) * 0.5), 0), 1)
    const conv = knife ? 0 : (this.round ? 0.82 : 0.15) * (1 - 0.8 * pressFan)
    const ci = this.clumpiness
    for (let j = 0; j < n; j++) {
      const ox = this.offs[2 * j] + (this.clump[2 * j] - this.offs[2 * j]) * ci, oy = this.offs[2 * j + 1] + (this.clump[2 * j + 1] - this.offs[2 * j + 1]) * ci
      const o3x = E1[0] * ox + E2[0] * oy, o3y = E1[1] * ox + E2[1] * oy, o3z = E1[2] * ox + E2[2] * oy
      const Rx = Fr[0] + o3x, Ry = Fr[1] + o3y, Rz = Fr[2] + o3z
      root[3 * j] = Rx; root[3 * j + 1] = Ry; root[3 * j + 2] = Rz
      const L = this.lens[j], k = this.clumpOf[j]
      const jx = this.dirj[2 * j] + this.clumpJit[2 * k], jy = this.dirj[2 * j + 1] + this.clumpJit[2 * k + 1]
      let dx = A[0] * L - o3x * conv + (E1[0] * jx + E2[0] * jy) * L
      let dy = A[1] * L - o3y * conv + (E1[1] * jx + E2[1] * jy) * L
      let dz = A[2] * L - o3z * conv + (E1[2] * jx + E2[2] * jy) * L
      const dl = Math.hypot(dx, dy, dz); dx /= dl; dy /= dl; dz /= dl
      const Fx = Rx + dx * L, Fy = Ry + dy * L, Fz = Rz + dz * L
      if (Fz >= 0) {
        tip[2 * j] = Fx; tip[2 * j + 1] = Fy; this.tipZ[j] = Fz; touching[j] = 0
        this.bendPt[2 * j] = Fx; this.bendPt[2 * j + 1] = Fy; eq[2 * j] = Fx; eq[2 * j + 1] = Fy
        continue
      }
      const c = Math.min(-Fz / Math.max(L * Math.abs(dz), 0.3), 1.6)
      let owx = o3x, owy = o3y
      const ol = len2(owx, owy)
      if (ol > 1e-4) { owx /= ol; owy /= ol } else { const a = this.dirj[2 * j] + 1e-3, b = this.dirj[2 * j + 1], l = len2(a, b); owx = a / l; owy = b / l }
      const splay = knife ? 0 : (this.round ? 0.45 : 0.25) * (ol / Math.max(W * 0.5, 0.1))
      let hx = dx + owx * splay * c, hy = dy + owy * splay * c
      if (len2(hx, hy) < 1e-3) { hx = owx; hy = owy }
      const hl = len2(hx, hy); hx /= hl; hy /= hl
      const reach = Math.max(L * len2(dx, dy), 0.88 * Math.sqrt(Math.max(L * L - Rz * Rz, 0)))
      // clumps and stray bristles wander sideways as they drag; edge bristles more
      const edge = Math.min(ol / Math.max(W * 0.5, 0.1), 1)
      const cw = noise1(this.travel * 0.45 + k * 7.3, 1) - 0.5, tw = noise1(this.travel * 1.3 + j * 3.7, 2) - 0.5
      const wander = knife ? 0 : cw * (0.12 + 0.35 * edge * edge) * c + tw * 0.08
      const Ex = Rx + hx * reach - hy * wander, Ey = Ry + hy * reach + hx * wander
      eq[2 * j] = Ex; eq[2 * j + 1] = Ey
      if (!touching[j]) { tip[2 * j] = Ex; tip[2 * j + 1] = Ey; this.lastTip[2 * j] = Ex; this.lastTip[2 * j + 1] = Ey }
      else {
        const ddx = tip[2 * j] - Ex, ddy = tip[2 * j + 1] - Ey
        const lag = 0.55 * Math.min(c, 1) * L / this.stiffness
        const ddl = len2(ddx, ddy)
        if (ddl > lag) { tip[2 * j] = Ex + ddx / ddl * lag * 0.93; tip[2 * j + 1] = Ey + ddy / ddl * lag * 0.93 }
      }
      touching[j] = 1; this.tipZ[j] = 0
      mlx += tip[2 * j] - Ex; mly += tip[2 * j + 1] - Ey; nt++
    }
    // cohesion: bristles drag as clumps, clumps loosely as a bundle
    if (nt > 0) {
      const mx = mlx / nt, my = mly / nt, K = this.nclump
      const cs = new Float32Array(2 * K), cn = new Float32Array(K)
      for (let j = 0; j < n; j++) if (touching[j]) { const k = this.clumpOf[j]; cs[2 * k] += tip[2 * j] - eq[2 * j]; cs[2 * k + 1] += tip[2 * j + 1] - eq[2 * j + 1]; cn[k]++ }
      for (let k = 0; k < K; k++) if (cn[k] > 0) { cs[2 * k] = cs[2 * k] / cn[k] * 0.6 + mx * 0.4; cs[2 * k + 1] = cs[2 * k + 1] / cn[k] * 0.6 + my * 0.4 }
      for (let j = 0; j < n; j++) if (touching[j]) {
        const k = this.clumpOf[j]
        tip[2 * j] = eq[2 * j] + (tip[2 * j] - eq[2 * j]) * 0.4 + cs[2 * k] * 0.6
        tip[2 * j + 1] = eq[2 * j + 1] + (tip[2 * j + 1] - eq[2 * j + 1]) * 0.4 + cs[2 * k + 1] * 0.6
      }
    }
    // the part of each touching bristle that lies on the surface
    let fs = 0
    for (let j = 0; j < n; j++) if (touching[j]) {
      const L = this.lens[j]
      const trx = root[3 * j] - tip[2 * j], try_ = root[3 * j + 1] - tip[2 * j + 1]
      const dl = len2(trx, try_)
      const Fz = root[3 * j + 2] + A[2] * L
      const c = Math.min(-Fz / Math.max(L * Math.abs(A[2]), 0.3), 1.6)
      this.lie[j] = Math.min(L * Math.min(Math.max(c * (knife ? 2.5 : 1.1), 0.04), 0.85), dl)
      if (dl > 1e-5) { this.bendPt[2 * j] = tip[2 * j] + trx / dl * this.lie[j]; this.bendPt[2 * j + 1] = tip[2 * j + 1] + try_ / dl * this.lie[j] }
      else { this.bendPt[2 * j] = tip[2 * j]; this.bendPt[2 * j + 1] = tip[2 * j + 1] }
      this.pressJ[j] = Math.min(c * (knife ? 8 : 1.6), 1)
      fs += this.lie[j] + 2 * this.rads[j]
    }
    this.foot = nt > 0 ? fs / nt : 0
    this.travel += moved
    if (!s || !out) { this.lastTip.set(tip); return { cnt: 0 } }
    let cnt = 0, sl = 0, sn = 0
    let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity
    const ox = s.origin[0], oy = s.origin[1]
    for (let j = 0; j < n; j++) {
      const L = this.lens[j]
      let ax, ay, bx = tip[2 * j], by = tip[2 * j + 1], press = 0, z = 0, len
      if (touching[j]) { ax = this.bendPt[2 * j]; ay = this.bendPt[2 * j + 1]; press = this.pressJ[j]; len = this.lie[j] }
      else if (this.tipZ[j] < 3) { ax = bx - A[0] * 0.6; ay = by - A[1] * 0.6; z = this.tipZ[j] / UNIT; len = 0.6 }
      else { this.lastTip[2 * j] = bx; this.lastTip[2 * j + 1] = by; continue }
      const r = this.rads[j] * (1 + 0.45 * press)
      const mvx = bx - this.lastTip[2 * j], mvy = by - this.lastTip[2 * j + 1], mv = len2(mvx, mvy)
      if (touching[j] && mv > 1e-4 && len > 1e-3) {
        const ul = Math.max(len2(bx - ax, by - ay), 1e-4), ux = (bx - ax) / ul, uy = (by - ay) / ul
        const s2 = ux * (mvy / mv) - uy * (mvx / mv); sl += s2 * s2; sn++
      }
      const rate = (Math.max(mv, 0.25 * moved) + 0.002) / Math.max(len + 2 * r, 0.25)
      this.lastTip[2 * j] = bx; this.lastTip[2 * j + 1] = by
      const tax = (ax - ox) * PPM, tay = (ay - oy) * PPM, tbx = (bx - ox) * PPM, tby = (by - oy) * PPM, rt = r * PPM
      const o = cnt * 12
      out[o] = tax; out[o + 1] = tay; out[o + 2] = tbx; out[o + 3] = tby
      out[o + 4] = rt; out[o + 5] = press; out[o + 6] = z; out[o + 7] = rate
      out[o + 8] = 2 * this.rads[j] * L * PPM * PPM; out[o + 9] = j; out[o + 10] = 0; out[o + 11] = 0
      cnt++
      const R = Math.trunc(rt * 2.2) + 3
      bx0 = Math.min(bx0, Math.trunc(Math.min(tax, tbx)) - R); by0 = Math.min(by0, Math.trunc(Math.min(tay, tby)) - R)
      bx1 = Math.max(bx1, Math.trunc(Math.max(tax, tbx)) + R); by1 = Math.max(by1, Math.trunc(Math.max(tay, tby)) + R)
    }
    if (sn > 0) this.slant = Math.sqrt(sl / sn)       // rms: the most slanted bristles decide
    return { cnt, bb: cnt > 0 ? rect(bx0, by0, bx1, by1) : NONE }
  }
  hover() { this.touching.fill(0) }
  // the reservoir is GPU state: clear it in command order
  clean(eng) { eng.enc().clearBuffer(this.res) }
}

// ---------------------------------------------------------------- tubes
export class Tube { constructor(p) { this.pigment = p; this.left = 1 } get name() { return PIGMENTS[this.pigment].name } }

// ---------------------------------------------------------------- .pirl v3 tile codec (shared with the Mac app)
// A file tile is 128x128 texels: 44 half-float planes (11 state slices x 4: top layer, under layer, aux, dried reflectance)
// plus two 16-bit planes holding the height as the bit difference from the height recomputed out of tooth and paint.
// State planes are predicted from their neighbours (LOCO-I's median predictor), every plane is split into high and low
// byte planes, and each tile is raw-deflated on its own. Bit-exact.
const FT = 128, FN = FT * FT, NPLANE = 46, TT = TILE * TILE
const H2F = new Float32Array(65536)
for (let i = 0; i < 65536; i++) {
  const e = (i >> 10) & 31, m = i & 1023, sg = i & 0x8000 ? -1 : 1
  H2F[i] = sg * (e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15))
}
const fb = new Float32Array(1), ub = new Uint32Array(fb.buffer)
const bitsOf = f => { fb[0] = f; return ub[0] }
function medPredict(q, i) {
  const x = i & (FT - 1)
  if (i >= FT && x > 0) {
    const a = q[i - 1], c = q[i - FT], d = q[i - FT - 1]
    return Math.max(Math.min(a, c), Math.min(Math.max(a, c), a + c - d))
  }
  return i === 0 ? 0 : q[x > 0 ? i - 1 : i - FT]
}
// height the file stores the residual against: tooth * amp + dried height + all wet paint, in float32 as the Mac app sums it
function heightGuess(pl, tooth, amp, i) {
  let paint = 0
  for (let h = 0; h < 32; h++) { const x = pl[h * FN + i]; if (x) paint = Math.fround(paint + H2F[x]) }     // adding exact zeros changes nothing
  return bitsOf(Math.fround(Math.fround(Math.fround(tooth * amp) + H2F[pl[34 * FN + i]]) + paint))
}
// run bytes through a (de)compression stream, into `out` when given (buffers are reused: fresh memory is slow to fault in
// when several workers allocate at once)
async function through(ts, input, out) {
  const w = ts.writable.getWriter(); w.write(input).catch(() => { }); w.close().catch(() => { })
  const rd = ts.readable.getReader(), chunks = []
  let n = 0
  for (; ;) { const { done, value } = await rd.read(); if (done) break; if (out) out.set(value, n); else chunks.push(value); n += value.length }
  if (out) return out.subarray(0, n)
  const r = new Uint8Array(n); n = 0
  for (const c of chunks) { r.set(c, n); n += c.length }
  return r
}
// raw DEFLATE decoder (RFC 1951) into a preallocated buffer; returns the decoded length
const LBASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258]
const LEXT = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0]
const DBASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577]
const DEXT = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]
const CLORD = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
let HL, HD, HC, LENS      // decode tables (entry = symbol << 4 | code length), indexed by the next 15 bits reversed
function huff(table, lens, n) {
  let max = 0
  for (let i = 0; i < n; i++) max = Math.max(max, lens[i])
  const count = new Int32Array(16), next = new Int32Array(16)
  for (let i = 0; i < n; i++) count[lens[i]]++
  count[0] = 0
  for (let b = 1, code = 0; b < 16; b++) { code = (code + count[b - 1]) << 1; next[b] = code }
  const size = 1 << max
  for (let sym = 0; sym < n; sym++) {
    const L = lens[sym]
    if (!L) continue
    let c = next[L]++, r = 0
    for (let k = 0; k < L; k++) { r = (r << 1) | (c & 1); c >>= 1 }
    for (let j = r; j < size; j += 1 << L) table[j] = (sym << 4) | L
  }
  return max
}
function inflateInto(src, out) {
  HL ||= new Int32Array(1 << 15); HD ||= new Int32Array(1 << 15); HC ||= new Int32Array(1 << 7); LENS ||= new Uint8Array(320)
  let ip = 0, bb = 0, bc = 0, op = 0, final = 0
  const need = n => { while (bc < n) { bb |= (ip < src.length ? src[ip] : 0) << bc; ip++; bc += 8 } }
  const bits = n => { need(n); const v = bb & ((1 << n) - 1); bb >>>= n; bc -= n; return v }
  const sym = (t, max) => { need(max); const e = t[bb & ((1 << max) - 1)], l = e & 15; if (!l) throw new Error('Corrupt tile in file'); bb >>>= l; bc -= l; return e >> 4 }
  while (!final) {
    final = bits(1)
    const type = bits(2)
    if (type === 0) {
      bb = 0; bc = 0
      const len = src[ip] | (src[ip + 1] << 8); ip += 4
      out.set(src.subarray(ip, ip + len), op); ip += len; op += len
      continue
    }
    let lmax, dmax
    if (type === 1) {
      for (let i = 0; i < 288; i++) LENS[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8
      lmax = huff(HL, LENS, 288)
      LENS.fill(5, 0, 30); dmax = huff(HD, LENS, 30)
    } else if (type === 2) {
      const nl = bits(5) + 257, nd = bits(5) + 1, nc = bits(4) + 4
      LENS.fill(0, 0, 19)
      for (let i = 0; i < nc; i++) LENS[CLORD[i]] = bits(3)
      const cmax = huff(HC, LENS, 19)
      const all = new Uint8Array(nl + nd)
      for (let i = 0; i < nl + nd;) {
        const c = sym(HC, cmax)
        if (c < 16) all[i++] = c
        else if (c === 16) { const r = 3 + bits(2), v = all[i - 1]; for (let k = 0; k < r; k++) all[i++] = v }
        else i += c === 17 ? 3 + bits(3) : 11 + bits(7)
      }
      lmax = huff(HL, all, nl); dmax = huff(HD, all.subarray(nl), nd)
    } else throw new Error('Corrupt tile in file')
    for (; ;) {
      const c = sym(HL, lmax)
      if (c < 256) { out[op++] = c; continue }
      if (c === 256) break
      const li = c - 257, len = LBASE[li] + bits(LEXT[li]), di = sym(HD, dmax), dist = DBASE[di] + bits(DEXT[di])
      if (dist > op || op + len > out.length) throw new Error('Corrupt tile in file')
      for (let k = 0; k < len; k++, op++) out[op] = out[op - dist]
    }
  }
  return op
}
let PL, BY      // per-worker scratch: planes and their bytes
const scratch = () => { PL ||= new Uint16Array(NPLANE * FN); BY ||= new Uint8Array(NPLANE * FN * 2) }
// a file tile from its four pool tiles (null where outside the surface); v: valid width/height of each; tooth: 128x128 halves
async function encodeTile({ subs, v, tooth, amp }) {
  scratch()
  const pl = PL.fill(0), o = BY
  subs.forEach((buf, k) => {
    if (!buf) return
    const S32 = new Uint32Array(buf), S16 = new Uint16Array(buf), ox = (k & 1) * TILE, oy = (k >> 1) * TILE
    for (let ly = 0; ly < v[k][1]; ly++) for (let lx = 0; lx < v[k][0]; lx++) {
      const li = ly * TILE + lx, i = (oy + ly) * FT + ox + lx
      for (let h = 0; h < 44; h++) pl[h * FN + i] = S16[((h >> 1) * TT + li) * 2 + (h & 1)]
      const r = (S32[22 * TT + li] - heightGuess(pl, H2F[tooth[i]], amp, i)) >>> 0
      pl[44 * FN + i] = r >>> 16; pl[45 * FN + i] = r & 0xffff
    }
  })
  for (let p = 0; p < NPLANE; p++) {
    const q = pl.subarray(p * FN, (p + 1) * FN), ob = 2 * p * FN
    for (let i = 0; i < FN; i++) { const d = (q[i] - (p < 44 ? medPredict(q, i) : 0)) & 0xffff; o[ob + i] = d >> 8; o[ob + FN + i] = d & 255 }
  }
  return { z: await through(new CompressionStream('deflate-raw'), o), subs }
}
// decodes into the given pool-tile buffers (`out`, null where outside the surface)
async function decodeTile({ z, v, tooth, amp, out }) {
  scratch()
  const r = BY, pl = PL
  if (inflateInto(z, r) !== NPLANE * FN * 2) throw new Error('Corrupt tile in file')
  for (let p = 0; p < NPLANE; p++) {
    const q = pl.subarray(p * FN, (p + 1) * FN), ob = 2 * p * FN
    for (let i = 0; i < FN; i++) q[i] = ((p < 44 ? medPredict(q, i) : 0) + ((r[ob + i] << 8) | r[ob + FN + i])) & 0xffff
  }
  out.forEach((buf, k) => {
    if (!buf) return
    const S32 = new Uint32Array(buf).fill(0), S16 = new Uint16Array(buf), ox = (k & 1) * TILE, oy = (k >> 1) * TILE
    for (let ly = 0; ly < v[k][1]; ly++) for (let lx = 0; lx < v[k][0]; lx++) {
      const li = ly * TILE + lx, i = (oy + ly) * FT + ox + lx
      for (let h = 0; h < 44; h++) S16[((h >> 1) * TT + li) * 2 + (h & 1)] = pl[h * FN + i]
      S32[22 * TT + li] = (heightGuess(pl, H2F[tooth[i]], amp, i) + ((pl[44 * FN + i] << 16) | pl[45 * FN + i])) >>> 0
    }
  })
  return out
}
const IN_WORKER = typeof document === 'undefined'
if (IN_WORKER) self.onmessage = async e => {
  const { job, args } = e.data
  try {
    const r = await (job === 'encode' ? encodeTile(args) : decodeTile(args))
    const bufs = job === 'encode' ? r.subs : r
    self.postMessage({ r }, bufs.filter(Boolean))
  } catch (err) { self.postMessage({ err: err.message }) }
}
// start the workers and get their code optimized (a cold first tile takes seconds): a few round trips of a dummy tile
// through each worker, one worker at a time, only while `idle()` (painting slows down otherwise)
export async function warmCodec(idle = () => true) {
  const v = [[TILE, TILE], [TILE, TILE], [TILE, TILE], [TILE, TILE]]
  const one = w => codecJob('encode', () => { const subs = v.map(() => { const b = new ArrayBuffer(TB); new Uint16Array(b).forEach((_, i, a) => { a[i] = (i * 2654435761) >>> 20 }); return b }); return [{ subs, v, tooth: new Uint16Array(FN), amp: 1 }, subs] }, w)
    .then(r => codecJob('decode', () => { const out = v.map(() => new ArrayBuffer(TB)); return [{ z: r.z, v, tooth: new Uint16Array(FN), amp: 1, out }, out] }, w))
  codecJob('none', () => [{}, []]).catch(() => { })      // spawns the workers
  for (let w = 0; w < workers.length; w++) for (let k = 0; k < 4; k++) {
    while (!idle()) await new Promise(r => setTimeout(r, 500))
    await one(w)
  }
}
// pool-tile sized buffers shuttled between the main thread and the workers
const spare = []
const tileBuf = () => spare.pop() || new ArrayBuffer(TB)
// one job per worker at a time; a job's inputs are made (by `make`) only when a worker takes it, to bound memory
const workers = [], queue = []
function codecJob(job, make, pin) {
  if (!workers.length) for (let i = 0; i < Math.max(2, Math.min(6, (navigator.hardwareConcurrency || 4) - 2)); i++) workers.push({ w: new Worker(new URL(import.meta.url), { type: 'module' }), busy: false })
  return new Promise((resolve, reject) => { queue.push({ job, make, pin, resolve, reject }); pump() })
}
function pump() {
  workers.forEach((wk, i) => {
    const k = wk.busy ? -1 : queue.findIndex(q => q.pin === undefined || q.pin === i)
    if (k < 0) return
    const q = queue.splice(k, 1)[0], [args, transfer] = q.make()
    wk.busy = true
    wk.w.onmessage = e => { wk.busy = false; e.data.err ? q.reject(new Error(e.data.err)) : q.resolve(e.data.r); pump() }
    wk.w.postMessage({ job: q.job, args }, transfer)
  })
}

// ---------------------------------------------------------------- engine
export class Engine {
  static trash = []      // GPU buffers to destroy after the next submit

  constructor() {
    this.surfaces = []
    this.brushes = []
    this.tubes = []
    this.active = 0
    this.pigBuf = buf(304 * 4, BU.UNIFORM, pigmentTable())
    this.fps = {}          // footprint targets by size (128, 256, FPMAX): a small brush doesn't clear a big one every substep
    // per-frame arena for bristle records and kernel parameters (uploaded once before submit)
    this.arenaCap = 32 << 20
    this.arenaAB = new ArrayBuffer(this.arenaCap)
    this.arenaF = new Float32Array(this.arenaAB); this.arenaI = new Int32Array(this.arenaAB)
    this.arena = buf(this.arenaCap + (256 << 10), BU.UNIFORM | BU.STORAGE | BU.COPY_DST | BU.COPY_SRC)
    this.arenaOff = 0
    this._enc = null
    // sim tuning
    // substeps: at least every `step` mm, scaled by `coarse` (1 = fastest, smaller is finer), and every `tick` s of sample time
    Object.assign(this, { step: 0.1, coarse: 0.45, tick: 0.008 })
    Object.assign(this, { alpha: 0.7, rho: 0.75, ratePass: 0.75, mixPass: 0.35, eps: 3.5, feed: 0.5, wick: 0.004, groove: 9, conform: 1.1, coat: 0.7, film: 12, sinkRep: 0.8, relief: 1.0 })
    this.medium = 0                      // 0 oil, 1 acrylic, 2 gouache
    this._dullDry = true
    // input / history
    this.cur = null; this.lastSample = null; this.undoRec = null
    this.undoStack = []; this.redoStack = []; this.history = []; this.redoOps = []
    this.undoBudget = Math.min(1.5e9, Math.max(2.5e8, (navigator.deviceMemory || 8) * 1.25e8))     // GPU memory for undo
    this.onChange = null

    this.surfaces.push(Engine.makeCanvas(400, 300, 0))
    const pal = new Surface(1, 1, 340, 260, [...Engine.paletteHome], -14, 34)
    pal.hole = [38 * PPM, 222 * PPM, 13 * PPM, 0]
    this.surfaces.push(pal)
    const specs = [
      ['Round 2', true, 2.2, 2.2, 9, 70, 0.14, [0.50, 0.08, 0.06]],
      ['Round 6', true, 4.6, 4.6, 15, 190, 0.17, [0.50, 0.08, 0.06]],
      ['Round 12', true, 8.2, 8.2, 23, 380, 0.21, [0.50, 0.08, 0.06]],
      ['Flat 4', false, 6.5, 1.8, 9.5, 170, 0.15, [0.05, 0.05, 0.05]],
      ['Flat 8', false, 11.5, 2.6, 14, 340, 0.18, [0.05, 0.05, 0.05]],
      ['Flat 14', false, 19, 3.6, 19, 640, 0.22, [0.05, 0.05, 0.05]],
    ]
    specs.forEach((s, i) => this.brushes.push(new Brush({ name: s[0], round: s[1], width: s[2], thick: s[3], length: s[4], n: s[5], tuftR: s[6], handle: s[7], seed: 1000 + i * 77 })))
    this.brushes.push(new Brush({ name: 'Palette knife', round: false, width: 22, thick: 0.5, length: 32, n: 260, tuftR: 0.13, handle: [0.42, 0.26, 0.13], seed: 9999, knife: true }))
    this.brushes.push(new Brush({ name: 'Round 0', round: true, width: 0.9, thick: 0.9, length: 7, n: 30, tuftR: 0.07, handle: [0.50, 0.08, 0.06], seed: 1700 }))
    this.brushes.push(new Brush({ name: 'Rigger 1', round: true, width: 1.2, thick: 1.2, length: 24, n: 50, tuftR: 0.07, handle: [0.12, 0.20, 0.12], seed: 1800 }))
    this.brushes.push(new Brush({ name: 'Pencil HB', round: true, width: 0.6, thick: 0.6, length: 14, n: 1, tuftR: 0.3, handle: [0.06, 0.20, 0.12], seed: 1900, pencil: true }))
    this.brushes[7].pressDepth = 0.35; this.brushes[8].pressDepth = 0.18; this.brushes[6].pressDepth = 0.1
    this.tubes = PIGMENTS.map((_, i) => new Tube(i))
    this.paletteStart = [...Engine.paletteHome]
    for (const s of this.surfaces) this.initSurface(s)
  }
  get canvas() { return this.surfaces[0] }
  get palette() { return this.surfaces[1] }
  get dullDry() { return this._dullDry }
  set dullDry(v) { this._dullDry = v; for (const s of this.surfaces) s.allDirty() }

  // canvases sit with their bottom-right corner at a fixed spot next to the palette and brushes
  static makeCanvas(wmm, hmm, texture) { return new Surface(0, 0, wmm, hmm, [200 - wmm, -150], 0, 0, texture) }
  static paletteHome = [225, -140]

  // ------------------------------------------------ command recording
  enc() { return this._enc ||= G.dev.createCommandEncoder() }
  alloc(bytes) {
    const o = this.arenaOff
    if (o + bytes > this.arenaCap) return -1
    this.arenaOff = (o + bytes + 255) & ~255
    return o
  }
  // parameters into the arena; returns the offset
  params(arr) { const o = this.alloc(arr.length * 4); if (o < 0) throw new Error('arena full'); this.arenaF.set(arr, o / 4); return o }
  submit() {
    if (!this._enc) return
    if (this.arenaOff) G.q.writeBuffer(this.arena, 0, this.arenaAB, 0, this.arenaOff)
    G.q.submit([this._enc.finish()])
    this._enc = null; this.arenaOff = 0
    for (const b of Engine.trash) b.destroy()
    Engine.trash = []
  }
  get arenaFull() { return this.arenaOff > 24 << 20 }

  initSurface(s) {
    const e = this.enc()
    const gb = buf(32, BU.UNIFORM, new Float32Array([s.w, s.h, s.kind === 1 ? 1 : s.texture, s.texture === 2 ? 4.2 : 5.6, s.id * 13.7 + 1.3, 0, s.grainOrg[0], s.grainOrg[1]]))
    const p = e.beginComputePass()
    p.setPipeline(G.P.ground); p.setBindGroup(0, bg(L.ground, [gb, s.ground.createView()]))
    p.dispatchWorkgroups(Math.ceil(s.w / 16), Math.ceil(s.h / 16))
    p.end()
    Engine.trash.push(gb)
    s.allDirty()
  }

  // ------------------------------------------------ tiles
  // allocate (and initialize) the paint state tiles under a region
  ensureTiles(s, r) {
    const fresh = []
    for (let ty = r.y0 >> 6; ty <= (r.y1 - 1) >> 6; ty++) for (let tx = r.x0 >> 6; tx <= (r.x1 - 1) >> 6; tx++) {
      const t = ty * s.tilesX + tx
      if (s.slots[t] < 0) fresh.push(t)
    }
    if (!fresh.length) return
    const e = this.enc()
    if (s.used + fresh.length > s.cap) s.grow(s.used + fresh.length, e)
    const o = this.alloc(fresh.length * 4)
    fresh.forEach((t, k) => {
      const slot = s.used
      s.slots[t] = slot; s.tileOf.push(t)
      this.arenaI[o / 4 + k] = slot
      e.copyBufferToBuffer(this.arena, o + k * 4, s.pt, t * 4, 4)
    })
    this.tileKernel(s, G.P.initTile, fresh, [s.bgSurf()], l => bg(L.tiles, [l]))
  }
  // a kernel over a list of tiles (one 8x8-group grid per tile)
  tileKernel(s, pipe, tiles, rest, mk, extraDyn = []) {
    for (let i = 0; i < tiles.length; i += 16384) {
      const part = tiles.slice(i, i + 16384)
      const o = this.alloc(part.length * 4)
      this.arenaI.set(part, o / 4)
      const p = this.enc().beginComputePass()
      p.setPipeline(pipe)
      p.setBindGroup(0, mk({ buffer: this.arena, size: 65536 }), [...extraDyn, o])
      rest.forEach((b, k) => p.setBindGroup(k + 1, b))
      p.dispatchWorkgroups(8, 8, part.length)
      p.end()
    }
  }
  // snapshot tiles under a region for undo (once per op)
  snapTiles(s, r) {
    this.ensureTiles(s, r)
    const ftx = Math.ceil(s.w / FT)
    for (let ty = r.y0 >> 7; ty <= (r.y1 - 1) >> 7; ty++) for (let tx = r.x0 >> 7; tx <= (r.x1 - 1) >> 7; tx++) s.packed.delete(ty * ftx + tx)
    const rec = this.undoRec
    if (!rec) return
    const e = this.enc()
    for (let ty = r.y0 >> 6; ty <= (r.y1 - 1) >> 6; ty++) for (let tx = r.x0 >> 6; tx <= (r.x1 - 1) >> 6; tx++) {
      const t = ty * s.tilesX + tx, key = s.id * 1e7 + t
      if (rec.keys.has(key)) continue
      rec.keys.add(key)
      const k = rec.tiles.length, ci = k >> 4
      if (ci >= rec.chunks.length) { rec.chunks.push(buf(16 * TB)); rec.bytes += 16 * TB }
      e.copyBufferToBuffer(s.pool, s.slots[t] * TB, rec.chunks[ci], (k & 15) * TB, TB)
      rec.tiles.push({ s: s.id, t })
    }
  }
  snapAll(s) { for (const t of s.tileOf) { const tx = t % s.tilesX, ty = Math.floor(t / s.tilesX); this.snapTiles(s, rect(tx * TILE, ty * TILE, tx * TILE + 1, ty * TILE + 1)) } }
  touch(s, r) { s.dirty = union(s.dirty, clip(rect(r.x0 - 2, r.y0 - 2, r.x1 + 2, r.y1 + 2), s.w, s.h)) }

  // ------------------------------------------------ ops
  newRec() { return { tiles: [], chunks: [], keys: new Set(), brushes: [], tubes: this.tubes.map(t => t.left), crop: [...this.canvas.mask], pal: [...this.palette.origin], bytes: 0 } }
  begin(op) {
    const r = this.newRec()
    this.undoRec = r
    if (op.t === 'stroke') {
      this.snapBrush(r, op.b)
      this.brushes[op.b].reclump((this.history.length + 1) * 7919 + op.b)
    }
    this.cur = { op, done: 0 }
    this.lastSample = null
  }
  add(s) { if (this.cur && this.cur.op.ss) this.cur.op.ss.push(s) }
  end() {
    const c = this.cur
    if (!c) return
    this.process()
    this.drain()
    if (c.op.t === 'stroke') {
      this.liftOff(this.brushes[c.op.b])
      if (this.brushes[c.op.b].knife) this.brushes[c.op.b].clean(this)     // wipe the knife on the rag
    }
    this.cur = null
    this.commit(c.op)
  }
  get backlog() { const c = this.cur; return c && c.op.ss ? c.op.ss.length - c.done : 0 }
  commit(op) {
    if (this.undoRec) { this.undoStack.push(this.undoRec); this.clearRedo(); this.trimUndo() }
    this.undoRec = null
    this.history.push(op); this.redoOps = []
    this.onChange?.()
  }
  freeRec(r) { for (const c of r.chunks) Engine.trash.push(c); for (const b of r.brushes) Engine.trash.push(b.buf) }
  clearRedo() { for (const r of this.redoStack) this.freeRec(r); this.redoStack = [] }
  trimUndo() {
    let total = this.undoStack.reduce((a, r) => a + r.bytes, 0)
    while (total > this.undoBudget && this.undoStack.length > 1) { const r = this.undoStack.shift(); total -= r.bytes; this.freeRec(r) }
  }
  // process pending samples of the current op (within a time budget, in ms)
  process(budget = Infinity) {
    const c = this.cur, t0 = performance.now()
    if (!c || !c.op.ss) return
    const more = () => c.done < c.op.ss.length && !this.arenaFull && performance.now() - t0 < budget
    if (c.op.t === 'stroke') { const b = this.brushes[c.op.b]; while (more()) this.strokeTo(b, c.op.ss[c.done++]) }
    else while (more()) this.squeezeTo(this.tubes[c.op.b], c.op.ss[c.done++])
  }
  // process everything still pending, submitting as the arena fills
  drain() { while (this.backlog > 0) { this.process(); if (this.backlog > 0) this.submit() } }
  surfaceFor(s) { return s.s >= 0 && s.s < this.surfaces.length ? this.surfaces[s.s] : null }

  // ------------------------------------------------ brush strokes
  strokeTo(b, s) {
    const STEP = this.step, COARSE = this.coarse
    const l = this.lastSample
    if (!l) { this.lastSample = s; b.hover(); this.substep(b, s, 0); return }
    const d = Math.hypot(s.x - l.x, s.y - l.y), dt = Math.max(s.t - l.t, 0)
    // transfer is normalized by distance moved, so a substep only has to be short relative to the bristles'
    // contact (and sideways, relative to a bristle's thickness, so slanted bristles don't stamp)
    const step = b.pencil ? 0.4 : b.foot > 0 ? Math.min(Math.max(STEP, COARSE * Math.min(0.25 * b.foot, 3 * b.tuftR / Math.max(b.slant, 0.3))), 1.2 * COARSE) : 1.0
    const n = Math.min(Math.max(Math.ceil(d / step), Math.ceil(dt / this.tick), 1), 600)
    for (let k = 1; k <= n; k++) {
      const f = k / n, m = {}
      for (const key of ['x', 'y', 'p', 'tx', 'ty', 'roll', 't']) m[key] = l[key] + (s[key] - l[key]) * f
      m.s = f < 0.5 ? l.s : s.s
      this.substep(b, m, d / n)
    }
    this.lastSample = s
  }
  liftOff(b) { b.hover(); this.lastSample = null }
  // flats turn so their width lies across the direction of travel, or along it (edgewise, for thin lines)
  static followRoll(roll, dir, edge = false) {
    const target = Math.atan2(dir[1], dir[0]) + (edge ? 0 : Math.PI / 2)
    let d = (target - roll) % Math.PI
    if (d > Math.PI / 2) d -= Math.PI; else if (d < -Math.PI / 2) d += Math.PI
    return roll + d * 0.18
  }
  // a mouse has no pressure falloff: emulate the hand lifting while still moving (recorded like real samples)
  static liftSamples(ss) {
    const l = ss[ss.length - 1]
    if (!l) return []
    let back = ss[0]
    for (let i = ss.length - 1; i >= 0; i--) if (l.t - ss[i].t > 0.03) { back = ss[i]; break }
    const dt = Math.max(l.t - back.t, 1e-3)
    let vx = (l.x - back.x) / dt, vy = (l.y - back.y) / dt
    const sp = Math.hypot(vx, vy); if (sp > 250) { vx *= 250 / sp; vy *= 250 / sp }
    const out = []
    for (let k = 1; k <= 8; k++) { const f = k / 8, t = f * 0.06; out.push({ ...l, x: l.x + vx * t, y: l.y + vy * t, t: l.t + t, p: l.p * (1 - f) * (1 - f) }) }
    return out
  }

  substep(b, s, moved) {
    const surf = this.surfaceFor(s)
    b.baseZ = surf ? surf.z : -20
    b.pose([s.x, s.y], s.p, [s.tx, s.ty], s.roll)
    if (b.pencil) { this.pencilStep(b, surf, s, moved); return }
    const off = this.alloc(b.n * 48)
    if (off < 0) return
    const out = new Float32Array(this.arenaAB, off, b.n * 12)
    const { cnt, bb } = b.simulate(surf, moved, surf ? out : null)
    if (!surf || cnt === 0) return
    // footprint region
    let ox = bb.x0, oy = bb.y0, w = bb.x1 - bb.x0, h = bb.y1 - bb.y0
    if (w > FPMAX || h > FPMAX) {
      const cx = (bb.x0 + bb.x1) >> 1, cy = (bb.y0 + bb.y1) >> 1
      w = Math.min(w, FPMAX); h = Math.min(h, FPMAX); ox = cx - (w >> 1); oy = cy - (h >> 1)
    }
    for (let i = 0; i < cnt; i++) { out[i * 12] -= ox; out[i * 12 + 1] -= oy; out[i * 12 + 2] -= ox; out[i * 12 + 3] -= oy }
    const sf = surf
    const region = clip(rect(ox, oy, ox + w, oy + h), sf.w, sf.h)
    if (empty(region)) return
    this.snapTiles(sf, region)
    this.touch(sf, region)

    const gr = b.knife ? 0 : (sf.kind === 0 ? this.groove : this.groove * 0.6)
    const fpo = this.params([w, h, ox, oy, sf.w, sf.h, sf.toothAmp, gr, b.knife ? 0.3 : this.conform, b.knife ? 0.2 : this.coat, 0, 0])
    const spo = this.params([ox, oy, w, h, sf.w, sf.h, b.n, b.knife ? 1 : 0,
      this.alpha, b.knife ? this.rho * 0.3 : this.rho, this.ratePass, b.knife ? 0.7 : this.mixPass,
      ...sf.mask, sf.maskR, sf.toothAmp, this.eps, this.film,
      this.sinkRep, gr, b.knife ? 0.5 : (this.history.length % 997) * 0.731, 0, ...sf.hole])
    const apo = this.params([b.n, 1 - Math.pow(1 - this.feed, Math.max(moved, 0.02)), 1 - Math.pow(1 - this.wick, Math.max(moved, 0.02)), 0, b.nz, 0, 0, 0])

    const e = this.enc(), fp = this.footprint(w <= 128 && h <= 128 ? 128 : w <= 256 && h <= 256 ? 256 : FPMAX)
    const rp = e.beginRenderPass({
      colorAttachments: [{ view: fp.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, -1, 0] }],
      depthStencilAttachment: { view: fp.depth, depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 1 },
    })
    rp.setViewport(0, 0, w, h, 0, 1)
    rp.setScissorRect(0, 0, w, h)
    rp.setPipeline(G.P.fp)
    this.fpBG ||= bg(L.fp, [{ buffer: this.arena, size: FPMAX * 48 }, { buffer: this.arena, size: 48 }])
    rp.setBindGroup(0, this.fpBG, [off, fpo])
    rp.setBindGroup(1, sf.bgSurf())
    rp.draw(4, cnt)
    rp.end()

    b.bgs ||= {}
    const tk = 'transfer' + fp.n
    b.bgs[tk] ||= bg(L.transfer, [fp.view, b.res, b.capBuf, b.dres, { buffer: this.arena, size: 112 }])
    b.bgs.apply ||= bg(L.apply, [b.res, b.dres, b.capBuf, { buffer: this.arena, size: 32 }, b.zoneBuf, b.flow])
    const cp = e.beginComputePass()
    cp.setPipeline(G.P.transfer)
    cp.setBindGroup(0, b.bgs[tk], [spo]); cp.setBindGroup(1, sf.bgSurf())
    cp.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16))
    cp.setPipeline(G.P.apply)
    cp.setBindGroup(0, b.bgs.apply, [apo])
    cp.dispatchWorkgroups(1)
    cp.end()
  }

  footprint(n) {
    if (!this.fps[n]) {
      const t = G.dev.createTexture({ size: [n, n], format: 'rgba32float', usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING })
      const d = G.dev.createTexture({ size: [n, n], format: 'depth32float', usage: TU.RENDER_ATTACHMENT })
      this.fps[n] = { n, view: t.createView(), depth: d.createView() }
    }
    return this.fps[n]
  }

  // graphite: the point sweeps a capsule from its last position, catching the tops of the weave under light pressure
  pencilStep(b, sf, s, moved) {
    const last = [b.tip[0], b.tip[1]]
    b.tip[0] = s.x; b.tip[1] = s.y
    if (!sf || s.p <= 0.01 || moved <= 0) return
    const r = (0.28 + 0.12 * s.p) * PPM
    const a = sf.toTexel(last), c = sf.toTexel([s.x, s.y])
    const R = Math.trunc(r) + 2
    const region = clip(rect(Math.trunc(Math.min(a[0], c[0])) - R, Math.trunc(Math.min(a[1], c[1])) - R, Math.trunc(Math.max(a[0], c[0])) + R, Math.trunc(Math.max(a[1], c[1])) + R), sf.w, sf.h)
    if (empty(region)) return
    this.snapTiles(sf, region)
    this.touch(sf, region)
    const L_ = Math.hypot(c[0] - a[0], c[1] - a[1])
    const o = this.params([region.x0, region.y0, region.x1 - region.x0, region.y1 - region.y0, sf.w, sf.h, ...a, ...c, r, s.p, sf.toothAmp, L_ / (L_ + 2 * r), 0, 0,
      ...sf.mask, sf.maskR, b.travel, 0, 0, ...sf.hole])
    b.travel += moved
    this.dynKernel(sf, G.P.pencil, o, 112, region)
  }
  dynKernel(sf, pipe, o, size, region) {
    this.dynBG ||= bg(L.dyn, [{ buffer: this.arena, size: 256 }])
    const p = this.enc().beginComputePass()
    p.setPipeline(pipe); p.setBindGroup(0, this.dynBG, [o]); p.setBindGroup(1, sf.bgSurf())
    p.dispatchWorkgroups(Math.ceil((region.x1 - region.x0) / 16), Math.ceil((region.y1 - region.y0) / 16))
    p.end()
  }

  // ------------------------------------------------ tube squeeze
  squeezeTo(t, s) {
    const l = this.lastSample || s
    this.lastSample = s
    const sf = this.surfaceFor(s)
    if (!sf) return            // tubes flatten as they're used but never run dry
    const d = Math.hypot(s.x - l.x, s.y - l.y), dt = Math.max(s.t - l.t, 0)
    const n = Math.max(Math.ceil(d / 0.25), 1)
    const rad = 2.0 * PPM                       // nozzle radius in texels
    const rope = 270                            // rope height (units)
    // peak deposit per substep so consecutive gaussians sum to a rope of height `rope`, plus a slow pile-up when held still
    const amt = s.p * (rope * d * PPM / (rad * 1.144) + 500 * dt) / n
    const dir = d > 1e-4 ? [(s.x - l.x) / d, (s.y - l.y) / d] : [1, 0]
    const comp = new Array(NCH).fill(0); comp[t.pigment] = 1
    for (let k = 1; k <= n; k++) {
      const f = k / n
      const c = sf.toTexel([l.x + (s.x - l.x) * f, l.y + (s.y - l.y) * f])
      const R = Math.trunc(rad * 1.7)
      const region = clip(rect(Math.trunc(c[0]) - R, Math.trunc(c[1]) - R, Math.trunc(c[0]) + R, Math.trunc(c[1]) + R), sf.w, sf.h)
      if (empty(region)) continue
      this.snapTiles(sf, region)
      this.touch(sf, region)
      t.left = Math.max(0.12, t.left - amt * rad * rad * 1.31 * UNIT / (PPM * PPM) / 21000 * 2)
      const o = this.params([region.x0, region.y0, region.x1 - region.x0, region.y1 - region.y0, sf.w, sf.h, ...c, ...comp,
        rad, amt, rope * 1.15, sf.toothAmp, ...sf.mask, sf.maskR, t.pigment * 3.1, ...dir, ...sf.hole])
      this.dynKernel(sf, G.P.squeeze, o, 160, region)
    }
  }

  // ------------------------------------------------ drying
  dryParams(hours, apply) {
    const speed = this.medium === 0 ? 1 : 60
    return [hours * speed, 25, this.medium === 1 ? 0.75 : 1, apply ? 1 : 0]
  }
  dryKernel(s, p) {
    if (!s.used) return
    // applying writes no preview, so it doesn't need the preview texture
    const k = s.prev ? 'dry' : 'dryApply'
    s.bgs[k] ||= bg(L.dry, [this.pigBuf, { buffer: this.arena, size: 16 }, { buffer: this.arena, size: 65536 }, (s.prev || G.dummy).createView()])
    const o = this.params(p)
    this.tileKernel(s, G.P.dry, s.tileOf, [s.bgSurf()], () => s.bgs[k], [o])
  }
  preview(hours, palette) {
    for (const s of this.surfaces) if (s.kind === 0 || palette) {
      s.ensurePrev()
      const p = this.enc().beginRenderPass({ colorAttachments: [{ view: s.prev.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] })
      p.end()
      this.dryKernel(s, this.dryParams(hours, false))
    }
  }
  dry(hours, palette) {
    this.undoRec = this.newRec()
    for (const s of this.surfaces) if (s.kind === 0 || palette) {
      this.snapAll(s)
      this.dryKernel(s, this.dryParams(hours, true))
      s.allDirty()
    }
    this.commit({ t: 'dry', hours, palette })
  }

  // ------------------------------------------------ misc ops
  cleanBrush(i) {
    const r = this.newRec()
    this.snapBrush(r, i)
    this.undoRec = r
    this.brushes[i].clean(this)
    this.commit({ t: 'clean', b: i })
  }
  // scrape the palette clean / wipe the canvas back to fresh gesso
  clearSurface(pal) {
    this.undoRec = this.newRec()
    const s = pal ? this.palette : this.canvas
    this.snapAll(s)
    if (s.used) this.tileKernel(s, G.P.initTile, s.tileOf, [s.bgSurf()], l => bg(L.tiles, [l]))
    s.allDirty()
    this.commit({ t: pal ? 'scrape' : 'wipe' })
  }
  // the palette rests on the table, or is held just above whatever it is moved over
  placePalette(o) {
    const p = this.palette, c = this.canvas
    p.origin = [Math.min(Math.max(o[0], -3000), 3000), Math.min(Math.max(o[1], -3000), 3000)]
    const pl = p.origin, ph = [pl[0] + p.w / PPM, pl[1] + p.h / PPM]
    const cl = c.toWorld([c.mask[0], c.mask[1]]), ch = c.toWorld([c.mask[2], c.mask[3]])
    // things it would rest on: the canvas, the brushes and the tube tray
    const under = [[cl[0], cl[1], ch[0], ch[1]], [-200, -440, 90, -150], [238, -264, 512, -150]]
    const over = under.some(u => pl[0] < u[2] && ph[0] > u[0] && pl[1] < u[3] && ph[1] > u[1])
    p.z = over ? c.z + 7 : -14
  }
  movePalette(o) { this.undoRec = this.newRec(); this.placePalette(o); this.commit({ t: 'move', x: o[0], y: o[1] }) }
  crop(r) {
    this.undoRec = this.newRec()
    this.canvas.mask = [...r]
    this.canvas.allDirty()
    this.commit({ t: 'crop', rect: [...r] })
  }
  newCanvas(wmm, hmm, texture) {
    this.flushAll()
    const c = Engine.makeCanvas(wmm, hmm, texture)
    this.destroySurface(this.canvas)
    this.surfaces[0] = c
    this.initSurface(c)
    this.placePalette(this.palette.origin); this.paletteStart = [...this.palette.origin]
    this.history = []; this.redoOps = []; this.undoStack = []; this.clearRedo()
  }
  flushAll() { for (const r of this.undoStack) this.freeRec(r); this.undoStack = []; this.clearRedo() }
  destroySurface(s) { for (const b of [s.pool, s.pt, s.su]) Engine.trash.push(b); for (const t of [s.ground, s.color, s.nrm, s.prev]) t?.destroy() }
  resetAll() {
    for (const s of this.surfaces) {
      s.packed.clear()
      if (s.used) this.tileKernel(s, G.P.initTile, s.tileOf, [s.bgSurf()], l => bg(L.tiles, [l]))
      s.mask = [0, 0, s.w, s.h]; s.allDirty()
    }
    for (const b of this.brushes) b.clean(this)
    for (const t of this.tubes) t.left = 1
    this.placePalette(this.paletteStart)
  }

  // ------------------------------------------------ undo (tile snapshots)
  snapBrush(r, i) {
    const b = this.brushes[i], copy = buf(b.resBytes)
    this.enc().copyBufferToBuffer(b.res, 0, copy, 0, b.resBytes)
    r.brushes.push({ b: i, buf: copy }); r.bytes += b.resBytes
  }
  // swap the recorded state with the current one (undo and redo are the same operation)
  swap(r) {
    const e = this.enc()
    if (r.tiles.length) {
      const fresh = r.chunks.map(() => buf(16 * TB))
      r.tiles.forEach(({ s, t }, k) => e.copyBufferToBuffer(this.surfaces[s].pool, this.surfaces[s].slots[t] * TB, fresh[k >> 4], (k & 15) * TB, TB))
      r.tiles.forEach(({ s, t }, k) => e.copyBufferToBuffer(r.chunks[k >> 4], (k & 15) * TB, this.surfaces[s].pool, this.surfaces[s].slots[t] * TB, TB))
      for (const c of r.chunks) Engine.trash.push(c)
      r.chunks = fresh
      for (const { s, t } of r.tiles) {
        const sf = this.surfaces[s], tx = t % sf.tilesX, ty = Math.floor(t / sf.tilesX)
        this.touch(sf, rect(tx * TILE, ty * TILE, tx * TILE + TILE, ty * TILE + TILE))
        sf.packed.delete((ty >> 1) * Math.ceil(sf.w / FT) + (tx >> 1))
      }
    }
    for (const rb of r.brushes) {
      const b = this.brushes[rb.b], tmp = buf(b.resBytes)
      e.copyBufferToBuffer(b.res, 0, tmp, 0, b.resBytes)
      e.copyBufferToBuffer(rb.buf, 0, b.res, 0, b.resBytes)
      Engine.trash.push(rb.buf); rb.buf = tmp
    }
    const t = this.tubes.map(x => x.left)
    r.tubes.forEach((v, i) => { this.tubes[i].left = v }); r.tubes = t
    const c = this.canvas.mask; this.canvas.mask = r.crop; r.crop = c
    if (c.some((v, i) => v !== this.canvas.mask[i])) this.canvas.allDirty()
    const po = this.palette.origin; this.placePalette(r.pal); r.pal = po
  }
  undo() {
    if (this.cur || !this.undoStack.length) return
    const r = this.undoStack.pop()
    this.swap(r); this.redoStack.push(r)
    if (this.history.length) this.redoOps.push(this.history.pop())
    this.onChange?.()
  }
  redo() {
    if (this.cur || !this.redoStack.length) return
    const r = this.redoStack.pop()
    this.swap(r); this.undoStack.push(r)
    if (this.redoOps.length) this.history.push(this.redoOps.pop())
    this.onChange?.()
  }

  // ------------------------------------------------ display caches
  shade() {
    for (const s of this.surfaces) {
      if (empty(s.dirty)) continue
      const r = s.dirty
      const o = this.params([r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0, this.relief, this.medium === 2 ? 0.2 : 0.45, s.maskR,
        this.medium === 0 ? 1 : (this.medium === 1 ? 0.75 : 0.25), ...s.mask, s.kind === 0 ? 0.05 : 0.45, this._dullDry ? 0.85 : 1, 0, 0])
      s.bgs.shade ||= bg(L.shade, [this.pigBuf, { buffer: this.arena, size: 64 }, s.color.createView({ baseMipLevel: 0, mipLevelCount: 1 }), s.nrm.createView({ baseMipLevel: 0, mipLevelCount: 1 })])
      const p = this.enc().beginComputePass()
      p.setPipeline(G.P.shade); p.setBindGroup(0, s.bgs.shade, [o]); p.setBindGroup(1, s.bgSurf())
      p.dispatchWorkgroups(Math.ceil((r.x1 - r.x0) / 16), Math.ceil((r.y1 - r.y0) / 16))
      // rebuild the mip chain of the display caches only where they changed
      if (!s.downBG) {
        s.downBG = []
        for (let l = 1; l < s.mips; l++) {
          const v = (t, m) => t.createView({ baseMipLevel: m, mipLevelCount: 1 })
          s.downBG.push([bg(L.down8, [v(s.color, l - 1), v(s.color, l), { buffer: this.arena, size: 16 }]), bg(L.down16, [v(s.nrm, l - 1), v(s.nrm, l), { buffer: this.arena, size: 16 }])])
        }
      }
      let rr = r
      for (let l = 1; l < s.mips; l++) {
        rr = rect(rr.x0 >> 1, rr.y0 >> 1, (rr.x1 + 1) >> 1, (rr.y1 + 1) >> 1)
        const ro = this.params([rr.x0, rr.y0, rr.x1, rr.y1])
        for (const [pipe, g] of [[G.P.down8, s.downBG[l - 1][0]], [G.P.down16, s.downBG[l - 1][1]]]) {
          p.setPipeline(pipe); p.setBindGroup(0, g, [ro])
          p.dispatchWorkgroups(Math.ceil((rr.x1 - rr.x0) / 16), Math.ceil((rr.y1 - rr.y0) / 16))
        }
      }
      p.end()
      s.dirty = NONE
    }
    const p = this.enc().beginComputePass()
    p.setPipeline(G.P.brushColor)
    for (const b of this.brushes) {
      b.bgs ||= {}
      b.bgs.color ||= bg(L.brushColor, [this.pigBuf, b.res, b.capBuf, b.colors, b.nbBuf])
      p.setBindGroup(0, b.bgs.color)
      p.dispatchWorkgroups(Math.ceil((b.n + 1) / 64))
    }
    p.end()
  }

  // ------------------------------------------------ save / load
  static packHistory(ops) {
    const f = []
    for (const op of ops) {
      if (op.t === 'stroke' || op.t === 'squeeze') {
        f.push(op.t === 'stroke' ? 0 : 1, op.b, op.ss.length)
        for (const m of op.ss) f.push(m.x, m.y, m.p, m.tx, m.ty, m.roll, m.t, m.s)
      } else if (op.t === 'dry') f.push(2, op.hours, op.palette ? 1 : 0)
      else if (op.t === 'clean') f.push(3, op.b)
      else if (op.t === 'scrape') f.push(4)
      else if (op.t === 'wipe') f.push(5)
      else if (op.t === 'crop') f.push(6, ...op.rect)
      else if (op.t === 'move') f.push(7, op.x, op.y)
    }
    return new Float32Array(f)
  }
  static unpackHistory(f) {
    const ops = []
    let i = 0
    while (i < f.length) {
      const tag = f[i++]
      if (tag === 0 || tag === 1) {
        const b = f[i++], c = f[i++], ss = []
        for (let k = 0; k < c; k++, i += 8) ss.push({ x: f[i], y: f[i + 1], p: f[i + 2], tx: f[i + 3], ty: f[i + 4], roll: f[i + 5], t: f[i + 6], s: f[i + 7] })
        ops.push({ t: tag === 0 ? 'stroke' : 'squeeze', b, ss })
      } else if (tag === 2) { ops.push({ t: 'dry', hours: f[i], palette: f[i + 1] > 0 }); i += 2 }
      else if (tag === 3) ops.push({ t: 'clean', b: f[i++] })
      else if (tag === 4) ops.push({ t: 'scrape' })
      else if (tag === 5) ops.push({ t: 'wipe' })
      else if (tag === 6) { ops.push({ t: 'crop', rect: Array.from(f.slice(i, i + 4)) }); i += 4 }
      else if (tag === 7) { ops.push({ t: 'move', x: f[i], y: f[i + 1] }); i += 2 }
      else throw new Error('bad history')
    }
    return ops
  }
  async readback(src, size) {
    const st = G.dev.createBuffer({ size, usage: BU.MAP_READ | BU.COPY_DST })
    const e = this.enc()
    e.copyBufferToBuffer(src, 0, st, 0, size)
    this.submit()
    await st.mapAsync(GPUMapMode.READ)
    const out = st.getMappedRange().slice(0)
    st.destroy()
    return out
  }
  // the tooth (as half floats), read back once per surface: the file's height residual is taken against it
  async tooth(s) {
    if (s.toothCache) return s.toothCache
    const bpr = Math.ceil(s.w * 8 / 256) * 256
    const st = G.dev.createBuffer({ size: bpr * s.h, usage: BU.MAP_READ | BU.COPY_DST })
    this.enc().copyTextureToBuffer({ texture: s.ground }, { buffer: st, bytesPerRow: bpr }, [s.w, s.h])
    this.submit()
    await st.mapAsync(GPUMapMode.READ)
    const src = new Uint16Array(st.getMappedRange()), out = new Uint16Array(s.w * s.h), row = bpr / 2
    for (let y = 0; y < s.h; y++) for (let x = 0; x < s.w; x++) out[y * s.w + x] = src[y * row + x * 4]
    st.destroy()
    return (s.toothCache = out)
  }
  // geometry of a file tile: pool tiles under it (or -1 outside the surface), their valid sizes, and its 128x128 tooth
  fileTile(s, f, tooth) {
    const ftx = Math.ceil(s.w / FT), x0 = (f % ftx) * FT, y0 = (f / ftx | 0) * FT
    const v = [0, 1, 2, 3].map(k => {
      const x = x0 + (k & 1) * TILE, y = y0 + (k >> 1) * TILE
      return x < s.w && y < s.h ? [Math.min(TILE, s.w - x), Math.min(TILE, s.h - y)] : null
    })
    const t = v.map((vk, k) => vk ? ((y0 >> 6) + (k >> 1)) * s.tilesX + (x0 >> 6) + (k & 1) : -1)
    const th = new Uint16Array(FN)
    for (let y = 0; y < FT && y0 + y < s.h; y++) th.set(tooth.subarray((y0 + y) * s.w + x0, (y0 + y) * s.w + Math.min(x0 + FT, s.w)), y * FT)
    return { v, t, th }
  }
  // painted 128-tiles of a surface as a .pirl block: u32 count, then (u32 tile, u32 length, deflated tile).
  // Encoded tiles are cached until something writes to them, so saving again only re-encodes what changed.
  async surfaceBlock(s) {
    const ftx = Math.ceil(s.w / FT)
    const ftiles = [...new Set(s.tileOf.map(t => ((t / s.tilesX | 0) >> 1) * ftx + ((t % s.tilesX) >> 1)))].sort((a, b) => a - b)
    const todo = ftiles.filter(f => !s.packed.has(f))
    if (todo.length) {
      // a file tile covers four pool tiles: make sure they all exist
      for (const f of todo) { const x = (f % ftx) * FT, y = (f / ftx | 0) * FT; this.ensureTiles(s, rect(x, y, Math.min(x + FT, s.w), Math.min(y + FT, s.h))) }
      const pool = await this.readback(s.pool, s.used * TB), tooth = await this.tooth(s), amp = Math.fround(s.toothAmp)
      await Promise.all(todo.map(f => codecJob('encode', () => {
        const { v, t, th } = this.fileTile(s, f, tooth)
        const subs = t.map(ti => { if (ti < 0) return null; const b = tileBuf(); new Uint32Array(b).set(new Uint32Array(pool, s.slots[ti] * TB, TB / 4)); return b })
        return [{ subs, v, tooth: th, amp }, [...subs.filter(Boolean), th.buffer]]
      }).then(r => { s.packed.set(f, r.z); spare.push(...r.subs.filter(Boolean)) })))
    }
    const out = [new Uint32Array([ftiles.length])]
    for (const f of ftiles) { const z = s.packed.get(f); out.push(new Uint32Array([f, z.length]), z) }
    return out
  }
  async loadSurface(s, d) {
    const dv = new DataView(d.buffer, d.byteOffset, d.byteLength), n = dv.getUint32(0, true), entries = []
    for (let k = 0, o = 4; k < n; k++) { const f = dv.getUint32(o, true), l = dv.getUint32(o + 4, true); entries.push([f, d.slice(o + 8, o + 8 + l)]); o += 8 + l }
    s.slots.fill(-1); s.tileOf = []; s.packed.clear()
    s.mask = [0, 0, s.w, s.h]; s.allDirty()
    if (n) {
      const tooth = await this.tooth(s), amp = Math.fround(s.toothAmp)
      const geo = entries.map(([f]) => this.fileTile(s, f, tooth))
      const need = geo.reduce((a, g) => a + g.t.filter(t => t >= 0).length, 0)
      if (need > s.cap) s.grow(need, null)
      await Promise.all(entries.map(([f, z], e) => {
        const { v, t, th } = geo[e]
        s.packed.set(f, z)
        return codecJob('decode', () => { const out = v.map(vk => vk ? tileBuf() : null); return [{ z, v, tooth: th, amp, out }, [th.buffer, ...out.filter(Boolean)]] })
          .then(out => out.forEach((b, k) => {
            if (!b) return
            const slot = s.used
            s.slots[t[k]] = slot; s.tileOf.push(t[k])
            G.q.writeBuffer(s.pool, slot * TB, b)
            spare.push(b)
          }))
      }))
    }
    G.q.writeBuffer(s.pt, 0, s.slots)
    s.allDirty()      // frames kept drawing while the tiles arrived: redraw it all now
  }
  // .pirl v3: "PAINTSIM", then u64-length-prefixed blocks: JSON header, canvas tiles, palette tiles, brush reservoirs, history
  async encode() {
    this.submit()
    const c = this.canvas, p = this.palette
    const h = { version: 3, history: [], tubes: this.tubes.map(t => t.left), crop: c.mask, medium: this.medium,
      canvas: [c.w / PPM, c.h / PPM, c.texture], palette: [...p.origin, ...this.paletteStart] }
    const br = []
    for (const b of this.brushes) br.push(new Uint8Array(await this.readback(b.res, b.resBytes)))
    const blocks = [[new TextEncoder().encode(JSON.stringify(h))], await this.surfaceBlock(c), await this.surfaceBlock(p),
      [await through(new CompressionStream('deflate-raw'), new Uint8Array(await new Blob(br).arrayBuffer()))],
      [await through(new CompressionStream('deflate-raw'), new Uint8Array(Engine.packHistory(this.history).buffer))]]
    const parts = [new TextEncoder().encode('PAINTSIM')]
    for (const b of blocks) parts.push(new BigUint64Array([BigInt(b.reduce((a, x) => a + x.byteLength, 0))]), ...b)
    return new Blob(parts)
  }
  async decode(ab) {
    const d = new Uint8Array(ab), dv = new DataView(ab)
    if (d.length < 16 || new TextDecoder().decode(d.subarray(0, 8)) !== 'PAINTSIM') throw new Error('Not a painting file')
    const blocks = []
    for (let o = 8; o + 8 <= d.length;) {
      const n = Number(dv.getBigUint64(o, true)); o += 8
      if (o + n > d.length) throw new Error('Truncated painting file')
      blocks.push(d.subarray(o, o + n)); o += n
    }
    const h = JSON.parse(new TextDecoder().decode(blocks[0]))
    if (h.version < 3) throw new Error('This file uses the older LZFSE format: open it in the Mac app and save it again')
    if (blocks.length < 5) throw new Error('Corrupt painting file')
    this.submit()
    this.flushAll()
    const [w, hh, tex] = h.canvas || [400, 300, 0]
    if (Math.round(w * PPM) !== this.canvas.w || Math.round(hh * PPM) !== this.canvas.h || tex !== this.canvas.texture) {
      this.destroySurface(this.canvas)
      this.surfaces[0] = Engine.makeCanvas(w, hh, tex)
      this.initSurface(this.canvas)
    }
    await this.loadSurface(this.canvas, blocks[1])
    await this.loadSurface(this.palette, blocks[2])
    const br = await through(new DecompressionStream('deflate-raw'), blocks[3])
    let off = 0
    for (const b of this.brushes) { if (off + b.resBytes <= br.length) G.q.writeBuffer(b.res, 0, br, off, b.resBytes); off += b.resBytes }
    const hist = await through(new DecompressionStream('deflate-raw'), blocks[4])
    this.history = Engine.unpackHistory(new Float32Array(hist.buffer, hist.byteOffset, hist.byteLength / 4))
    h.tubes.forEach((v, i) => { if (this.tubes[i]) this.tubes[i].left = v })
    if (h.crop?.length === 4) this.canvas.mask = [...h.crop]
    const pp = h.palette?.length === 4 ? h.palette : [...Engine.paletteHome, ...Engine.paletteHome]
    this.placePalette(pp.slice(0, 2)); this.paletteStart = pp.slice(2, 4)
    this.medium = h.medium || 0
    this.redoOps = []; this.undoStack = []; this.redoStack = []
    this.cur = null; this.lastSample = null; this.undoRec = null
  }
}
