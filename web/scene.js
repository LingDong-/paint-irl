// Scene: desk, canvas, palette, tube tray, brushes. Meshes are generated procedurally.
import { G, PPM, NPIG, PIGMENTS, TUBE_ORDER, mixColor, linToSRGB, buf, bg } from './engine.js'

const BU = GPUBufferUsage, TU = GPUTextureUsage
export const TABLE_Z = -20
export const BRUSH_ORDER = [8, 7, 0, 1, 2, 3, 4, 5, 6, 9]      // brushes on the desk, left to right (also keys 1-9, 0)

// ---------------------------------------------------------------- math (column-major 4x4)
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s]
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const norm = a => { const l = Math.hypot(...a); return l > 0 ? mul(a, 1 / l) : a }
const smoothstep = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t) }
export const m4 = {
  mul(a, b) { const r = new Float32Array(16); for (let c = 0; c < 4; c++) for (let i = 0; i < 4; i++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + i] * b[c * 4 + k]; r[c * 4 + i] = s } return r },
  perspective(fovy, aspect, n, f) { const y = 1 / Math.tan(fovy / 2), x = y / aspect, z = f / (n - f); return new Float32Array([x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, -1, 0, 0, z * n, 0]) },
  ortho(l, r, b, t, n, f) { return new Float32Array([2 / (r - l), 0, 0, 0, 0, 2 / (t - b), 0, 0, 0, 0, -1 / (f - n), 0, -(r + l) / (r - l), -(t + b) / (t - b), -n / (f - n), 1]) },
  lookAt(eye, c, up) {
    const f = norm(sub(c, eye)), s = norm(cross(f, up)), u = cross(s, f)
    return new Float32Array([s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0, -dot(s, eye), -dot(u, eye), dot(f, eye), 1])
  },
  frame(x, y, z, o) { return new Float32Array([...x, 0, ...y, 0, ...z, 0, ...o, 1]) },
  translate(t) { return m4.frame([1, 0, 0], [0, 1, 0], [0, 0, 1], t) },
  scale(s) { return new Float32Array([s[0], 0, 0, 0, 0, s[1], 0, 0, 0, 0, s[2], 0, 0, 0, 0, 1]) },
  id() { return m4.scale([1, 1, 1]) },
  apply(m, v) { const r = [0, 0, 0, 0]; for (let i = 0; i < 4; i++) for (let k = 0; k < 4; k++) r[i] += m[k * 4 + i] * v[k]; return r },
  invert(m) {
    const a = m, inv = new Float32Array(16)
    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10]
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10]
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9]
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9]
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10]
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10]
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9]
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9]
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6]
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6]
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5]
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5]
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6]
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6]
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5]
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5]
    const det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12]
    return inv.map(v => v / det)
  },
}

// ---------------------------------------------------------------- meshes (8 floats per vertex: p.xyz, u, n.xyz, v)
// parametric surface (u around/across, v along) -> triangle list with finite-difference normals
function sweep(nu, nv, f, flip = false) {
  const grid = []
  for (let j = 0; j <= nv; j++) for (let i = 0; i <= nu; i++) {
    const u = i / nu, v = j / nv, p = f(u, v), e = 1e-3
    const du = sub(f(u + e, v), f(u - e, v))
    let dv = sub(f(u, Math.min(v + e, 1)), f(u, Math.max(v - e, 0)))
    if (Math.hypot(...dv) < 1e-7) dv = sub(f(u, Math.min(v + 0.02, 1)), f(u, Math.max(v - 0.02, 0)))
    let n = cross(du, dv)
    n = Math.hypot(...n) > 1e-9 ? norm(n) : [0, 0, 1]
    if (flip) n = mul(n, -1)
    grid.push([...p, u, ...n, v])
  }
  const out = [], W = nu + 1
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
    const a = grid[j * W + i], b = grid[j * W + i + 1], c = grid[(j + 1) * W + i], d = grid[(j + 1) * W + i + 1]
    out.push(...(flip ? [a, c, b, b, c, d] : [a, b, c, b, d, c]))
  }
  return out
}
function lathe(nu, prof) {     // [radius, z] pairs
  return sweep(nu, prof.length - 1, (u, v) => {
    const fi = v * (prof.length - 1), i = Math.min(Math.floor(fi), prof.length - 2), t = fi - i
    const r = prof[i][0] + (prof[i + 1][0] - prof[i][0]) * t, z = prof[i][1] + (prof[i + 1][1] - prof[i][1]) * t
    const a = u * 2 * Math.PI
    return [Math.cos(a) * r, Math.sin(a) * r, z]
  })
}
function disk(nu, r, z, up) {
  const out = [], n = [0, 0, up ? 1 : -1]
  for (let i = 0; i < nu; i++) {
    const a0 = i / nu * 2 * Math.PI, a1 = (i + 1) / nu * 2 * Math.PI
    const c = [0, 0, z, 0, ...n, 0], p0 = [Math.cos(a0) * r, Math.sin(a0) * r, z, 0, ...n, 0], p1 = [Math.cos(a1) * r, Math.sin(a1) * r, z, 0, ...n, 0]
    out.push(...(up ? [c, p0, p1] : [c, p1, p0]))
  }
  return out
}
function box(lo, hi) {
  const out = []
  const quad = (a, b, c, d, n, uv) => out.push([...a, 0, ...n, 0], [...b, uv[0], ...n, 0], [...c, uv[0], ...n, uv[1]], [...a, 0, ...n, 0], [...c, uv[0], ...n, uv[1]], [...d, 0, ...n, uv[1]])
  const [x0, y0, z0] = lo, [x1, y1, z1] = hi
  quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, 1], [x1 - x0, y1 - y0])
  quad([x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0], [0, 0, -1], [y1 - y0, x1 - x0])
  quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], [0, -1, 0], [x1 - x0, z1 - z0])
  quad([x1, y1, z0], [x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [0, 1, 0], [x1 - x0, z1 - z0])
  quad([x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [-1, 0, 0], [y1 - y0, z1 - z0])
  quad([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1], [1, 0, 0], [y1 - y0, z1 - z0])
  return out
}
function roundedOutline(x0, y0, x1, y1, r) {
  const pts = []
  for (const [c, a0] of [[[x1 - r, y0 + r], -Math.PI / 2], [[x1 - r, y1 - r], 0], [[x0 + r, y1 - r], Math.PI / 2], [[x0 + r, y0 + r], Math.PI]])
    for (let k = 0; k <= 10; k++) { const a = a0 + k / 10 * Math.PI / 2; pts.push([c[0] + Math.cos(a) * r, c[1] + Math.sin(a) * r]) }
  pts.push(pts[0])
  return pts
}
// side walls of a rounded rectangle slab
function roundedWall(x0, y0, x1, y1, r, z0, z1) {
  const pts = roundedOutline(x0, y0, x1, y1, r), out = []
  let acc = 0
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1], l = Math.hypot(b[0] - a[0], b[1] - a[1]), d = [(b[0] - a[0]) / l, (b[1] - a[1]) / l], n = [d[1], -d[0], 0]
    const va = [a[0], a[1], z0, acc, ...n, 0], vb = [b[0], b[1], z0, acc + l, ...n, 0], vc = [b[0], b[1], z1, acc + l, ...n, 1], vd = [a[0], a[1], z1, acc, ...n, 1]
    out.push(va, vb, vc, va, vc, vd)
    acc += l
  }
  return out
}

// brush handle + ferrule in brush local space: z up along the handle, bristles exit at z = 0
function brushMeshes(b) {
  const rf = b.round ? b.width * 0.5 + 0.35 : b.width * 0.24 + 1.6
  const lf = b.round ? 14 + b.width * 1.5 : 16 + b.width * 0.9
  const p = b.round ? 2 : 3.5
  const ferrule = sweep(48, 24, (u, v) => {
    const a = u * 2 * Math.PI, s = smoothstep(0, 0.55, v)
    const ax = b.round ? (b.width * 0.5 + 0.3) * (1 - s) + rf * s : (b.width * 0.5 + 0.25) * (1 - s) + rf * s
    const ay = b.round ? ax : (b.thick * 0.5 + 0.3) * (1 - s) + rf * s
    const pe = b.round ? 2 : p * (1 - s) + 2 * s
    const c = Math.cos(a), sn = Math.sin(a)
    const cx = Math.sign(c || 1) * Math.pow(Math.abs(c), 2 / pe), cy = Math.sign(sn || 1) * Math.pow(Math.abs(sn), 2 / pe)
    const crimp = (v > 0.6 && v < 0.66) || (v > 0.7 && v < 0.74) ? 0.94 : 1
    return [cx * ax * crimp, cy * ay * crimp, v * lf]
  })
  const hl = b.handleLen, rh = rf * 0.98
  const prof = [[rh * 0.9, lf - 1], [rh, lf], [rh * 1.18, lf + hl * 0.12], [rh * 1.28, lf + hl * 0.28], [rh * 1.12, lf + hl * 0.5],
    [rh * 0.85, lf + hl * 0.8], [rh * 0.7, lf + hl * 0.97], [rh * 0.5, lf + hl], [0.01, lf + hl + 0.5]]
  return [[...ferrule, ...disk(48, rf, lf, true)], lathe(40, prof)]
}
const ferruleLen = b => b.pencil ? 0 : b.knife ? 10 : (b.round ? 14 + b.width * 1.5 : 16 + b.width * 0.9)

// palette knife: wooden handle, short collar, thin trowel blade running from the collar to the tip (local -z)
function knifeMeshes(b) {
  const collar = [...lathe(32, [[2.0, -0.5], [3.6, 1.5], [4.2, 6], [4.2, 10]]), ...disk(32, 4.2, 10, true)]
  const hl = b.handleLen
  const handle = lathe(40, [[4.2, 9.5], [4.6, 12], [5.6, 10 + hl * 0.35], [5.2, 10 + hl * 0.8], [3.8, 10 + hl], [0.01, 10 + hl + 1]])
  const blade = sweep(24, 40, (u, v) => {
    const a = u * 2 * Math.PI
    const w = v < 0.12 ? 1.6 : b.width * 0.5 * (1 - 0.55 * Math.pow(Math.abs(v - 0.45) / 0.55, 2)) * smoothstep(0.1, 0.3, v) + 1.6 * (1 - smoothstep(0.1, 0.3, v))
    const tip = 1 - smoothstep(0.9, 1.0, v) * 0.85
    return [Math.cos(a) * w * tip, Math.sin(a) * 0.35, -v * b.length]
  })
  return [collar, handle, blade]
}

// pencil: hexagonal lacquered body from z = 0 up, sharpened wood cone down to the graphite point at z = -length
function pencilMeshes(b) {
  const R = 4, L = b.length, lead = 2.5, s = Math.PI / 3
  const hex = a => R * Math.cos(s / 2) / Math.cos((a % s) - s / 2)
  const body = [...sweep(48, 1, (u, v) => { const a = u * 2 * Math.PI; return [Math.cos(a) * hex(a), Math.sin(a) * hex(a), v * b.handleLen] }), ...disk(6, R * 0.98, b.handleLen, true)]
  // the knife cuts a cone; where it meets the flats it leaves a scalloped edge
  const wood = sweep(48, 12, (u, v) => { const a = u * 2 * Math.PI, r = Math.min(0.55 + (R - 0.55) * v, hex(a)); return [Math.cos(a) * r, Math.sin(a) * r, -(L - lead) * (1 - v)] })
  const tip = lathe(24, [[0.05, -L], [0.3, -L + 0.6], [0.55, -L + lead]])
  return [wood, body, tip]
}

// paint tube: nozzle at z=0 pointing -z, crimp end at z = 112
const TUBE_R = 9.5
function tubeBody(left, seed) {
  const f0 = 0.5 + 0.43 * left
  return sweep(48, 60, (u, v) => {
    const a = u * 2 * Math.PI, z = 6 + v * 106, s = smoothstep(f0, 0.93, v)
    const dent = (1 - left) * 0.12 * Math.sin(v * 23 + seed) * Math.sin(a * 3 + seed * 2)
    return [Math.cos(a) * TUBE_R * (1 + 0.55 * s), Math.sin(a) * Math.max(TUBE_R * (1 - 0.95 * s) * (1 + dent), 0.5), z]
  })
}
const tubeShoulder = () => [...lathe(48, [[3.6, -6], [4.0, -5.5], [4.0, -1], [4.4, -0.5], [6.5, 1.8], [8.6, 4.2], [TUBE_R, 6.2]]), ...disk(24, 3.6, -6, false)]
const tubeCap = () => [...lathe(64, [[6.4, -17], [6.9, -16.6], [7.0, -2], [6.6, -1.2]]), ...disk(32, 6.4, -17, false)]

let meshIds = 0
class Mesh {
  constructor(v) {
    this.id = ++meshIds
    this.count = v.length
    const f = new Float32Array(Math.max(v.length, 1) * 8)
    v.forEach((x, i) => f.set(x, i * 8))
    this.buf = buf(f.byteLength, BU.STORAGE, f)
    this.bg = bg(G.L.mesh, [this.buf])
  }
  destroy() { this.buf.destroy() }
}

// ---------------------------------------------------------------- labels (canvas text into an atlas)
const LABEL_W = 512, LABEL_H = 160
function makeLabels() {
  const W = LABEL_W, H = LABEL_H * NPIG
  const cv = new OffscreenCanvas(W, H), ctx = cv.getContext('2d')
  const css = c => `rgb(${c.map(v => Math.round(linToSRGB(v) * 255)).join(',')})`
  const font = (w, s) => `${w} ${s}px "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif`
  PIGMENTS.forEach((p, i) => {
    const y0 = i * LABEL_H
    ctx.fillStyle = 'rgb(237,233,222)'; ctx.fillRect(0, y0, W, LABEL_H)
    const v = new Array(NPIG).fill(0); v[i] = 1
    const tv = new Array(NPIG).fill(0); tv[i] = 0.3; tv[0] += 0.7
    // colour band near the shoulder: masstone with a strip of tint
    ctx.fillStyle = css(mixColor(v, 60)); ctx.fillRect(0, y0, 150, LABEL_H)
    ctx.fillStyle = css(mixColor(tv, 60)); ctx.fillRect(150, y0, 14, LABEL_H)
    const ink = 'rgb(31,28,26)'
    ctx.fillStyle = ink; ctx.fillRect(172, y0 + 46.5, 330, 1.5); ctx.fillRect(172, y0 + 112, 330, 1.5)
    const name = p.name.toUpperCase()
    let fs = 26
    while (fs > 12) { ctx.font = font(600, fs); if (ctx.measureText(name).width < 325) break; fs-- }
    ctx.fillText(name, 174, y0 + 88)
    ctx.font = font(500, 13); ctx.fillText("ARTISTS' OIL COLOUR", 174, y0 + 38)
    ctx.font = font(400, 13); ctx.fillText(`${p.code}   ·   37 ml   ·   series ${1 + i % 4}`, 174, y0 + 136)
  })
  const levels = Math.floor(Math.log2(Math.max(W, H))) + 1
  const t = G.dev.createTexture({ size: [W, H], format: 'rgba8unorm-srgb', mipLevelCount: levels, usage: TU.TEXTURE_BINDING | TU.COPY_DST })
  let src = cv
  for (let l = 0; l < levels; l++) {
    const w = Math.max(1, W >> l), h = Math.max(1, H >> l)
    let c = src
    if (l > 0) { c = new OffscreenCanvas(w, h); const x = c.getContext('2d'); x.imageSmoothingQuality = 'high'; x.drawImage(src, 0, 0, w, h) }
    const d = c.getContext('2d').getImageData(0, 0, w, h)
    G.q.writeTexture({ texture: t, mipLevel: l }, d.data, { bytesPerRow: w * 4 }, [w, h])
    src = c
  }
  return t
}

// ---------------------------------------------------------------- camera
export class Camera {
  constructor() { this.center = [90, -45]; this.dist = 980; this.fov = 26 * Math.PI / 180 }
  get eye() { return [this.center[0], this.center[1], this.dist] }
  viewProj(aspect) { return m4.mul(m4.perspective(this.fov, aspect, Math.max(this.dist * 0.02, 2), this.dist + 400), m4.lookAt(this.eye, [this.center[0], this.center[1], 0], [0, 1, 0])) }
  // ray through normalised device coords
  ray(ndc, aspect) {
    const inv = m4.invert(this.viewProj(aspect))
    const a = m4.apply(inv, [ndc[0], ndc[1], 0, 1]), b = m4.apply(inv, [ndc[0], ndc[1], 1, 1])
    const p0 = [a[0] / a[3], a[1] / a[3], a[2] / a[3]], p1 = [b[0] / b[3], b[1] / b[3], b[2] / b[3]]
    return [p0, norm(sub(p1, p0))]
  }
}
export const hitPlane = (r, z) => { const t = (z - r[0][2]) / r[1][2]; return [r[0][0] + r[1][0] * t, r[0][1] + r[1][1] * t] }

// ---------------------------------------------------------------- scene
const HPT = 3     // hairs per tuft
export class Scene {
  constructor(eng) {
    this.eng = eng
    this.cam = new Camera()
    this.held = { brush: 1 }
    this.cursor = [0, 0, 0]
    this.heat = false
    this.hideMode = 0          // held brush: 0 whole, 1 without handle, 2 hidden
    this.time = 0; this.debug = 0
    this.exportView = null
    this.sigF = new Float32Array(1); this.sigU = new Uint32Array(this.sigF.buffer)
    this.lightDir = norm([-0.42, 0.52, 0.74])
    const dev = G.dev, L = G.L
    this.colorFormat = G.format.replace(/-srgb$/, '') + '-srgb'
    const pipe = (name, layouts, { topo = 'triangle-list', cull = 'none', shadow = false } = {}) => dev.createRenderPipeline({
      layout: dev.createPipelineLayout({ bindGroupLayouts: layouts }),
      vertex: { module: G.module(name), entryPoint: 'vs' },
      fragment: shadow ? undefined : { module: G.module(name), entryPoint: 'fs', targets: [{ format: this.colorFormat }] },
      primitive: { topology: topo, cullMode: cull, frontFace: 'ccw' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less-equal', ...(shadow ? { depthBias: 1, depthBiasSlopeScale: 2, depthBiasClamp: 0.01 } : {}) },
      multisample: { count: shadow ? 1 : 4 },
    })
    this.surfPipe = pipe('surf', [L.frameFull, L.surfDraw], { topo: 'triangle-strip' })
    this.objPipe = pipe('obj', [L.frameFull, L.mesh, L.objDraw], { cull: 'back' })
    this.hairPipe = pipe('hair', [L.frameFull, L.hair], { topo: 'triangle-strip' })
    this.shadowPipe = pipe('shadow', [L.frameOnly, L.mesh, L.objDraw], { shadow: true })
    const st = n => dev.createTexture({ size: [n, n], format: 'depth32float', usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING })
    this.shadowTex = st(2048); this.shadowHeld = st(2048)
    this.shadowSig = [NaN, NaN]        // what each shadow map holds; redrawn only when that changes
    const noShadow = st(1)             // export: no shadows
    this.labels = makeLabels()
    this.frameBuf = buf(256, BU.UNIFORM | BU.COPY_DST)
    this.drawBuf = buf(256 * 256, BU.UNIFORM | BU.COPY_DST)
    this.drawAB = new Float32Array(64 * 256)
    const samplers = [dev.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' }),
      dev.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 8 })]
    this.frameFull = bg(L.frameFull, [this.frameBuf, this.shadowTex.createView(), this.shadowHeld.createView(), ...samplers, this.labels.createView()])
    this.frameExport = bg(L.frameFull, [this.frameBuf, noShadow.createView(), noShadow.createView(), ...samplers, this.labels.createView()])
    const e = dev.createCommandEncoder()
    e.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: noShadow.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 } }).end()
    G.q.submit([e.finish()])
    this.frameOnly = bg(L.frameOnly, [this.frameBuf])
    this.drawBG = bg(L.objDraw, [{ buffer: this.drawBuf, size: 112 }])
    this.buildMeshes()
    // hair records, one aligned range per brush
    let off = 0
    this.hairOff = eng.brushes.map(b => { const o = off; off += Math.ceil(b.n * HPT * 64 / 256) * 256; return o })
    this.hairBuf = buf(off, BU.STORAGE | BU.COPY_DST)
    this.hairAB = new Float32Array(off / 4)
    this.hairBG = eng.brushes.map((b, i) => bg(L.hair, [{ buffer: this.hairBuf, offset: this.hairOff[i], size: b.n * HPT * 64 }, b.colors]))
    eng.brushes.forEach((b, i) => this.restPose(b, i))
  }

  buildMeshes() {
    const M = this.meshes = {}
    M.table = new Mesh(box([-2500, -2500, TABLE_Z - 30], [2500, 2500, TABLE_Z]))
    // unit square outline (x,y in 0..1) extruded z -1..0
    M.unitSides = new Mesh(sweep(4, 1, (u, v) => {
      const c = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], t = u * 4, k = Math.min(Math.max(Math.trunc(t), 0), 3), f = t - k
      return [c[k][0] + (c[k + 1][0] - c[k][0]) * f, c[k][1] + (c[k + 1][1] - c[k][1]) * f, v - 1]
    }).map(x => [x[0], x[1], x[2], x[3], -x[4], -x[5], -x[6], x[7]]))
    M.unitTop = new Mesh(box([0, 0, -1], [1, 1, 0]))
    const p = this.eng.palette
    const x1 = p.w / PPM, y1 = p.h / PPM, th = 5.5
    const hc = [p.hole[0] / PPM, p.hole[1] / PPM], hr = p.hole[2] / PPM
    const holeWall = sweep(48, 1, (u, v) => [hc[0] + Math.cos(u * 2 * Math.PI) * hr, hc[1] + Math.sin(u * 2 * Math.PI) * hr, -th + v * th], true)
    M.palette = new Mesh([...roundedWall(0, 0, x1, y1, p.maskR / PPM, -th, 0), ...holeWall])
    // shadow caster: the top face with the thumb hole cut out (ring between the outline and the hole, by angle around it)
    const o = roundedOutline(0, 0, x1, y1, p.maskR / PPM), cap = []
    const ring = q => { const d = [q[0] - hc[0], q[1] - hc[1]], l = Math.hypot(...d); return [hc[0] + d[0] / l * hr, hc[1] + d[1] / l * hr] }
    for (let i = 0; i < o.length - 1; i++) {
      const k = Math.max(1, Math.floor(Math.hypot(o[i + 1][0] - o[i][0], o[i + 1][1] - o[i][1]) / 4))
      for (let j = 0; j < k; j++) {
        const a = [o[i][0] + (o[i + 1][0] - o[i][0]) * j / k, o[i][1] + (o[i + 1][1] - o[i][1]) * j / k]
        const b = [o[i][0] + (o[i + 1][0] - o[i][0]) * (j + 1) / k, o[i][1] + (o[i + 1][1] - o[i][1]) * (j + 1) / k]
        for (const q of [ring(a), a, b, ring(a), b, ring(b)]) cap.push([q[0], q[1], 0, 0, 0, 0, 1, 0])
      }
    }
    M.paletteCap = new Mesh(cap)
    // tube tray
    const tx0 = 238, tx1 = 512, ty0 = -264, ty1 = -150, Z = TABLE_Z
    M.tray = new Mesh([...box([tx0, ty0, Z], [tx1, ty1, Z + 3]), ...box([tx0, ty0, Z], [tx1, ty0 + 6, Z + 16]), ...box([tx0, ty1 - 6, Z], [tx1, ty1, Z + 16]),
      ...box([tx0, ty0, Z], [tx0 + 6, ty1, Z + 16]), ...box([tx1 - 6, ty0, Z], [tx1, ty1, Z + 16])])
    M.shoulder = new Mesh(tubeShoulder())
    M.cap = new Mesh(tubeCap())
    this.tubeMesh = this.eng.tubes.map((t, i) => new Mesh(tubeBody(t.left, i)))
    this.tubeLeft = this.eng.tubes.map(t => t.left)
    this.brushMesh = this.eng.brushes.map(b => (b.knife ? knifeMeshes(b) : b.pencil ? pencilMeshes(b) : brushMeshes(b)).map(m => new Mesh(m)))
  }

  // ------------------------------------------------ placement
  static TS = 0.72     // small (21 ml) tubes
  tubeRest(i) {
    const TS = Scene.TS, x = 250 + TUBE_ORDER.indexOf(i) * 16.6
    return m4.frame([-TS, 0, 0], [0, 0, -TS], [0, -TS, 0], [x, -175, TABLE_Z + 3 + TUBE_R * 0.92 * TS])    // caps toward the palette, label facing up
  }
  tubeHeld(at) {
    const TS = Scene.TS, ax = norm([0.32, -0.42, 0.85]), x = norm(cross([0, 0, 1], ax)), y = cross(ax, x)
    return m4.frame(mul(x, TS), mul(y, TS), mul(ax, TS), add(at, mul(ax, 2)))
  }
  setHeld(h) {
    if (h.brush === this.held.brush && h.tube === this.held.tube) return
    if (this.held.brush !== undefined) this.restPose(this.eng.brushes[this.held.brush], this.held.brush)
    this.held = h
  }
  // brushes lie below the canvas with their bristles pointing at it
  brushRestPos(i) { const b = this.eng.brushes[i]; return [-175 + BRUSH_ORDER.indexOf(i) * 26, -176 - b.length, b.knife ? 6 : (b.pencil ? 3.5 : 4.6)] }    // z relative to the table
  restPose(b, i) {
    b.baseZ = TABLE_Z
    b.axis = [0, 1, 0]; b.e1 = [1, 0, 0]; b.e2 = cross(b.axis, b.e1)
    b.ferrule = this.brushRestPos(i)
    b.simulate(null, 0, null)
  }
  brushModel(b) { return m4.frame(b.e1, mul(b.e2, -1), mul(b.axis, -1), add(b.ferrule, [0, 0, b.baseZ])) }
  // tube body regenerates as paint gets used
  refreshTubes() {
    this.eng.tubes.forEach((t, i) => {
      if (Math.abs(t.left - this.tubeLeft[i]) > 0.004) { this.tubeMesh[i].destroy(); this.tubeMesh[i] = new Mesh(tubeBody(t.left, i)); this.tubeLeft[i] = t.left }
    })
  }

  // frame the canvas, palette, tube tray and the bristle ends of the brushes
  fitDesk(aspect) {
    const c = this.eng.canvas, p = this.eng.palette
    this.fit([Math.min(c.origin[0] - 15, -195), -272], [p.origin[0] + p.w / PPM + 12, Math.max(c.origin[1] + c.h / PPM, 120) + 15], aspect)
  }
  fitCanvas(aspect) {
    const c = this.eng.canvas, lo = c.toWorld([c.mask[0], c.mask[1]]), hi = c.toWorld([c.mask[2], c.mask[3]])
    this.fit([lo[0] - 12, lo[1] - 12], [hi[0] + 12, hi[1] + 12], aspect)
  }
  fit(lo, hi, aspect) {
    const t = Math.tan(this.cam.fov / 2)
    this.cam.center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2]
    this.cam.dist = Math.max((hi[0] - lo[0]) / (2 * t * aspect), (hi[1] - lo[1]) / (2 * t))
  }
  project(w, size) {
    const c = m4.apply(this.cam.viewProj(size[0] / size[1]), [...w, 1])
    return [(c[0] / c[3] * 0.5 + 0.5) * size[0], (0.5 - c[1] / c[3] * 0.5) * size[1]]
  }

  // ------------------------------------------------ picking
  // what a click picks up; clicking the empty slot of the held thing puts it down ({})
  pick(r) {
    const pt = hitPlane(r, TABLE_Z + 10)
    for (let i = 0; i < this.eng.tubes.length; i++) {
      const c = this.tubeRest(i).slice(12, 15)
      if (Math.abs(pt[0] - c[0]) < 7.5 && pt[1] > c[1] - 82 && pt[1] < c[1] + 13) return this.held.tube === i ? {} : { tube: i }
    }
    const pb = hitPlane(r, TABLE_Z + 4)
    for (let i = 0; i < this.eng.brushes.length; i++) {
      const p = this.brushRestPos(i), b = this.eng.brushes[i]
      if (Math.abs(pb[0] - p[0]) < (b.knife ? 11 : 6.5) && pb[1] < p[1] + b.length && pb[1] > p[1] - ferruleLen(b) - b.handleLen) return this.held.brush === i ? {} : { brush: i }
    }
    return null
  }

  // ------------------------------------------------ render
  hairs(b, f, o) {
    const z = b.baseZ, R0 = b.rads, e1 = b.e1, e2 = b.e2
    let k = 0
    for (let j = 0; j < b.n; j++) {
      const R = [b.root[3 * j], b.root[3 * j + 1], b.root[3 * j + 2] + z], L = b.lens[j]
      let p1, p2, p3
      if (b.touching[j]) {
        p3 = [b.tip[2 * j], b.tip[2 * j + 1], z + 0.08]
        const bp = [b.bendPt[2 * j], b.bendPt[2 * j + 1], z + 0.1]
        p1 = add(R, mul(b.axis, L * 0.35))
        const d = sub(p1, bp), dl = Math.hypot(...d)
        p2 = add(bp, mul(norm(d), Math.min(dl * 0.3, L * 0.2)))
        p2[2] = Math.max(p2[2], z + 0.1)
      } else {
        p3 = [b.tip[2 * j], b.tip[2 * j + 1], b.tipZ[j] + z]
        p1 = add(R, mul(sub(p3, R), 0.33)); p2 = add(R, mul(sub(p3, R), 0.66))
      }
      for (let h = 0; h < HPT; h++, k++) {
        const g = b.hairR, gi = k * 9, r = R0[j]
        const ox = (e1[0] * g[gi] + e2[0] * g[gi + 1]) * r * 0.6, oy = (e1[1] * g[gi] + e2[1] * g[gi + 1]) * r * 0.6, oz = (e1[2] * g[gi] + e2[2] * g[gi + 1]) * r * 0.6
        const tx = (e1[0] * g[gi + 2] + e2[0] * g[gi + 3]) * r * 0.9, ty = (e1[1] * g[gi + 2] + e2[1] * g[gi + 3]) * r * 0.9, tz = (e1[2] * g[gi + 2] + e2[2] * g[gi + 3]) * r * 0.9
        f.set([R[0] + ox, R[1] + oy, R[2] + oz, j, p1[0] + ox * 0.8, p1[1] + oy * 0.8, p1[2] + oz * 0.8, r * 0.9,
          p2[0] + tx * 0.6, p2[1] + ty * 0.6, p2[2] + tz * 0.6, 0.5, p3[0] + tx * 0.4, p3[1] + ty * 0.4, p3[2] + tz * 0.4, g[gi + 8]], o + k * 16)
      }
    }
    return k
  }

  frameUniforms(aspect) {
    const cv = this.eng.canvas, pl = this.eng.palette
    const lo = [Math.min(cv.origin[0], pl.origin[0], -200), Math.min(cv.origin[1], pl.origin[1], -290)]
    const hi = [Math.max(cv.origin[0] + cv.w / PPM, pl.origin[0] + pl.w / PPM, 570), Math.max(cv.origin[1] + cv.h / PPM, pl.origin[1] + pl.h / PPM, 130)]
    const m = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2], ext = Math.max(560, Math.max(hi[0] - lo[0], hi[1] - lo[1]) * 0.5 + 60)
    const c = [m[0], m[1], TABLE_Z]
    const lvp = m4.mul(m4.ortho(-ext, ext, -ext, ext, 400, 2600), m4.lookAt(add(c, mul(this.lightDir, 1500)), c, [0, 1, 0]))
    const ev = this.exportView
    return new Float32Array([...(ev ? ev.vp : this.cam.viewProj(aspect)), ...lvp, ...(ev ? ev.eye : this.cam.eye), 1, ...this.lightDir, 0,
      1.12, 0.95 * 1.12, 0.88 * 1.12, 0, 0.30 * 0.62, 0.31 * 0.62, 0.34 * 0.62, 0, this.heat ? 1 : 0, this.time, 0, this.debug])
  }

  drawList() {
    const list = [], E = this.eng
    const add_ = (mesh, model, base, { metal = 0, rough = 0.5, mode = 0, fade = 0, extra = [0, 0, 0, 0], shadow = true, main = true } = {}) =>
      list.push({ mesh, d: [...model, ...base, 1, metal, rough, mode, fade, ...extra], shadow, main, isHeld: fade > 0 })
    const M = this.meshes
    add_(M.table, m4.id(), [0.4, 0.25, 0.15], { rough: 0.55, mode: 5, shadow: false })
    const cv = E.canvas, m = cv.mask
    const cx0 = cv.origin[0] + m[0] / PPM, cy0 = cv.origin[1] + m[1] / PPM, cw = (m[2] - m[0]) / PPM, ch = (m[3] - m[1]) / PPM
    const cm = m4.mul(m4.translate([cx0, cy0, 0]), m4.scale([cw, ch, -TABLE_Z]))
    add_(M.unitSides, cm, [0.86, 0.84, 0.79], { rough: 0.9, mode: 4, extra: [cw * 0.75, 15, 0, 0] })
    add_(M.unitTop, cm, [0, 0, 0], { main: false })
    const pm = m4.translate([E.palette.origin[0], E.palette.origin[1], E.palette.z])
    add_(M.palette, pm, [0.74, 0.55, 0.34], { rough: 0.4, mode: 2, extra: [1, 0, 0, 2] })
    add_(M.paletteCap, pm, [0, 0, 0], { main: false })
    add_(M.tray, m4.id(), [0.55, 0.40, 0.26], { rough: 0.6, mode: 2, extra: [1, 30, 0, 5] })
    for (let i = 0; i < E.tubes.length; i++) {
      const inTray = this.held.tube !== i
      const model = inTray ? this.tubeRest(i) : this.tubeHeld(this.cursor)
      const fade = inTray ? 0 : 160
      add_(this.tubeMesh[i], model, [0.85, 0.85, 0.85], { metal: 1, rough: 0.3, mode: 1, fade, extra: [0.16, 0.84, i / NPIG, 1 / NPIG] })
      add_(M.shoulder, model, [0.86, 0.86, 0.85], { metal: 1, rough: 0.25, fade })
      add_(M.cap, this.tubeRest(i), [0.03, 0.03, 0.03], { rough: 0.35, mode: 3 })
    }
    E.brushes.forEach((b, i) => {
      let fade = 0, hide = 0
      if (this.held.brush === i) { fade = 140; hide = this.hideMode }
      const model = this.brushModel(b), bm = this.brushMesh[i]
      if (hide < 2 && b.pencil) {
        add_(bm[0], model, [0.78, 0.60, 0.42], { rough: 0.8, fade: fade * 0.6 })
        add_(bm[2], model, [0.09, 0.09, 0.10], { metal: 0.6, rough: 0.35, fade: fade > 0 ? 1 : 0 })
      } else if (hide < 2) {
        add_(bm[0], model, [0.80, 0.78, 0.74], { metal: 1, rough: 0.22, fade: fade * 0.6 })
        if (b.knife) add_(bm[2], model, [0.82, 0.82, 0.83], { metal: 1, rough: 0.18, mode: 10, fade: fade > 0 ? 1 : 0 })
      }
      if (hide < 1) add_(bm[1], model, b.handleColor, { rough: b.knife ? 0.45 : (b.pencil ? 0.3 : 0.18), fade })
    })
    return list
  }

  // encode shadows + the main pass into `enc`, drawing into the given MSAA/resolve/depth views
  encode(enc, target, aspect) {
    this.refreshTubes()
    this.frameF = this.frameUniforms(aspect)
    G.q.writeBuffer(this.frameBuf, 0, this.frameF)
    const list = this.exportView ? [] : this.drawList()
    list.forEach((d, i) => this.drawAB.set(d.d, i * 64))
    if (list.length) G.q.writeBuffer(this.drawBuf, 0, this.drawAB, 0, list.length * 64)
    // shadow maps: the resting things and the held thing, each redrawn only when what casts into it moved
    if (!this.exportView) [[this.shadowTex, false], [this.shadowHeld, true]].forEach(([tex, held], k) => {
      const lvp = this.frameF.subarray(16, 32)
      let h = 2166136261
      const mix = f => { this.sigF[0] = f; h = Math.imul(h ^ this.sigU[0], 16777619) }
      lvp.forEach(mix)
      for (const d of list) if (d.shadow && d.isHeld === held) { mix(d.mesh.id); d.d.forEach(mix) }
      if (h === this.shadowSig[k]) return
      this.shadowSig[k] = h
      const p = enc.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: tex.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 } })
      p.setPipeline(this.shadowPipe); p.setBindGroup(0, this.frameOnly)
      list.forEach((d, i) => { if (d.shadow && d.isHeld === held) { p.setBindGroup(1, d.mesh.bg); p.setBindGroup(2, this.drawBG, [i * 256]); p.draw(d.mesh.count) } })
      p.end()
    })
    const p = enc.beginRenderPass({
      colorAttachments: [{ view: target.msaa, resolveTarget: target.resolve, loadOp: 'clear', storeOp: 'discard', clearValue: [0.1, 0.1, 0.1, 1] }],
      depthStencilAttachment: { view: target.depth, depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 1 },
    })
    p.setBindGroup(0, this.exportView ? this.frameExport : this.frameFull)
    // objects
    p.setPipeline(this.objPipe)
    list.forEach((d, i) => { if (d.main) { p.setBindGroup(1, d.mesh.bg); p.setBindGroup(2, this.drawBG, [i * 256]); p.draw(d.mesh.count) } })
    // paint surfaces
    p.setPipeline(this.surfPipe)
    for (const s of this.eng.surfaces) {
      if (this.exportView && s.kind !== 0) continue
      s.sdBuf ||= buf(64, BU.UNIFORM | BU.COPY_DST)
      s.drawBG ||= bg(G.L.surfDraw, [s.sdBuf, s.color.createView(), s.nrm.createView(), (s.prev || G.dummy).createView()])
      G.q.writeBuffer(s.sdBuf, 0, new Float32Array([...s.origin, s.w, s.h, ...s.mask, s.z, s.maskR, s.kind, 0, ...s.hole]))
      p.setBindGroup(1, s.drawBG)
      p.draw(4)
    }
    // bristles
    if (!this.exportView) {
      p.setPipeline(this.hairPipe)
      this.eng.brushes.forEach((b, i) => {
        if (b.knife || b.pencil || (this.held.brush === i && this.hideMode === 2)) return
        if (b.moved) {      // resting brushes keep their hairs
          b.hairN = this.hairs(b, this.hairAB, this.hairOff[i] / 4)
          G.q.writeBuffer(this.hairBuf, this.hairOff[i], this.hairAB, this.hairOff[i] / 4, b.hairN * 16)
          b.moved = false
        }
        p.setBindGroup(1, this.hairBG[i])
        p.draw(22, b.hairN)
      })
    }
    p.end()
  }

  // offscreen targets (MSAA + depth + resolve), cached by size
  target(w, h, resolve) {
    const k = `${w}x${h}`
    if (this._t?.k !== k) {
      this._t?.msaaT.destroy(); this._t?.depthT.destroy()
      const msaaT = G.dev.createTexture({ size: [w, h], format: this.colorFormat, sampleCount: 4, usage: TU.RENDER_ATTACHMENT })
      const depthT = G.dev.createTexture({ size: [w, h], format: 'depth32float', sampleCount: 4, usage: TU.RENDER_ATTACHMENT })
      this._t = { k, msaaT, depthT, msaa: msaaT.createView(), depth: depthT.createView() }
    }
    return { ...this._t, resolve }
  }

  // render offscreen and read back as RGBA pixels
  async snapshot(w, h) {
    const tex = G.dev.createTexture({ size: [w, h], format: this.colorFormat, usage: TU.RENDER_ATTACHMENT | TU.COPY_SRC })
    const t = { msaa: null, depth: null }
    const msaaT = G.dev.createTexture({ size: [w, h], format: this.colorFormat, sampleCount: 4, usage: TU.RENDER_ATTACHMENT })
    const depthT = G.dev.createTexture({ size: [w, h], format: 'depth32float', sampleCount: 4, usage: TU.RENDER_ATTACHMENT })
    t.msaa = msaaT.createView(); t.depth = depthT.createView(); t.resolve = tex.createView({ format: this.colorFormat })
    const enc = this.eng.enc()
    this.eng.shade()
    this.encode(enc, t, w / h)
    const bpr = Math.ceil(w * 4 / 256) * 256
    const rb = G.dev.createBuffer({ size: bpr * h, usage: BU.COPY_DST | BU.MAP_READ })
    enc.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: bpr }, [w, h])
    this.eng.submit()
    await rb.mapAsync(GPUMapMode.READ)
    const src = new Uint8Array(rb.getMappedRange()), px = new Uint8ClampedArray(w * h * 4)
    const bgra = this.colorFormat.startsWith('bgra')
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * bpr + x * 4, o = (y * w + x) * 4
      px[o] = src[i + (bgra ? 2 : 0)]; px[o + 1] = src[i + 1]; px[o + 2] = src[i + (bgra ? 0 : 2)]; px[o + 3] = 255
    }
    rb.destroy(); tex.destroy(); msaaT.destroy(); depthT.destroy()
    return new ImageData(px, w, h)
  }
  // the (cropped) canvas straight from above at `scale` pixels per texel
  async exportImage(scale = 1) {
    const c = this.eng.canvas, m = c.mask, o = c.origin
    const w = Math.round((m[2] - m[0]) * scale), h = Math.round((m[3] - m[1]) * scale)
    if (w <= 0 || h <= 0 || w > 8192 || h > 8192) return null
    const x0 = o[0] + m[0] / PPM, y0 = o[1] + m[1] / PPM, x1 = o[0] + m[2] / PPM, y1 = o[1] + m[3] / PPM
    const eye = [(x0 + x1) / 2, (y0 + y1) / 2, 3000]
    this.exportView = { vp: m4.mul(m4.ortho((x0 - x1) / 2, (x1 - x0) / 2, (y0 - y1) / 2, (y1 - y0) / 2, 100, 4000), m4.lookAt(eye, [eye[0], eye[1], 0], [0, 1, 0])), eye }
    try { return await this.snapshot(w, h) } finally { this.exportView = null }
  }
}
