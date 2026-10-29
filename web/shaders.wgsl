// Paint sim shaders: simulation kernels + scene rendering (WGSL port of the Metal version).
// Sections start with "//// name dep dep..."; each module is compiled as: common + deps + section.
// Paint amounts are in "units" of 10 µm thickness. Surfaces are 10 texels/mm.
// Paint state lives in a pool of 64x64 tiles (storage buffer) found through a page table; 23 u32 planes per texel:
// 0..7 top wet layer (16 halves), 8..15 under (set) layer, 16..17 aux (du, dt, hdry, graphite), 18..21 dry reflectance, 22 height (f32).

//// common
const PI = 3.14159265;
const FP = 1024.0;          // fixed point scale for brush reservoir atomics
const UNIT_MM = 0.01;
const TEXEL_MM = 0.1;
const TT = 4096u;           // texels per tile
const NPL = 23u;
const P_U = 8u;
const P_AUX = 16u;
const P_DRY = 18u;
const P_H = 22u;

// 16-channel paint vector (pigments + medium)
struct PV { a: vec4f, b: vec4f, c: vec4f, d: vec4f }
fn pvz() -> PV { return PV(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0)); }
fn padd(x: PV, y: PV) -> PV { return PV(x.a + y.a, x.b + y.b, x.c + y.c, x.d + y.d); }
fn psub(x: PV, y: PV) -> PV { return PV(x.a - y.a, x.b - y.b, x.c - y.c, x.d - y.d); }
fn pmul(x: PV, s: f32) -> PV { return PV(x.a * s, x.b * s, x.c * s, x.d * s); }
fn psum(x: PV) -> f32 { let s = x.a + x.b + x.c + x.d; return s.x + s.y + s.z + s.w; }
fn pmax0(x: PV) -> PV { return PV(max(x.a, vec4f(0.0)), max(x.b, vec4f(0.0)), max(x.c, vec4f(0.0)), max(x.d, vec4f(0.0))); }

struct R8 { a: vec4f, b: vec4f }    // 8-band spectrum

fn ss(a: f32, b: f32, x: f32) -> f32 { let t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
fn mob(d: f32) -> f32 { let m = clamp(1.0 - d, 0.0, 1.0); return m * m; }

// reflectance of a layer (absorption KX, scattering SX, already multiplied by thickness) over background Rg
fn km(KX: vec4f, SX: vec4f, Rg: vec4f) -> vec4f {
    let S = max(SX, vec4f(1e-4));
    let a = 1.0 + KX / S;
    let b = sqrt(max(a * a - 1.0, vec4f(1e-8)));
    let x = clamp(b * S, vec4f(1e-5), vec4f(30.0));
    let e = exp(-2.0 * x);
    let cth = (1.0 + e) / (1.0 - e);
    let R = (1.0 - Rg * (a - b * cth)) / (a - Rg + b * cth);
    return select(clamp(R, vec4f(0.0), vec4f(1.0)), Rg, (KX + SX) < vec4f(1e-5));
}
// Saunderson correction: internal reflectance -> what leaves the surface diffusely
fn saund(R: vec4f, k2: f32) -> vec4f { return 0.96 * (1.0 - k2) * R / (1.0 - k2 * R); }

// noise
fn h21(p: vec2f) -> f32 { var p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
fn vn(p: vec2f) -> f32 {
    let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + vec2f(1.0, 0.0)), u.x), mix(h21(i + vec2f(0.0, 1.0)), h21(i + vec2f(1.0, 1.0)), u.x), u.y);
}
fn fbm(p0: vec2f) -> f32 { var p = p0; var s = 0.0; var a = 0.5; for (var i = 0; i < 5; i++) { s += a * vn(p); p = p * 2.03 + 17.1; a *= 0.5; } return s; }

fn inMask(x: vec2f, m: vec4f, r: f32, hole: vec4f) -> bool {
    if (hole.z > 0.0 && distance(x, hole.xy) < hole.z) { return false; }
    let c = (m.xy + m.zw) * 0.5; let h = (m.zw - m.xy) * 0.5;
    let q = abs(x - c) - h + r;
    return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r <= 0.0;
}

// rgb (linear) -> rough 8 band spectrum
fn rgb2spec(c: vec3f) -> R8 {
    return R8(c.b * vec4f(1.0, 1.0, 0.7, 0.15) + c.g * vec4f(0.0, 0.0, 0.3, 0.85),
              c.g * vec4f(1.0, 0.35, 0.0, 0.0) + c.r * vec4f(0.0, 0.65, 1.0, 1.0));
}

// wood albedo, p in mm, grain along x
fn woodCol(p: vec2f, seed: f32, light: vec3f, dark: vec3f) -> vec3f {
    let n = fbm(vec2f(p.x * 0.012, p.y * 0.09) + seed);
    let w = p.y * 0.55 + n * 9.0 + 2.0 * fbm(vec2f(p.x * 0.004, p.y * 0.02) + seed * 1.7);
    let ring = fract(w);
    let band = smoothstep(0.0, 0.18, ring) * (1.0 - smoothstep(0.55, 1.0, ring));
    let fine = vn(vec2f(p.x * 0.08, p.y * 4.0) + seed);
    let pores = smoothstep(0.75, 0.95, vn(vec2f(p.x * 0.6, p.y * 9.0) + seed * 3.1));
    var c = mix(dark, light, 0.55 * band + 0.3 * fine + 0.15);
    c *= 0.88 + 0.24 * fbm(p * 0.03 + seed);
    return c * (1.0 - 0.25 * pores);
}

//// pig
// pigments / Kubelka-Munk (8 spectral bands)
struct Pig {
    K: array<vec4f, 32>,      // K[2i], K[2i+1]: absorption for channel i, bands 0-3, 4-7 (per unit thickness)
    S: array<vec4f, 32>,      // scattering
    M: array<vec4f, 6>,       // spectral -> linear sRGB rows (r0 r1 g0 g1 b0 b1)
    dry: array<vec4f, 4>,     // drying rate per channel (per hour)
    bristle0: vec4f, bristle1: vec4f,
}
@group(0) @binding(0) var<uniform> pig: Pig;

fn kmLayer(v: PV, R: R8) -> R8 {
    if (psum(v) < 1e-4) { return R; }
    var w = array<f32, 16>(v.a.x, v.a.y, v.a.z, v.a.w, v.b.x, v.b.y, v.b.z, v.b.w, v.c.x, v.c.y, v.c.z, v.c.w, v.d.x, v.d.y, v.d.z, v.d.w);
    var K0 = vec4f(0.0); var K1 = vec4f(0.0); var S0 = vec4f(0.0); var S1 = vec4f(0.0);
    for (var i = 0; i < 16; i++) {
        let x = w[i];
        if (x <= 0.0) { continue; }
        K0 += x * pig.K[2 * i]; K1 += x * pig.K[2 * i + 1];
        S0 += x * pig.S[2 * i]; S1 += x * pig.S[2 * i + 1];
    }
    return R8(km(K0, S0, R.a), km(K1, S1, R.b));
}
fn spec2rgb(R: R8) -> vec3f {
    return vec3f(dot(pig.M[0], R.a) + dot(pig.M[1], R.b), dot(pig.M[2], R.a) + dot(pig.M[3], R.b), dot(pig.M[4], R.a) + dot(pig.M[5], R.b));
}

//// state
struct SurfU { size: vec2f, tiles: vec2f, toothAmp: f32, kind: f32, pad: vec2f, g0: vec4f, g1: vec4f }
@group(1) @binding(0) var<storage, read_write> pool: array<u32>;
@group(1) @binding(1) var<storage, read> pt: array<i32>;
@group(1) @binding(2) var<uniform> SU: SurfU;
@group(1) @binding(3) var ground: texture_2d<f32>;    // x tooth, y tint (canvas) / yzw wood rgb (palette)

fn slotAt(p: vec2i) -> i32 { return pt[u32((p.y >> 6u) * i32(SU.tiles.x) + (p.x >> 6u))]; }
fn baseOf(slot: i32, p: vec2i) -> u32 { return u32(slot) * NPL * TT + u32(((p.y & 63) << 6u) | (p.x & 63)); }
fn ld4(b: u32, k: u32) -> vec4f { return vec4f(unpack2x16float(pool[b + k * TT]), unpack2x16float(pool[b + (k + 1u) * TT])); }
fn st4(b: u32, k: u32, v: vec4f) { pool[b + k * TT] = pack2x16float(v.xy); pool[b + (k + 1u) * TT] = pack2x16float(v.zw); }
fn ldPV(b: u32, k: u32) -> PV { return PV(ld4(b, k), ld4(b, k + 2u), ld4(b, k + 4u), ld4(b, k + 6u)); }
fn stPV(b: u32, k: u32, v: PV) { st4(b, k, v.a); st4(b, k + 2u, v.b); st4(b, k + 4u, v.c); st4(b, k + 6u, v.d); }
fn ldH(b: u32) -> f32 { return bitcast<f32>(pool[b + P_H * TT]); }
fn stH(b: u32, h: f32) { pool[b + P_H * TT] = bitcast<u32>(h); }
fn toothAt(p: vec2i) -> f32 { return textureLoad(ground, p, 0).x; }
fn groundSpec(p: vec2i) -> R8 {
    let g = textureLoad(ground, p, 0);
    if (SU.kind > 0.5) { return rgb2spec(g.yzw); }
    return R8(SU.g0 * g.y, SU.g1 * g.y);
}
fn heightAt(p: vec2i) -> f32 {
    let s = slotAt(p);
    if (s < 0) { return toothAt(p) * SU.toothAmp; }
    return ldH(baseOf(s, p));
}

//// kGround
// dense ground: tooth (relief of the weave / gesso / wood) and its colour
struct GroundP { size: vec2f, kind: f32, period: f32, seed: f32, pad: f32, org: vec2f }
@group(0) @binding(0) var<uniform> G: GroundP;
@group(0) @binding(1) var outg: texture_storage_2d<rgba16float, write>;

// one family of threads running along b, spaced 1 apart in a: nearest thread centre, profile and index
fn threads(a: f32, b: f32, seed: f32) -> vec3f {
    var best = -1.0; var bi = 0.0; var bd = 1.0;
    for (var k = -1; k <= 1; k++) {
        let i = floor(a) + f32(k);
        let c = i + 0.5 + 0.34 * (vn(vec2f(i * 3.1 + seed, b * 0.07)) - 0.5) + 0.16 * (h21(vec2f(i, seed)) - 0.5);
        let th = 0.5 + 0.3 * vn(vec2f(i * 1.7 + seed, b * 0.11)) + 0.22 * h21(vec2f(i + seed, 3.0)) + 0.12 * smoothstep(0.8, 0.95, vn(vec2f(i * 5.3, b * 0.3)));
        let d = abs(a - c) / (0.5 * th);
        if (d < 1.0) {
            let pr = sqrt(1.0 - d * d) * th;
            if (pr > best) { best = pr; bi = i; bd = a - c; }
        }
    }
    return vec3f(best, bi, bd);
}
fn weaveH(x: vec2f, P: f32, seed: f32) -> f32 {
    var g = x / P;
    g += 0.6 * vec2f(fbm(vec2f(g.x * 0.02, g.y * 0.003) + seed) - 0.5, fbm(vec2f(g.x * 0.003, g.y * 0.02) + seed + 9.0) - 0.5);
    let wa = threads(g.x, g.y, seed);          // warp (vertical threads)
    let we = threads(g.y, g.x, seed + 17.0);   // weft (horizontal threads)
    // over/under: warp i rises over weft j when i+j is even; undulation follows distance to the crossing thread
    let pw = select(-1.0, 1.0, abs(wa.y + we.y) % 2.0 < 0.5);
    let kv = 0.45 + 1.0 * h21(vec2f(wa.y, we.y) + seed);
    let uw = select(0.0, pw * cos(clamp(we.z, -1.0, 1.0) * PI * 0.5), we.x > 0.0);
    let hw = select(-1.0, 0.42 * wa.x + 0.28 * uw * kv, wa.x > 0.0);
    let uf = select(0.0, -pw * cos(clamp(wa.z, -1.0, 1.0) * PI * 0.5), wa.x > 0.0);
    let hf = select(-1.0, 0.42 * we.x + 0.28 * uf * kv, we.x > 0.0);
    var h = max(max(hw, hf), -0.3);
    let fib = select(vn(vec2f(x.x * 0.25 + x.y * 0.4, x.y * 1.3)), vn(vec2f(x.x * 1.3, x.y * 0.25 + x.x * 0.4)), hw > hf);
    h += 0.05 * (fib - 0.5);
    return clamp((h + 0.3) / 1.05, 0.0, 1.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (any(vec2f(gid.xy) >= G.size)) { return; }
    let x = vec2f(gid.xy) + 0.5;
    var t: f32;
    var o = vec4f(0.0, 1.0, 0.0, 0.0);
    let kind = i32(G.kind + 0.5);
    if (kind == 3) {
        // gessoed panel: smooth, with the faint ridges of the brushed-on gesso
        let q = x * 0.1;
        let strokes = fbm(vec2f(q.x * 0.08, q.y * 1.6) + G.seed) - 0.5;
        let cross = fbm(vec2f(q.x * 1.4, q.y * 0.07) + G.seed * 2.0) - 0.5;
        t = clamp(0.5 + 0.55 * strokes + 0.25 * cross + 0.08 * (vn(x * 0.7) - 0.5), 0.0, 1.0);
    } else if (kind == 0 || kind == 2) {
        // primed cotton or linen canvas: weave softened by gesso, plus gesso brush texture
        var wv = weaveH(x, G.period, G.seed);
        let sm = 0.25 * (weaveH(x + vec2f(0.8, 0.0), G.period, G.seed) + weaveH(x - vec2f(0.8, 0.0), G.period, G.seed) +
                         weaveH(x + vec2f(0.0, 0.8), G.period, G.seed) + weaveH(x - vec2f(0.0, 0.8), G.period, G.seed));
        wv = mix(wv, sm, 0.45);
        // gesso fills the interstices and leaves its own faint brush marks
        let gesso = fbm(x * vec2f(0.004, 0.03) + G.seed) - 0.5;
        let fill = 0.47 + 0.12 * fbm(x * 0.05 + 2.0);
        wv = max(wv, fill);
        let grit = vn(x * 0.9) - 0.5;
        t = clamp((wv - 0.3) / 0.7 + 0.14 * gesso + 0.05 * grit, 0.0, 1.0);
        o.y = 0.985 + 0.03 * fbm(x * 0.01 + 4.0) - 0.02 * (1.0 - wv);
    } else {
        // oiled wooden palette: very fine grain relief
        let mm = x * TEXEL_MM + G.org;
        let c = woodCol(mm.yx, G.seed, vec3f(0.74, 0.55, 0.34), vec3f(0.55, 0.37, 0.20));
        o = vec4f(0.0, c);
        t = 0.5 + 0.35 * (vn(vec2f(mm.y * 0.1, mm.x * 6.0)) - 0.5) + 0.1 * (vn(x * 0.7) - 0.5);
    }
    o.x = t;
    textureStore(outg, gid.xy, o);
}

//// kInitTile state
// fresh tiles: no paint, the ground as the dried layer
@group(0) @binding(0) var<storage, read> tl: array<i32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let t = tl[gid.z];
    let tx = t % i32(SU.tiles.x); let ty = t / i32(SU.tiles.x);
    let p = vec2i(tx * 64 + i32(gid.x), ty * 64 + i32(gid.y));
    let slot = pt[t];
    let b = u32(slot) * NPL * TT + gid.y * 64u + gid.x;
    for (var k = 0u; k < P_DRY; k++) { pool[b + k * TT] = 0u; }
    if (any(vec2f(p) >= SU.size)) { for (var k = P_DRY; k < NPL; k++) { pool[b + k * TT] = 0u; } return; }
    let R = groundSpec(p);
    st4(b, P_DRY, R.a); st4(b, P_DRY + 2u, R.b);
    stH(b, toothAt(p) * SU.toothAmp);
}

//// fp state
// brush footprint raster: one bristle tuft per instance, contact segment a->b (tip at b), in footprint pixel coords
struct BrG { seg: vec4f, p0: vec4f, p1: vec4f }   // p0: radius(texels), press, z (units above substrate peaks, 0 = touching), rate; p1: coat area, id
struct FPP { vp: vec2f, org: vec2f, tsz: vec2f, toothAmp: f32, groove: f32, conform: f32, coat: f32, pad: vec2f }
@group(0) @binding(0) var<storage, read> B: array<BrG>;
@group(0) @binding(1) var<uniform> FP_: FPP;
struct FPV { @builtin(position) pos: vec4f, @location(0) @interpolate(flat) a: vec2f, @location(1) @interpolate(flat) b: vec2f,
             @location(2) @interpolate(flat) p: vec4f, @location(3) @interpolate(flat) id: u32 }

@vertex fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> FPV {
    let g = B[iid];
    let a = g.seg.xy; let b = g.seg.zw;
    let R = g.p0.x * (1.0 + FP_.coat) + 1.5;
    let d = b - a; let L = length(d);
    let u = select(vec2f(1.0, 0.0), d / L, L > 1e-4); let n = vec2f(-u.y, u.x);
    var c = select(a - u * R, b + u * R, (vid & 1u) != 0u);
    c += select(-n, n, (vid & 2u) != 0u) * R;
    var o: FPV;
    o.pos = vec4f(c.x / FP_.vp.x * 2.0 - 1.0, 1.0 - c.y / FP_.vp.y * 2.0, 0.0, 1.0);
    if (g.p0.x <= 0.0) { o.pos = vec4f(-9.0, -9.0, 0.0, 1.0); }
    o.a = a; o.b = b; o.p = g.p0; o.id = u32(g.p1.y + 0.5);
    return o;
}
struct FPO { @location(0) c: vec4f, @builtin(frag_depth) d: f32 }
@fragment fn fs(i: FPV) -> FPO {
    let p = i.pos.xy;
    let ab = i.b - i.a; let l2 = dot(ab, ab);
    let t = select(1.0, clamp(dot(p - i.a, ab) / l2, 0.0, 1.0), l2 > 1e-6);
    let dist = length(p - (i.a + ab * t));
    let r = i.p.x; let R = r * (1.0 + FP_.coat);
    if (dist > R) { discard; }
    let tp = clamp(vec2i(p) + vec2i(FP_.org), vec2i(0), vec2i(FP_.tsz) - 1);
    let n = toothAt(tp);
    let press = i.p.y * mix(1.0, 0.6, t);
    let g01 = smoothstep(r * 0.45, R, dist);
    let valley = FP_.toothAmp * (1.0 - n);
    let zb = select(valley * (1.0 - clamp(press * FP_.conform, 0.0, 1.0)), i.p.z + valley, i.p.z > 0.0);
    let gap = max(zb, 0.0) + g01 * FP_.groove * (1.0 - 0.5 * press);
    var o: FPO;
    o.c = vec4f(max(zb, 0.0), floor(press * 255.0) + min(g01, 0.999), f32(i.id), i.p.w);
    o.d = clamp(gap / 8192.0, 0.0, 1.0);
    return o;
}

//// kTransfer state
// paint transfer brush <-> surface
struct StepP {
    org: vec2f, fpsz: vec2f, surf: vec2f, nb: f32, knife: f32,
    alpha: f32, rho: f32, ratePass: f32, mixPass: f32,
    mask: vec4f,
    maskR: f32, toothAmp: f32, eps: f32, film: f32,
    sinkRep: f32, groove: f32, seed: f32, pad: f32,
    hole: vec4f,
}
@group(0) @binding(0) var fp: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> res: array<vec4f>;
@group(0) @binding(2) var<storage, read> cap: array<f32>;
@group(0) @binding(3) var<storage, read_write> dres: array<atomic<i32>>;
@group(0) @binding(4) var<uniform> P: StepP;

fn ldRes(id: i32) -> PV { let i = u32(id) * 4u; return PV(res[i], res[i + 1u], res[i + 2u], res[i + 3u]); }
fn addRes(id: i32, d: PV) {
    let q = u32(id) * 16u;
    var v = array<f32, 16>(d.a.x, d.a.y, d.a.z, d.a.w, d.b.x, d.b.y, d.b.z, d.b.w, d.c.x, d.c.y, d.c.z, d.c.w, d.d.x, d.d.y, d.d.z, d.d.w);
    for (var k = 0u; k < 16u; k++) { if (v[k] != 0.0) { atomicAdd(&dres[q + k], i32(round(v[k] * FP))); } }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    var id = -1;
    var dB = pvz();
    if (all(vec2f(gid.xy) < P.fpsz)) {
        let f = textureLoad(fp, gid.xy, 0);
        let pi = vec2i(gid.xy) + vec2i(P.org);
        if (f.z >= 0.0 && all(pi >= vec2i(0)) && all(vec2f(pi) < P.surf) && inMask(vec2f(pi) + 0.5, P.mask, P.maskR, P.hole)) {
            let slot = slotAt(pi);
            if (slot >= 0) {
                let bid = i32(f.z + 0.5);
                let zb = f.x; let press = floor(f.y) / 255.0; let g01 = fract(f.y); let s = f.w;
                let gap = zb + g01 * P.groove * (1.0 - 0.5 * press);
                let area = cap[bid] * (1.0 / 60.0);
                let b = baseOf(slot, pi);
                let T = ldPV(b, 0u); let U = ldPV(b, P_U);
                var ax = ld4(b, P_AUX);
                let hT = psum(T); let hU = psum(U);
                let mT = mob(ax.y); let mU = mob(ax.x);
                let hm = mT * hT + mU * hU;
                // some bristles carry more paint than others (varies per stroke): dry brush breaks up along bristle tracks
                let aff = 0.35 + 1.3 * h21(vec2f(f32(bid) * 1.37, P.seed));
                let Bc = pmul(ldRes(bid), 1.0 / area);
                let hb = psum(Bc); let hbc = hb * aff;
                let reach = hT + hU + hbc - gap;
                let c = smoothstep(-P.eps, P.eps, reach);
                let tn = toothAt(pi);
                if (P.knife > 0.5 && c > 0.002) {
                    // palette knife: lifts wet (and tacky) paint clean off, leaving only a faint stain
                    let rate = 1.0 - pow(1.0 - 0.9, s);
                    let mU2 = max(mU, 0.7);
                    let k = c * rate;
                    let kT = pmul(T, mT * k); let kU = pmul(U, mU2 * k);
                    let T2 = pmax0(psub(T, kT)); let U2 = pmax0(psub(U, kU));
                    if (psum(U2) < 1e-4) { ax.x = 0.0; }
                    stPV(b, 0u, T2); stPV(b, P_U, U2); st4(b, P_AUX, ax);
                    stH(b, tn * P.toothAmp + ax.z + psum(U2) + psum(T2));
                    dB = padd(kT, kU); id = bid;
                } else if (c > 0.002) {
                    let rate = 1.0 - pow(1.0 - P.ratePass, s);
                    let mixr = 1.0 - pow(1.0 - P.mixPass, s);
                    // bristles ride on a film of paint; between clumps the film is thicker (furrows and ridges)
                    // paint mostly fills the weave valleys rather than following them
                    let valley = P.toothAmp * (1.0 - tn);
                    let film = P.rho * (1.0 - 0.55 * press) * (hm + hb) * mix(0.45, 1.2, g01);
                    let geff = max(zb, valley * (1.0 - 0.35 * press)) + max(g01 * P.groove, film);
                    let tgt = P.alpha * min(hm + hb, geff);
                    let d = c * rate * (tgt - hm);
                    var dT = pvz(); var dU = pvz();
                    let hmT = mT * hT; let hmU = mU * hU;
                    if (d > 0.0 && hb > 1e-5) {
                        // fresh paint lands on top; the old top film sinks under it
                        let dep = pmul(Bc, min(d, hb) / hb);
                        let dv = psum(dep);
                        let sink = min(hT, dv * P.sinkRep + max(0.0, hT + dv - P.film));
                        if (sink > 0.0 && hT > 1e-5) {
                            let sk = pmul(T, sink / hT);
                            dT = pmul(sk, -1.0); dU = sk;
                            ax.x = ax.x * hU / (hU + sink) + ax.y * sink / (hU + sink);
                        }
                        dT = padd(dT, dep); dB = pmul(dep, -1.0);
                    } else if (d < 0.0 && hm > 1e-5) {
                        // lift from the top film first, then from what lies beneath
                        let pk = min(-d, hm);
                        let fT = min(pk, hmT); let fU = pk - fT;
                        var kT = pvz(); var kU = pvz();
                        if (hT > 1e-5) { kT = pmul(T, fT / hT); }
                        if (hU > 1e-5) { kU = pmul(U, fU / hU); }
                        dT = pmul(kT, -1.0); dU = pmul(kU, -1.0); dB = padd(kT, kU);
                    }
                    let e = c * mixr * min(hb, hm);
                    if (e > 1e-6 && hm > 1e-5 && hb > 1e-5) {
                        let eT = min(e, hmT); let eU = e - eT;
                        let xB = pmul(Bc, e / hb);
                        var xT = pvz(); var xU = pvz();
                        if (hT > 1e-5) { xT = pmul(T, eT / hT); }
                        if (hU > 1e-5) { xU = pmul(U, eU / hU); }
                        dT = psub(padd(dT, xB), xT); dU = psub(dU, xU); dB = psub(padd(padd(dB, xT), xU), xB);
                    }
                    let T2 = pmax0(padd(T, dT)); let U2 = pmax0(padd(U, dU));
                    let hT2 = psum(T2);
                    let added = max(psum(dT), 0.0);
                    ax.y = select(0.0, ax.y * clamp((hT2 - added) / hT2, 0.0, 1.0), hT2 > 1e-4);
                    if (psum(U2) < 1e-4) { ax.x = 0.0; }
                    stPV(b, 0u, T2); stPV(b, P_U, U2); st4(b, P_AUX, ax);
                    stH(b, tn * P.toothAmp + ax.z + psum(U2) + hT2);
                    id = bid;
                }
            }
        }
    }
    // reservoir deltas: one atomic set per subgroup when lanes share a bristle
//#if SUBGROUPS
    let first = subgroupBroadcastFirst(id);
    if (subgroupAll(id == first)) {
        if (first >= 0) {
            let sa = subgroupAdd(dB.a); let sb = subgroupAdd(dB.b); let sc = subgroupAdd(dB.c); let sd = subgroupAdd(dB.d);
            if (subgroupElect()) { addRes(first, PV(sa, sb, sc, sd)); }
        }
    } else if (id >= 0) { addRes(id, dB); }
//#else
    if (id >= 0) { addRes(id, dB); }
//#endif
}

//// kApply
// apply reservoir deltas; each tuft trades paint with the reservoir of its zone of the brush (index nb + zone).
// Zones keep streaks of unmixed colour alive inside the brush. One workgroup.
struct ApplyP { nb: f32, feed: f32, wick: f32, clean: f32, nz: f32, pad0: f32, pad1: f32, pad2: f32 }
@group(0) @binding(0) var<storage, read_write> res: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dres: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> cap: array<f32>;
@group(0) @binding(3) var<uniform> A: ApplyP;
@group(0) @binding(4) var<storage, read> zoneOf: array<i32>;
@group(0) @binding(5) var<storage, read_write> flow: array<vec4f>;

fn ldRes(id: i32) -> PV { let i = u32(id) * 4u; return PV(res[i], res[i + 1u], res[i + 2u], res[i + 3u]); }
fn stRes(id: i32, r: PV) { let i = u32(id) * 4u; res[i] = r.a; res[i + 1u] = r.b; res[i + 2u] = r.c; res[i + 3u] = r.d; }

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32) {
    let nb = i32(A.nb);
    for (var j = i32(tid); j < nb; j += 256) {
        let z = nb + zoneOf[j];
        let belly = pmul(ldRes(z), 1.0 - A.clean);
        let vb = psum(belly); let cb = cap[z];
        let q = u32(j) * 16u;
        var v: array<f32, 16>;
        for (var k = 0u; k < 16u; k++) { v[k] = f32(atomicExchange(&dres[q + k], 0)) / FP; }
        var r = pmul(ldRes(j), 1.0 - A.clean);
        r = pmax0(padd(r, PV(vec4f(v[0], v[1], v[2], v[3]), vec4f(v[4], v[5], v[6], v[7]), vec4f(v[8], v[9], v[10], v[11]), vec4f(v[12], v[13], v[14], v[15]))));
        let vol = psum(r); let cj = cap[j];
        var f = A.feed * (vb / cb - vol / cj) * cj;
        f = min(f, vb * cj / cb);
        var d = pvz();
        if (f > 0.0 && vb > 1e-6) { d = pmul(belly, f / vb); }
        else if (f < 0.0 && vol > 1e-6) { d = pmul(r, f / vol); }
        let sw = A.wick * min(vol, vb * cj / cb);
        if (sw > 0.0 && vol > 1e-6 && vb > 1e-6) { d = psub(padd(d, pmul(belly, sw / vb)), pmul(r, sw / vol)); }
        r = pmax0(padd(r, d));
        stRes(j, r);
        let i = u32(j) * 4u;
        flow[i] = d.a; flow[i + 1u] = d.b; flow[i + 2u] = d.c; flow[i + 3u] = d.d;
    }
    storageBarrier();
    for (var z = i32(tid); z < i32(A.nz); z += 256) {
        var t = pvz();
        for (var j = 0; j < nb; j++) {
            if (zoneOf[j] == z) { let i = u32(j) * 4u; t = padd(t, PV(flow[i], flow[i + 1u], flow[i + 2u], flow[i + 3u])); }
        }
        stRes(nb + z, pmax0(psub(pmul(ldRes(nb + z), 1.0 - A.clean), t)));
    }
}

//// kSqueeze state
// tube squeeze: deposit a bead of paint
struct SqzP { org: vec2f, sz: vec2f, surf: vec2f, c: vec2f, comp0: vec4f, comp1: vec4f, comp2: vec4f, comp3: vec4f,
              rad: f32, amt: f32, hmax: f32, toothAmp: f32, mask: vec4f, maskR: f32, seed: f32, dir: vec2f, hole: vec4f }
@group(0) @binding(0) var<uniform> Q: SqzP;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (any(vec2f(gid.xy) >= Q.sz)) { return; }
    let pi = vec2i(gid.xy) + vec2i(Q.org);
    if (any(pi < vec2i(0)) || any(vec2f(pi) >= Q.surf)) { return; }
    let x = vec2f(pi) + 0.5;
    if (!inMask(x, Q.mask, Q.maskR, Q.hole)) { return; }
    let rel = (x - Q.c) / Q.rad;
    let d2 = dot(rel, rel);
    if (d2 > 2.5) { return; }
    let slot = slotAt(pi);
    if (slot < 0) { return; }
    let b = baseOf(slot, pi);
    let h = ldH(b);
    var w = exp(-2.4 * d2);
    // nozzle ridges along the extrusion direction
    let across = dot(rel, vec2f(-Q.dir.y, Q.dir.x));
    w *= 1.0 + 0.06 * sin(across * 17.0 + Q.seed) + 0.05 * (vn(x * 0.35 + Q.seed) - 0.5);
    w *= select(select(1.0, 0.4, h > 0.8 * Q.hmax), 0.08, h > Q.hmax);
    let a = Q.amt * w;
    if (a <= 0.0) { return; }
    var T = ldPV(b, 0u);
    let hT = psum(T);
    var ax = ld4(b, P_AUX);
    T = padd(T, pmul(PV(Q.comp0, Q.comp1, Q.comp2, Q.comp3), a));
    ax.y = ax.y * hT / (hT + a);
    stPV(b, 0u, T);
    st4(b, P_AUX, ax);
    stH(b, h + a);
}

//// kPencil state
// pencil: graphite on the ground (aux.w), under any paint
struct PenP { org: vec2f, sz: vec2f, surf: vec2f, a: vec2f, b: vec2f, r: f32, press: f32, toothAmp: f32, share: f32, pad: vec2f, mask: vec4f, maskR: f32, seed: f32, pad2: vec2f, hole: vec4f }
@group(0) @binding(0) var<uniform> Pp: PenP;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (any(vec2f(gid.xy) >= Pp.sz)) { return; }
    let pi = vec2i(gid.xy) + vec2i(Pp.org);
    if (any(vec2f(pi) >= Pp.surf) || any(pi < vec2i(0))) { return; }
    let x = vec2f(pi) + 0.5;
    if (!inMask(x, Pp.mask, Pp.maskR, Pp.hole)) { return; }
    let ab = Pp.b - Pp.a;
    let t = clamp(dot(x - Pp.a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    let d = length(x - (Pp.a + ab * t));
    if (d > Pp.r) { return; }
    let slot = slotAt(pi);
    if (slot < 0) { return; }
    let b = baseOf(slot, pi);
    // a light touch only catches the tops of the weave; pressing harder reaches into the valleys
    let depth = Pp.toothAmp * (1.0 - toothAt(pi));
    let reach = 2.0 + Pp.press * 14.0;
    let c = ss(reach + 1.2, reach - 1.2, depth + 1.5 * (h21(x * 0.71 + 3.1) - 0.5));
    let edge = 1.0 - smoothstep(Pp.r * 0.45, Pp.r, d);
    let grain = 0.65 + 0.35 * h21(x * 1.37 + floor(Pp.seed * 3.0) * 17.0);
    var k = min(0.6 + 0.6 * Pp.press, 1.0) * c * edge * grain;
    let T = ldPV(b, 0u); let U = ldPV(b, P_U);
    var ax = ld4(b, P_AUX);
    k *= exp(-(psum(T) + psum(U) + ax.z) / 3.0);     // the sketch lies under all paint, wet or dry
    ax.w = 1.0 - (1.0 - ax.w) * pow(1.0 - k, Pp.share);
    st4(b, P_AUX, ax);
}

//// kDry pig state
struct DryP { hours: f32, thick0: f32, shrink: f32, apply: f32 }
@group(0) @binding(1) var<uniform> D: DryP;
@group(0) @binding(2) var<storage, read> tl: array<i32>;
@group(0) @binding(3) var prev: texture_storage_2d<rgba8unorm, write>;
fn dryRate(v: PV, h: f32) -> f32 {
    if (h < 1e-4) { return 0.0; }
    let r = v.a * pig.dry[0] + v.b * pig.dry[1] + v.c * pig.dry[2] + v.d * pig.dry[3];
    return (r.x + r.y + r.z + r.w) / h;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let t = tl[gid.z];
    let p = vec2i((t % i32(SU.tiles.x)) * 64 + i32(gid.x), (t / i32(SU.tiles.x)) * 64 + i32(gid.y));
    if (any(vec2f(p) >= SU.size)) { return; }
    let b = baseOf(pt[t], p);
    var T = ldPV(b, 0u); var U = ldPV(b, P_U);
    var ax = ld4(b, P_AUX);
    let hT = psum(T); let hU = psum(U); let hn = hT + hU;
    if (hn < 1e-4) { if (D.apply < 0.5) { textureStore(prev, p, vec4f(0.0)); } return; }
    let rT = dryRate(T, hT) / (1.0 + hT / D.thick0);
    let rU = dryRate(U, hU) / (1.0 + (hU + 2.0 * hT) / D.thick0);   // covered paint dries slower
    let dt = clamp(ax.y + rT * D.hours, 0.0, 1.0); let du = clamp(ax.x + rU * D.hours, 0.0, 1.0);
    let dn = (dt * hT + du * hU) / hn;
    if (D.apply < 0.5) { textureStore(prev, p, vec4f(dn, clamp(hn / 20.0, 0.0, 1.0), 0.0, 1.0)); return; }
    // each layer sets on its own; they never mix (light strokes over dark ones stay on top)
    ax.x = du; ax.y = dt;
    if (dn >= 0.995) {
        var R = R8(ld4(b, P_DRY), ld4(b, P_DRY + 2u));
        // graphite stays under the paint: fold it into the ground before the dried layers go on top
        R = R8(mix(R.a, vec4f(0.075), ax.w * 0.92), mix(R.b, vec4f(0.075), ax.w * 0.92)); ax.w = 0.0;
        R = kmLayer(U, R);
        R = kmLayer(T, R);
        st4(b, P_DRY, R.a); st4(b, P_DRY + 2u, R.b);
        ax.z += hn * D.shrink;
        U = pvz(); T = pvz(); ax.x = 0.0; ax.y = 0.0;
    }
    stPV(b, 0u, T); stPV(b, P_U, U);
    st4(b, P_AUX, ax);
    stH(b, toothAt(p) * SU.toothAmp + ax.z + psum(U) + psum(T));
}

//// kShade pig state
// display cache (color + normal/height), per dirty rect. Colour is stored sqrt-encoded in rgba8.
struct ShadeP { org: vec2f, sz: vec2f, relief: f32, k2: f32, maskR: f32, glossPaint: f32, mask: vec4f, glossGround: f32, dryGloss: f32, pad: vec2f }
@group(0) @binding(1) var<uniform> Sh: ShadeP;
@group(0) @binding(2) var col: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var nrm: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (any(vec2f(gid.xy) >= Sh.sz)) { return; }
    let pi = vec2i(gid.xy) + vec2i(Sh.org);
    if (any(vec2f(pi) >= SU.size)) { return; }
    let slot = slotAt(pi);
    var T = pvz(); var U = pvz(); var ax = vec4f(0.0); var R: R8; var h: f32;
    if (slot < 0) { R = groundSpec(pi); h = toothAt(pi) * SU.toothAmp; }
    else {
        let b = baseOf(slot, pi);
        T = ldPV(b, 0u); U = ldPV(b, P_U); ax = ld4(b, P_AUX);
        R = R8(ld4(b, P_DRY), ld4(b, P_DRY + 2u)); h = ldH(b);
    }
    R = R8(mix(R.a, vec4f(0.075), ax.w * 0.92), mix(R.b, vec4f(0.075), ax.w * 0.92));     // graphite
    R = kmLayer(U, R);
    R = kmLayer(T, R);
    let hT = psum(T); let hU = psum(U);
    let hp = hT + hU + ax.z;
    let k2 = Sh.k2 * smoothstep(0.0, 3.0, hp);   // bare ground has a weaker interface effect
    let rgb = max(spec2rgb(R8(saund(R.a, k2), saund(R.b, k2))), vec3f(0.0));
    let m = vec2i(SU.size) - 1;
    let hl = heightAt(clamp(pi - vec2i(1, 0), vec2i(0), m)); let hr = heightAt(clamp(pi + vec2i(1, 0), vec2i(0), m));
    let hd = heightAt(clamp(pi - vec2i(0, 1), vec2i(0), m)); let hu = heightAt(clamp(pi + vec2i(0, 1), vec2i(0), m));
    let sc = 0.5 * UNIT_MM / TEXEL_MM * Sh.relief;
    let wet = clamp((hT * mob(ax.y) + hU * mob(ax.x)) / 4.0, 0.0, 1.0);
    let paint = smoothstep(0.5, 6.0, hp);
    let gloss = mix(Sh.glossGround + 0.3 * ax.w, Sh.glossPaint * mix(Sh.dryGloss, 1.0, wet), paint);
    textureStore(col, pi, vec4f(sqrt(min(rgb, vec3f(1.0))), gloss));
    textureStore(nrm, pi, vec4f((hr - hl) * sc, (hu - hd) * sc, h * UNIT_MM, wet));
}

//// kDown
// mip chain of the display caches, one level, within a region
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<FORMAT, write>;
@group(0) @binding(2) var<uniform> Rg: vec4f;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let p = vec2i(gid.xy) + vec2i(Rg.xy);
    let ds = vec2i(textureDimensions(dst));
    if (any(p >= vec2i(Rg.zw)) || any(p >= ds)) { return; }
    let m = vec2i(textureDimensions(src)) - 1;
    let q = p * 2;
    let a = textureLoad(src, min(q, m), 0) + textureLoad(src, min(q + vec2i(1, 0), m), 0) + textureLoad(src, min(q + vec2i(0, 1), m), 0) + textureLoad(src, min(q + 1, m), 0);
    textureStore(dst, p, a * 0.25);
}

//// kBrushColor pig
// per tuft display color for bristle rendering
@group(0) @binding(1) var<storage, read> res: array<vec4f>;
@group(0) @binding(2) var<storage, read> cap: array<f32>;
@group(0) @binding(3) var<storage, read_write> outc: array<vec4f>;
@group(0) @binding(4) var<uniform> NB: vec4f;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i32(i) > i32(NB.x)) { return; }
    let r = PV(res[i * 4u], res[i * 4u + 1u], res[i * 4u + 2u], res[i * 4u + 3u]);
    let v = psum(r);
    let fill = v / cap[i];
    var R = R8(vec4f(0.8), vec4f(0.8));
    // render a thick coat of the carried mixture
    if (v > 1e-5) { R = kmLayer(pmul(r, 30.0 / v), R); }
    let c = max(spec2rgb(R8(saund(R.a, 0.45), saund(R.b, 0.45))), vec3f(0.0));
    outc[i] = vec4f(c, clamp(fill * 6.0, 0.0, 1.0));
}

//// rcommon
// ================================================================= rendering
struct Frame {
    vp: mat4x4f,
    lvp: mat4x4f,
    cam: vec4f,        // xyz camera position
    ldir: vec4f,       // xyz to light
    lcol: vec4f,
    amb: vec4f,
    params: vec4f,     // x heat on, y time, z hide handle, w debug
}
@group(0) @binding(0) var<uniform> F: Frame;
struct Vtx { p: vec4f, n: vec4f }       // p.w = u, n.w = v
struct ObjDraw { model: mat4x4f, base: vec4f, mat: vec4f, extra: vec4f }   // mat: metal, rough, mode, fade-near

//// rtex rcommon
@group(0) @binding(1) var shadowTex: texture_depth_2d;
@group(0) @binding(2) var shadowHeld: texture_depth_2d;
@group(0) @binding(3) var sCmp: sampler_comparison;
@group(0) @binding(4) var sLin: sampler;
@group(0) @binding(5) var labels: texture_2d<f32>;

// soft shadows that harden near contact: the key light is a big window, so the penumbra grows with occluder distance.
// 8 blocker-search taps + 8 filtered taps on a golden-angle spiral, rotated per pixel. Radii are in 1/4096ths of the map
// (the size the look was tuned at), so softness in mm doesn't depend on the map resolution.
fn spiral(i: i32) -> vec2f { let a = f32(i) * 2.39996; return vec2f(cos(a), sin(a)) * sqrt((f32(i) + 0.5) / 8.0); }
fn shadowAt(wp: vec3f, held: bool, bias: f32) -> f32 {
    let l = F.lvp * vec4f(wp, 1.0);
    let q = l.xyz / l.w;
    let uv = vec2f(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5);
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return 1.0; }
    let dim = vec2f(textureDimensions(shadowTex));
    let zr = q.z - bias;
    let a = h21(wp.xy * 7.3) * 6.2832;
    let rot = mat2x2f(cos(a), sin(a), -sin(a), cos(a));
    var sum = 0.0; var cnt = 0.0;
    for (var i = 0; i < 8; i++) {
        let c = vec2i(clamp((uv + rot * spiral(i) * (48.0 / 4096.0)) * dim, vec2f(0.0), dim - 1.0));
        var d: f32;
        if (held) { d = textureLoad(shadowHeld, c, 0); } else { d = textureLoad(shadowTex, c, 0); }
        if (d < zr) { sum += d; cnt += 1.0; }
    }
    if (cnt == 0.0) { return 1.0; }
    let distMM = (zr - sum / cnt) * 2200.0;
    let rad = clamp(distMM * 0.45, 1.5, 70.0) / 4096.0;
    var s = 0.0;
    for (var i = 0; i < 8; i++) {
        let u = uv + rot * spiral(i) * rad;
        if (held) { s += textureSampleCompareLevel(shadowHeld, sCmp, u, zr); } else { s += textureSampleCompareLevel(shadowTex, sCmp, u, zr); }
    }
    s /= 8.0;
    // light bounced around the room fills in shadows of things far above the surface
    return mix(s, 1.0, clamp(distMM / 220.0, 0.0, 1.0) * 0.5);
}

// studio environment: warm room, large window toward the key light
fn envCol(r: vec3f, rough: f32) -> vec3f {
    let L = F.ldir.xyz;
    let base = mix(vec3f(0.10, 0.085, 0.07), vec3f(0.55, 0.52, 0.48), clamp(r.z * 0.5 + 0.5, 0.0, 1.0));
    let k = mix(300.0, 6.0, rough);
    let win = exp(k * (dot(r, L) - 1.0));
    return base * 0.6 + F.lcol.rgb * win * mix(0.9, 0.25, rough);
}

fn lit(N: vec3f, V: vec3f, alb: vec3f, metal: f32, rough: f32, sh: f32, ao: f32) -> vec3f {
    let L = F.ldir.xyz;
    let Hh = normalize(L + V);
    let nl = clamp(dot(N, L), 0.0, 1.0); let nv = clamp(dot(N, V), 0.0, 1.0) + 1e-3; let nh = clamp(dot(N, Hh), 0.0, 1.0);
    let a = max(rough * rough, 0.002); let a2 = a * a;
    let D = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    let k = a * 0.5;
    let G = (nl / (nl * (1.0 - k) + k)) * (nv / (nv * (1.0 - k) + k));
    let f0 = mix(vec3f(0.04), alb, metal);
    let Fr = f0 + (1.0 - f0) * pow(1.0 - clamp(dot(Hh, V), 0.0, 1.0), 5.0);
    let spec = D * G * Fr / (4.0 * nv * max(nl, 1e-3));
    let diff = alb * (1.0 - metal);
    var c = (diff / PI * 3.0 + spec) * F.lcol.rgb * nl * sh;
    let skyAmb = F.amb.rgb * (0.65 + 0.35 * N.z);
    c += diff * skyAmb * ao;
    // nv goes up to 1.001: pow of a negative base is undefined (NaN, i.e. black pixels, on drivers that do exp2(y * log2(x)))
    let Fe = f0 + (1.0 - f0) * pow(max(1.0 - nv, 0.0), 5.0) * (1.0 - rough);
    c += envCol(reflect(-V, N), rough) * Fe * ao * mix(1.0, sh, 0.5);
    return c;
}

//// surf rtex rcommon
// paint surfaces (canvas top / palette top)
struct SurfDraw { orgSize: vec4f, mask: vec4f, misc: vec4f, hole: vec4f }  // orgSize: world origin xy (mm), size wh (texels); misc: z, maskR, kind
@group(1) @binding(0) var<uniform> SD: SurfDraw;
@group(1) @binding(1) var colT: texture_2d<f32>;
@group(1) @binding(2) var nrmT: texture_2d<f32>;
@group(1) @binding(3) var prevT: texture_2d<f32>;
struct SV { @builtin(position) pos: vec4f, @location(0) wp: vec3f, @location(1) tx: vec2f }
@vertex fn vs(@builtin(vertex_index) vid: u32) -> SV {
    let c = vec2f(select(SD.mask.x, SD.mask.z, (vid & 1u) != 0u), select(SD.mask.y, SD.mask.w, (vid & 2u) != 0u));   // texel coords
    let wp = vec3f(SD.orgSize.xy + c * TEXEL_MM, SD.misc.x);
    var o: SV; o.pos = F.vp * vec4f(wp, 1.0); o.wp = wp; o.tx = c; return o;
}
@fragment fn fs(i: SV) -> @location(0) vec4f {
    let uv = i.tx / SD.orgSize.zw;
    var c = textureSample(colT, sLin, uv);
    c = vec4f(c.rgb * c.rgb, c.a);
    let n = textureSample(nrmT, sLin, uv);
    let hb = textureSampleBias(nrmT, sLin, uv, 4.0).z;
    let pv = textureSample(prevT, sLin, uv);
    if (!inMask(i.tx, SD.mask, SD.misc.y, SD.hole)) { discard; }
    let N = normalize(vec3f(-n.x, -n.y, 1.0));
    let wp = i.wp + vec3f(0.0, 0.0, n.z);
    let V = normalize(F.cam.xyz - wp);
    let ao = clamp(1.0 - (hb - n.z) * 1.6, 0.0, 1.0);
    let sh = shadowAt(wp, false, 0.0015) * shadowAt(wp, true, 0.0015);
    let rough = mix(0.85, 0.16, c.a);
    var alb = c.rgb;
    if (F.params.w > 0.5) { alb = vec3f(0.5); }
    var rgb = lit(N, V, alb, 0.0, rough, sh, ao);
    if (F.params.x > 0.5) {
        // drying preview: wet = blue, tacky = yellow, dry = red; bare canvas dimmed
        let d = pv.x;
        var hc = mix(vec3f(0.10, 0.35, 1.0), vec3f(1.0, 0.85, 0.15), smoothstep(0.1, 0.55, d));
        hc = mix(hc, vec3f(0.95, 0.15, 0.08), smoothstep(0.55, 0.95, d));
        let lum = dot(rgb, vec3f(0.3, 0.5, 0.2));
        rgb = mix(vec3f(lum * 0.5), hc * (0.55 + 0.6 * lum), smoothstep(0.0, 0.15, pv.y));
    }
    return vec4f(rgb, 1.0);
}

//// obj rtex rcommon
// 3D objects
@group(1) @binding(0) var<storage, read> VB: array<Vtx>;
@group(2) @binding(0) var<uniform> OD: ObjDraw;
struct OV { @builtin(position) pos: vec4f, @location(0) wp: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f, @location(3) lp: vec3f }
@vertex fn vs(@builtin(vertex_index) vid: u32) -> OV {
    let v = VB[vid];
    let wp = OD.model * vec4f(v.p.xyz, 1.0);
    var o: OV; o.pos = F.vp * wp; o.wp = wp.xyz; o.n = normalize((OD.model * vec4f(v.n.xyz, 0.0)).xyz); o.uv = vec2f(v.p.w, v.n.w); o.lp = v.p.xyz;
    return o;
}
struct OF { @location(0) c: vec4f, @builtin(sample_mask) mask: u32 }
@fragment fn fs(i: OV) -> OF {
    var N = normalize(i.n);
    let V = normalize(F.cam.xyz - i.wp);
    if (dot(N, V) < 0.0 && OD.mat.z > 9.5) { N = -N; }
    var alb = OD.base.rgb;
    var metal = OD.mat.x; var rough = OD.mat.y;
    let mode = i32(OD.mat.z + 0.5) % 10;
    let uv = i.uv;
    // label atlas lookup (sampled in uniform control flow)
    let la = vec2f((uv.y - OD.extra.x) / max(OD.extra.y - OD.extra.x, 1e-4), fract(uv.x + 0.25));
    let lab = textureSample(labels, sLin, vec2f(la.x, OD.extra.z + la.y * OD.extra.w)).rgb;
    if (mode == 1) {            // tube: label band / aluminium
        let lu = vec2f(uv.y, uv.x);  // u around, v along
        if (uv.y > OD.extra.x && uv.y < OD.extra.y) {
            alb = lab;
            metal = 0.0; rough = 0.42;
        } else {
            let cr = vn(vec2f(lu.x * 30.0, lu.y * 8.0)) - 0.5;
            N = normalize(N + 0.08 * cr * vec3f(1.0, 1.0, 0.3));
            alb = vec3f(0.86, 0.86, 0.85); metal = 1.0; rough = 0.28 + 0.1 * cr;
        }
    } else if (mode == 2) {     // wood (palette edge, tray, natural handles)
        alb = woodCol(i.lp.xy * OD.extra.x + OD.extra.yz, OD.extra.w, OD.base.rgb, OD.base.rgb * 0.55);
    } else if (mode == 3) {     // ribbed plastic cap
        let rib = sin(uv.x * 2.0 * PI * 24.0);
        let t = normalize(cross(N, vec3f(0.0, 0.0, 1.0)) + 1e-4);
        N = normalize(N + t * rib * 0.25);
    } else if (mode == 4) {     // canvas edge fabric
        let q = uv * vec2f(OD.extra.x, OD.extra.y);
        let w = 0.5 + 0.5 * sin(q.x * 2.0 * PI) * sin(q.y * 2.0 * PI);
        alb *= 0.9 + 0.12 * w;
    } else if (mode == 5) {     // table
        alb = woodCol(i.wp.xy * 0.6, 3.0, vec3f(0.30, 0.18, 0.10), vec3f(0.13, 0.075, 0.04));
    }
    var sh = shadowAt(i.wp + N * 0.4, false, 0.004);
    if (OD.mat.w <= 0.0) { sh *= shadowAt(i.wp + N * 0.4, true, 0.004); }
    let c = lit(N, V, alb, metal, rough, sh, 1.0);
    var o: OF; o.c = vec4f(c, 1.0); o.mask = 0xFu;
    if (OD.mat.w > 0.0) {      // fade when near the camera (brush handle), dithered over MSAA samples
        let dz = F.cam.z - i.wp.z;
        var a = smoothstep(OD.mat.w * 0.35, OD.mat.w, dz);
        if (F.params.z > 0.5) { a = 0.0; }
        let n = i32(clamp(floor(a * 4.0 + h21(i.pos.xy)), 0.0, 4.0));
        o.mask = (1u << u32(n)) - 1u;
        if (n == 0) { discard; }
    }
    return o;
}

//// shadow rcommon
@group(1) @binding(0) var<storage, read> VB: array<Vtx>;
@group(2) @binding(0) var<uniform> OD: ObjDraw;
@vertex fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    return F.lvp * (OD.model * vec4f(VB[vid].p.xyz, 1.0));
}

//// hair rcommon
// bristle hairs (camera facing ribbons along cubic curves)
struct Hair { p0: vec4f, p1: vec4f, p2: vec4f, p3: vec4f }   // p0.w = tuft id, p1.w = width (mm), p2.w = paint start t, p3.w = shade jitter
@group(1) @binding(0) var<storage, read> HS: array<Hair>;
@group(1) @binding(1) var<storage, read> TC: array<vec4f>;
struct HV { @builtin(position) pos: vec4f, @location(0) wp: vec3f, @location(1) t: vec3f, @location(2) s: f32,
            @location(3) @interpolate(flat) paint: f32, @location(4) @interpolate(flat) pc: vec4f, @location(5) @interpolate(flat) jit: f32 }
const HSEG = 10;
const HAIR_COL = vec3f(0.82, 0.70, 0.50);
@vertex fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> HV {
    let h = HS[iid];
    let seg = i32(vid / 2u); let side = select(-1.0, 1.0, (vid & 1u) != 0u);
    let t = f32(seg) / f32(HSEG);
    let u = 1.0 - t;
    var P = u * u * u * h.p0.xyz + 3.0 * u * u * t * h.p1.xyz + 3.0 * u * t * t * h.p2.xyz + t * t * t * h.p3.xyz;
    let T = normalize(3.0 * u * u * (h.p1.xyz - h.p0.xyz) + 6.0 * u * t * (h.p2.xyz - h.p1.xyz) + 3.0 * t * t * (h.p3.xyz - h.p2.xyz) + 1e-5);
    let V = normalize(F.cam.xyz - P);
    let Bv = normalize(cross(T, V) + 1e-5);
    let w = h.p1.w * mix(1.0, 0.45, t * t);
    P += Bv * side * w * 0.5;
    var o: HV; o.pos = F.vp * vec4f(P, 1.0); o.wp = P; o.t = T; o.s = t;
    o.paint = h.p2.w; o.pc = TC[i32(h.p0.w)]; o.jit = h.p3.w;
    return o;
}
@fragment fn fs(i: HV) -> @location(0) vec4f {
    let T = normalize(i.t);
    let L = F.ldir.xyz; let V = normalize(F.cam.xyz - i.wp);
    var alb = HAIR_COL * (0.85 + 0.3 * i.jit);
    // paint coats the lower part of the bristles
    let pa = i.pc.a * smoothstep(i.paint - 0.08, i.paint + 0.05, i.s);
    alb = mix(alb, i.pc.rgb, pa);
    let shine = mix(1.0, 1.6, pa);
    let tl = dot(T, L); let tv = dot(T, V);
    let sl = sqrt(max(1.0 - tl * tl, 0.0)); let sv = sqrt(max(1.0 - tv * tv, 0.0));
    let spec = pow(clamp(sl * sv - tl * tv, 0.0, 1.0), 40.0) * 0.35 * shine;
    let c = alb * (sl * F.lcol.rgb * 0.75 + F.amb.rgb) + spec * F.lcol.rgb;
    return vec4f(c, 1.0);
}
