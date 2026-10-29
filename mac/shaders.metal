// Paint sim shaders: simulation kernels + scene rendering.
// Paint amounts are in "units" of 10 µm thickness. Surfaces are 10 texels/mm.
#include <metal_stdlib>
using namespace metal;

constant float PI = 3.14159265;
constant float FP = 1024.0;          // fixed point scale for brush reservoir atomics
constant float UNIT_MM = 0.01;
constant float TEXEL_MM = 0.1;

// ---------------------------------------------------------------- 16-channel paint vector (pigments + medium)
struct PV { float4 a, b, c, d; };
inline PV pv(float4 a, float4 b, float4 c, float4 d) { PV p; p.a = a; p.b = b; p.c = c; p.d = d; return p; }
inline PV pvz() { return pv(0, 0, 0, 0); }
inline PV operator+(PV x, PV y) { return pv(x.a + y.a, x.b + y.b, x.c + y.c, x.d + y.d); }
inline PV operator-(PV x, PV y) { return pv(x.a - y.a, x.b - y.b, x.c - y.c, x.d - y.d); }
inline PV operator-(PV x) { return pv(-x.a, -x.b, -x.c, -x.d); }
inline PV operator*(PV x, float s) { return pv(x.a * s, x.b * s, x.c * s, x.d * s); }
inline float psum(PV x) { float4 s = x.a + x.b + x.c + x.d; return s.x + s.y + s.z + s.w; }
inline PV pmax0(PV x) { return pv(max(x.a, 0.0), max(x.b, 0.0), max(x.c, 0.0), max(x.d, 0.0)); }

// state texture slices: 0..3 top wet layer, 4..7 under (set) layer, 8 aux (du, dt, hdry, -), 9..10 dry reflectance
constant uint SL_U = 4, SL_AUX = 8, SL_DRY = 9;
inline PV ldS(texture2d_array<half, access::read_write> S, uint2 p, uint s) {
    return pv(float4(S.read(p, s)), float4(S.read(p, s + 1)), float4(S.read(p, s + 2)), float4(S.read(p, s + 3)));
}
inline PV ldR(texture2d_array<half, access::read> S, uint2 p, uint s) {
    return pv(float4(S.read(p, s)), float4(S.read(p, s + 1)), float4(S.read(p, s + 2)), float4(S.read(p, s + 3)));
}
inline void stS(texture2d_array<half, access::read_write> S, uint2 p, uint s, PV v) {
    S.write(half4(v.a), p, s); S.write(half4(v.b), p, s + 1); S.write(half4(v.c), p, s + 2); S.write(half4(v.d), p, s + 3);
}

// ---------------------------------------------------------------- pigments / Kubelka-Munk (8 spectral bands)
struct Pig {
    float4 K[32];       // K[2i], K[2i+1]: absorption for channel i, bands 0-3, 4-7 (per unit thickness)
    float4 S[32];       // scattering
    float4 M[6];        // spectral -> linear sRGB rows (r0 r1 g0 g1 b0 b1)
    float4 dry[4];      // drying rate per channel (per hour)
    float4 bristle0, bristle1;  // bristle reflectance spectrum
};

inline void ksOf(PV v, constant Pig& g, thread float4& K0, thread float4& K1, thread float4& S0, thread float4& S1) {
    float w[16] = { v.a.x, v.a.y, v.a.z, v.a.w, v.b.x, v.b.y, v.b.z, v.b.w, v.c.x, v.c.y, v.c.z, v.c.w, v.d.x, v.d.y, v.d.z, v.d.w };
    K0 = 0; K1 = 0; S0 = 0; S1 = 0;
    for (int i = 0; i < 16; i++) {
        float x = w[i];
        if (x <= 0) continue;
        K0 += x * g.K[2 * i]; K1 += x * g.K[2 * i + 1];
        S0 += x * g.S[2 * i]; S1 += x * g.S[2 * i + 1];
    }
}

// reflectance of a layer (absorption KX, scattering SX, already multiplied by thickness) over background Rg
inline float4 km(float4 KX, float4 SX, float4 Rg) {
    float4 S = max(SX, 1e-4);
    float4 a = 1.0 + KX / S;
    float4 b = sqrt(max(a * a - 1.0, 1e-8));
    float4 x = clamp(b * S, 1e-5, 30.0);
    float4 e = exp(-2.0 * x);
    float4 cth = (1.0 + e) / (1.0 - e);
    float4 R = (1.0 - Rg * (a - b * cth)) / (a - Rg + b * cth);
    return select(saturate(R), Rg, (KX + SX) < 1e-5);
}
inline void kmLayer(PV v, constant Pig& g, thread float4& R0, thread float4& R1) {
    if (psum(v) < 1e-4) return;
    float4 K0, K1, S0, S1; ksOf(v, g, K0, K1, S0, S1);
    R0 = km(K0, S0, R0); R1 = km(K1, S1, R1);
}
inline float3 spec2rgb(float4 R0, float4 R1, constant Pig& g) {
    return float3(dot(g.M[0], R0) + dot(g.M[1], R1), dot(g.M[2], R0) + dot(g.M[3], R1), dot(g.M[4], R0) + dot(g.M[5], R1));
}
// Saunderson correction: internal reflectance -> what leaves the surface diffusely (k1 handled by specular)
inline float4 saund(float4 R, float k2) { return 0.96 * (1.0 - k2) * R / (1.0 - k2 * R); }
inline float mob(float d) { float m = saturate(1.0 - d); return m * m; }

// ---------------------------------------------------------------- noise
inline float h21(float2 p) { float3 p3 = fract(float3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
inline float vn(float2 p) {
    float2 i = floor(p), f = fract(p); float2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(h21(i), h21(i + float2(1, 0)), u.x), mix(h21(i + float2(0, 1)), h21(i + float2(1, 1)), u.x), u.y);
}
inline float fbm(float2 p) { float s = 0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * vn(p); p = p * 2.03 + 17.1; a *= 0.5; } return s; }

inline bool inMask(float2 x, float4 m, float r, float4 hole) {
    if (hole.z > 0 && distance(x, hole.xy) < hole.z) return false;
    float2 c = (m.xy + m.zw) * 0.5, h = (m.zw - m.xy) * 0.5;
    float2 q = abs(x - c) - h + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r <= 0.0;
}

// rgb (linear) -> rough 8 band spectrum
inline void rgb2spec(float3 c, thread float4& R0, thread float4& R1) {
    R0 = c.b * float4(1, 1, 0.7, 0.15) + c.g * float4(0, 0, 0.3, 0.85) + c.r * float4(0, 0, 0, 0);
    R1 = c.g * float4(1, 0.35, 0, 0) + c.r * float4(0, 0.65, 1, 1);
}

// wood albedo, p in mm, grain along x
inline float3 woodCol(float2 p, float seed, float3 light, float3 dark) {
    float n = fbm(float2(p.x * 0.012, p.y * 0.09) + seed);
    float w = p.y * 0.55 + n * 9.0 + 2.0 * fbm(float2(p.x * 0.004, p.y * 0.02) + seed * 1.7);
    float ring = fract(w);
    float band = smoothstep(0.0, 0.18, ring) * (1.0 - smoothstep(0.55, 1.0, ring));
    float fine = vn(float2(p.x * 0.08, p.y * 4.0) + seed);
    float pores = smoothstep(0.75, 0.95, vn(float2(p.x * 0.6, p.y * 9.0) + seed * 3.1));
    float3 c = mix(dark, light, 0.55 * band + 0.3 * fine + 0.15);
    c *= 0.88 + 0.24 * fbm(p * 0.03 + seed);
    return c * (1.0 - 0.25 * pores);
}

// ---------------------------------------------------------------- surface init
struct InitP { int2 size; int kind; float period; float4 g0; float4 g1; float toothAmp; float seed; float2 org; };

// one family of threads running along b, spaced 1 apart in a: nearest thread centre, profile and index
inline float3 threads(float a, float b, float seed) {
    float best = -1.0, bi = 0.0, bd = 1.0;
    for (int k = -1; k <= 1; k++) {
        float i = floor(a) + float(k);
        float c = i + 0.5 + 0.34 * (vn(float2(i * 3.1 + seed, b * 0.07)) - 0.5) + 0.16 * (h21(float2(i, seed)) - 0.5);
        float th = 0.5 + 0.3 * vn(float2(i * 1.7 + seed, b * 0.11)) + 0.22 * h21(float2(i + seed, 3.0)) + 0.12 * smoothstep(0.8, 0.95, vn(float2(i * 5.3, b * 0.3)));
        float d = abs(a - c) / (0.5 * th);
        if (d < 1.0) {
            float pr = sqrt(1.0 - d * d) * th;
            if (pr > best) { best = pr; bi = i; bd = a - c; }
        }
    }
    return float3(best, bi, bd);
}
inline float weaveH(float2 x, float P, float seed) {
    float2 g = x / P;
    g += 0.6 * float2(fbm(float2(g.x * 0.02, g.y * 0.003) + seed) - 0.5, fbm(float2(g.x * 0.003, g.y * 0.02) + seed + 9.0) - 0.5);
    float3 wa = threads(g.x, g.y, seed);          // warp (vertical threads)
    float3 we = threads(g.y, g.x, seed + 17.0);   // weft (horizontal threads)
    // over/under: warp i rises over weft j when i+j is even; undulation follows distance to the crossing thread
    float pw = fmod(abs(wa.y + we.y), 2.0) < 0.5 ? 1.0 : -1.0;
    float kv = 0.45 + 1.0 * h21(float2(wa.y, we.y) + seed);
    float uw = we.x > 0 ? pw * cos(clamp(we.z, -1.0, 1.0) * PI * 0.5) : 0.0;
    float hw = wa.x > 0 ? 0.42 * wa.x + 0.28 * uw * kv : -1.0;
    float uf = wa.x > 0 ? -pw * cos(clamp(wa.z, -1.0, 1.0) * PI * 0.5) : 0.0;
    float hf = we.x > 0 ? 0.42 * we.x + 0.28 * uf * kv : -1.0;
    float h = max(max(hw, hf), -0.3);
    float fib = hw > hf ? vn(float2(x.x * 1.3, x.y * 0.25 + x.x * 0.4)) : vn(float2(x.x * 0.25 + x.y * 0.4, x.y * 1.3));
    h += 0.05 * (fib - 0.5);
    return saturate((h + 0.3) / 1.05);
}

kernel void kInit(texture2d_array<half, access::write> S [[texture(0)]],
                  texture2d<half, access::write> tooth [[texture(1)]],
                  texture2d<float, access::write> H [[texture(2)]],
                  constant InitP& P [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= uint2(P.size))) return;
    float2 x = float2(gid) + 0.5;
    float t;
    float4 R0 = P.g0, R1 = P.g1;
    if (P.kind == 3) {
        // gessoed panel: smooth, with the faint ridges of the brushed-on gesso
        float2 q = x * 0.1;
        float strokes = fbm(float2(q.x * 0.08, q.y * 1.6) + P.seed) - 0.5;
        float cross = fbm(float2(q.x * 1.4, q.y * 0.07) + P.seed * 2.0) - 0.5;
        t = saturate(0.5 + 0.55 * strokes + 0.25 * cross + 0.08 * (vn(x * 0.7) - 0.5));
    } else if (P.kind == 0 || P.kind == 2) {
        // primed cotton or linen canvas: weave softened by gesso, plus gesso brush texture
        float wv = weaveH(x, P.period, P.seed);
        float sm = 0.25 * (weaveH(x + float2(0.8, 0), P.period, P.seed) + weaveH(x - float2(0.8, 0), P.period, P.seed) +
                           weaveH(x + float2(0, 0.8), P.period, P.seed) + weaveH(x - float2(0, 0.8), P.period, P.seed));
        wv = mix(wv, sm, 0.45);
        // gesso fills the interstices and leaves its own faint brush marks
        float gesso = fbm(x * float2(0.004, 0.03) + P.seed) - 0.5;
        float fill = 0.47 + 0.12 * fbm(x * 0.05 + 2.0);
        wv = max(wv, fill) + 0.25 * max(wv - fill, 0.0) * 0.0;
        float grit = vn(x * 0.9) - 0.5;
        t = saturate((wv - 0.3) / 0.7 + 0.14 * gesso + 0.05 * grit);
        float tint = 0.985 + 0.03 * fbm(x * 0.01 + 4.0) - 0.02 * (1.0 - wv);
        R0 *= tint; R1 *= tint;
    } else {
        // oiled wooden palette: very fine grain relief
        float2 mm = x * TEXEL_MM + P.org;
        float3 c = woodCol(mm.yx, P.seed, float3(0.74, 0.55, 0.34), float3(0.55, 0.37, 0.20));
        rgb2spec(c, R0, R1);
        t = 0.5 + 0.35 * (vn(float2(mm.y * 0.1, mm.x * 6.0)) - 0.5) + 0.1 * (vn(x * 0.7) - 0.5);
    }
    for (uint s = 0; s < SL_DRY; s++) S.write(half4(0), gid, s);
    S.write(half4(R0), gid, SL_DRY);
    S.write(half4(R1), gid, SL_DRY + 1);
    tooth.write(half4(t), gid);
    H.write(float4(t * P.toothAmp), gid);
}

// ---------------------------------------------------------------- brush footprint raster
// One bristle tuft per instance: contact segment a->b (tip at b), in footprint pixel coords.
struct BrG { float4 seg; float4 p0; float4 p1; };   // p0: radius(texels), press, z (units above substrate peaks, 0 = touching), rate exponent; p1: coat area, -, -, -
struct FPP { float2 vp; int2 org; int2 tsz; float toothAmp; float groove; float conform; float coat; float2 pad; };
struct FPV { float4 pos [[position]]; float2 a [[flat]]; float2 b [[flat]]; float4 p [[flat]]; uint id [[flat]]; };

vertex FPV fpVert(uint vid [[vertex_id]], uint iid [[instance_id]], device const BrG* B [[buffer(0)]], constant FPP& P [[buffer(1)]]) {
    BrG g = B[iid];
    float2 a = g.seg.xy, b = g.seg.zw;
    float R = g.p0.x * (1.0 + P.coat) + 1.5;
    float2 d = b - a; float L = length(d);
    float2 u = L > 1e-4 ? d / L : float2(1, 0), n = float2(-u.y, u.x);
    float2 c = (vid & 1) ? b + u * R : a - u * R;
    c += ((vid & 2) ? n : -n) * R;
    FPV o;
    o.pos = float4(c.x / P.vp.x * 2.0 - 1.0, 1.0 - c.y / P.vp.y * 2.0, 0, 1);
    if (g.p0.x <= 0) o.pos = float4(-9, -9, 0, 1);
    o.a = a; o.b = b; o.p = g.p0; o.id = uint(g.p1.y + 0.5);
    return o;
}
struct FPO { float4 c [[color(0)]]; float d [[depth(any)]]; };
fragment FPO fpFrag(FPV in [[stage_in]], texture2d<half, access::read> tooth [[texture(0)]], constant FPP& P [[buffer(1)]]) {
    float2 p = in.pos.xy;
    float2 ab = in.b - in.a; float l2 = dot(ab, ab);
    float t = l2 > 1e-6 ? saturate(dot(p - in.a, ab) / l2) : 1.0;
    float dist = length(p - (in.a + ab * t));
    float r = in.p.x, R = r * (1.0 + P.coat);
    if (dist > R) discard_fragment();
    int2 tp = clamp(int2(p) + P.org, int2(0), P.tsz - 1);
    float n = float(tooth.read(uint2(tp)).r);
    float press = in.p.y * mix(1.0, 0.6, t);
    float g01 = smoothstep(r * 0.45, R, dist);
    float valley = P.toothAmp * (1.0 - n);
    float zb = in.p.z > 0 ? in.p.z + valley : valley * (1.0 - saturate(press * P.conform));
    float gap = max(zb, 0.0) + g01 * P.groove * (1.0 - 0.5 * press);
    FPO o;
    o.c = float4(max(zb, 0.0), floor(press * 255.0) + min(g01, 0.999), float(in.id), in.p.w);
    o.d = saturate(gap / 8192.0);
    return o;
}

// ---------------------------------------------------------------- paint transfer brush <-> surface
struct StepP {
    int2 org; int2 fpsz; int2 surf; int nb; int pad0;
    float alpha; float rho; float ratePass; float mixPass;
    float4 mask;
    float maskR; float toothAmp; float eps; float film;
    float sinkRep; float groove; float seed; float pad1;
    float4 hole;
};

inline PV ldRes(device const float4* r, int id) { return pv(r[id * 4], r[id * 4 + 1], r[id * 4 + 2], r[id * 4 + 3]); }

kernel void kTransfer(texture2d<float, access::read> fp [[texture(0)]],
                      texture2d_array<half, access::read_write> S [[texture(1)]],
                      texture2d<float, access::read_write> H [[texture(2)]],
                      texture2d<half, access::read> tooth [[texture(3)]],
                      device const float4* res [[buffer(0)]],
                      device const float* cap [[buffer(1)]],
                      device atomic_int* dres [[buffer(2)]],
                      constant StepP& P [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    int id = -1;
    PV dB = pvz();
    if (all(gid < uint2(P.fpsz))) {
        float4 f = fp.read(gid);
        int2 pi = int2(gid) + P.org;
        if (f.z >= 0 && all(pi >= 0) && all(pi < P.surf) && inMask(float2(pi) + 0.5, P.mask, P.maskR, P.hole)) {
            int bid = int(f.z + 0.5);
            float zb = f.x, press = floor(f.y) / 255.0, g01 = fract(f.y), s = f.w;
            float gap = zb + g01 * P.groove * (1.0 - 0.5 * press);
            float area = cap[bid] * (1.0 / 60.0);
            uint2 p = uint2(pi);
            PV T = ldS(S, p, 0), U = ldS(S, p, SL_U);
            float4 ax = float4(S.read(p, SL_AUX));
            float hT = psum(T), hU = psum(U);
            float mT = mob(ax.y), mU = mob(ax.x);
            float hm = mT * hT + mU * hU;
            // some bristles carry more paint than others (varies per stroke): dry brush breaks up along bristle tracks
            float aff = 0.35 + 1.3 * h21(float2(float(bid) * 1.37, P.seed));
            PV Bc = ldRes(res, bid) * (1.0 / area);
            float hb = psum(Bc), hbc = hb * aff;
            float reach = hT + hU + hbc - gap;
            float c = smoothstep(-P.eps, P.eps, reach);
            if (P.pad1 > 0.5 && c > 0.002) {
                // palette knife: lifts wet (and tacky) paint clean off, leaving only a faint stain
                float rate = 1.0 - pow(1.0 - 0.9, s);
                float mU2 = max(mU, 0.7);
                float k = c * rate;
                PV kT = T * (mT * k), kU = U * (mU2 * k);
                PV T2 = pmax0(T - kT), U2 = pmax0(U - kU);
                if (psum(U2) < 1e-4) ax.x = 0;
                stS(S, p, 0, T2); stS(S, p, SL_U, U2);
                S.write(half4(ax), p, SL_AUX);
                H.write(float4(float(tooth.read(p).r) * P.toothAmp + ax.z + psum(U2) + psum(T2)), p);
                dB = kT + kU; id = bid;
            } else if (c > 0.002) {
                float rate = 1.0 - pow(1.0 - P.ratePass, s);
                float mixr = 1.0 - pow(1.0 - P.mixPass, s);
                // bristles ride on a film of paint; between clumps the film is thicker (furrows and ridges)
                // paint mostly fills the weave valleys rather than following them
                float valley = P.toothAmp * (1.0 - float(tooth.read(p).r));
                float film = P.rho * (1.0 - 0.55 * press) * (hm + hb) * mix(0.45, 1.2, g01);
                float geff = max(zb, valley * (1.0 - 0.35 * press)) + max(g01 * P.groove, film);
                float target = P.alpha * min(hm + hb, geff);
                float d = c * rate * (target - hm);
                PV dT = pvz(), dU = pvz();
                float hmT = mT * hT, hmU = mU * hU;
                if (d > 0 && hb > 1e-5) {
                    // fresh paint lands on top; the old top film sinks under it
                    PV dep = Bc * (min(d, hb) / hb);
                    float dv = psum(dep);
                    float sink = min(hT, dv * P.sinkRep + max(0.0, hT + dv - P.film));
                    if (sink > 0 && hT > 1e-5) {
                        PV sk = T * (sink / hT);
                        dT = -sk; dU = sk;
                        ax.x = ax.x * hU / (hU + sink) + ax.y * sink / (hU + sink);
                    }
                    dT = dT + dep; dB = -dep;
                } else if (d < 0 && hm > 1e-5) {
                    // lift from the top film first, then from what lies beneath
                    float pk = min(-d, hm);
                    float fT = min(pk, hmT), fU = pk - fT;
                    PV kT = hT > 1e-5 ? T * (fT / hT) : pvz(), kU = hU > 1e-5 ? U * (fU / hU) : pvz();
                    dT = -kT; dU = -kU; dB = kT + kU;
                }
                float e = c * mixr * min(hb, hm);
                if (e > 1e-6 && hm > 1e-5 && hb > 1e-5) {
                    float eT = min(e, hmT), eU = e - eT;
                    PV xB = Bc * (e / hb), xT = hT > 1e-5 ? T * (eT / hT) : pvz(), xU = hU > 1e-5 ? U * (eU / hU) : pvz();
                    dT = dT + xB - xT; dU = dU - xU; dB = dB + xT + xU - xB;
                }
                PV T2 = pmax0(T + dT), U2 = pmax0(U + dU);
                float hT2 = psum(T2);
                float added = max(psum(dT), 0.0);
                ax.y = hT2 > 1e-4 ? ax.y * saturate((hT2 - added) / hT2) : 0.0;
                if (psum(U2) < 1e-4) ax.x = 0;
                stS(S, p, 0, T2); stS(S, p, SL_U, U2);
                S.write(half4(ax), p, SL_AUX);
                float tn = float(tooth.read(p).r);
                H.write(float4(tn * P.toothAmp + ax.z + psum(U2) + hT2), p);
                id = bid;
            }
        }
    }
    // reservoir deltas: one atomic set per simdgroup when lanes share a bristle
    int first = simd_broadcast_first(id);
    if (simd_all(id == first)) {
        if (first >= 0) {
            float4 sa = simd_sum(dB.a), sb = simd_sum(dB.b), sc = simd_sum(dB.c), sd = simd_sum(dB.d);
            if (simd_is_first()) {
                device atomic_int* q = dres + first * 16;
                float v[16] = { sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w, sc.x, sc.y, sc.z, sc.w, sd.x, sd.y, sd.z, sd.w };
                for (int k = 0; k < 16; k++) if (v[k] != 0) atomic_fetch_add_explicit(q + k, int(rint(v[k] * FP)), memory_order_relaxed);
            }
        }
    } else if (id >= 0) {
        device atomic_int* q = dres + id * 16;
        float v[16] = { dB.a.x, dB.a.y, dB.a.z, dB.a.w, dB.b.x, dB.b.y, dB.b.z, dB.b.w, dB.c.x, dB.c.y, dB.c.z, dB.c.w, dB.d.x, dB.d.y, dB.d.z, dB.d.w };
        for (int k = 0; k < 16; k++) if (v[k] != 0) atomic_fetch_add_explicit(q + k, int(rint(v[k] * FP)), memory_order_relaxed);
    }
}

// apply reservoir deltas; each tuft trades paint with the reservoir of its zone of the brush (index nb + zone).
// Zones keep streaks of unmixed colour alive inside the brush. One threadgroup of 1024.
struct ApplyP { int nb; float feed; float wick; float clean; int nz; int pad0; int pad1; int pad2; };
kernel void kApply(device float4* res [[buffer(0)]], device atomic_int* dres [[buffer(1)]],
                   device const float* cap [[buffer(2)]], constant ApplyP& A [[buffer(3)]],
                   device const int* zoneOf [[buffer(4)]], device float4* flow [[buffer(5)]],
                   uint tid [[thread_index_in_threadgroup]]) {
    int nb = A.nb;
    for (int j = int(tid); j < nb; j += 1024) {
        int z = nb + zoneOf[j];
        PV belly = ldRes(res, z) * (1.0 - A.clean);
        float vb = psum(belly), cb = cap[z];
        device atomic_int* q = dres + j * 16;
        float v[16];
        for (int k = 0; k < 16; k++) v[k] = float(atomic_exchange_explicit(q + k, 0, memory_order_relaxed)) / FP;
        PV r = ldRes(res, j) * (1.0 - A.clean);
        r = pmax0(r + pv(float4(v[0], v[1], v[2], v[3]), float4(v[4], v[5], v[6], v[7]), float4(v[8], v[9], v[10], v[11]), float4(v[12], v[13], v[14], v[15])));
        float vol = psum(r), cj = cap[j];
        float f = A.feed * (vb / cb - vol / cj) * cj;
        f = min(f, vb * cj / cb);
        PV d = pvz();
        if (f > 0 && vb > 1e-6) d = belly * (f / vb);
        else if (f < 0 && vol > 1e-6) d = r * (f / vol);
        float sw = A.wick * min(vol, vb * cj / cb);
        if (sw > 0 && vol > 1e-6 && vb > 1e-6) d = d + belly * (sw / vb) - r * (sw / vol);
        r = pmax0(r + d);
        res[j * 4] = r.a; res[j * 4 + 1] = r.b; res[j * 4 + 2] = r.c; res[j * 4 + 3] = r.d;
        flow[j * 4] = d.a; flow[j * 4 + 1] = d.b; flow[j * 4 + 2] = d.c; flow[j * 4 + 3] = d.d;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (int z = int(tid); z < A.nz; z += 1024) {
        PV t = pvz();
        for (int j = 0; j < nb; j++) if (zoneOf[j] == z) t = t + pv(flow[j * 4], flow[j * 4 + 1], flow[j * 4 + 2], flow[j * 4 + 3]);
        PV belly = pmax0(ldRes(res, nb + z) * (1.0 - A.clean) - t);
        int i = nb + z;
        res[i * 4] = belly.a; res[i * 4 + 1] = belly.b; res[i * 4 + 2] = belly.c; res[i * 4 + 3] = belly.d;
    }
}

// ---------------------------------------------------------------- tube squeeze: deposit a bead of paint
struct SqzP { int2 org; int2 sz; int2 surf; float2 c; float4 comp0; float4 comp1; float4 comp2; float4 comp3; float rad; float amt; float hmax; float toothAmp; float4 mask; float maskR; float seed; float2 dir; float4 hole; };
kernel void kSqueeze(texture2d_array<half, access::read_write> S [[texture(0)]],
                     texture2d<float, access::read_write> H [[texture(1)]],
                     texture2d<half, access::read> tooth [[texture(2)]],
                     constant SqzP& Q [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= uint2(Q.sz))) return;
    int2 pi = int2(gid) + Q.org;
    if (any(pi < 0) || any(pi >= Q.surf)) return;
    float2 x = float2(pi) + 0.5;
    if (!inMask(x, Q.mask, Q.maskR, Q.hole)) return;
    float2 rel = (x - Q.c) / Q.rad;
    float d2 = dot(rel, rel);
    if (d2 > 2.5) return;
    uint2 p = uint2(pi);
    float h = H.read(p).x;
    float w = exp(-2.4 * d2);
    // nozzle ridges along the extrusion direction
    float across = dot(rel, float2(-Q.dir.y, Q.dir.x));
    w *= 1.0 + 0.06 * sin(across * 17.0 + Q.seed) + 0.05 * (vn(x * 0.35 + Q.seed) - 0.5);
    w *= h > Q.hmax ? 0.08 : (h > 0.8 * Q.hmax ? 0.4 : 1.0);
    float a = Q.amt * w;
    if (a <= 0) return;
    PV T = ldS(S, p, 0);
    float hT = psum(T);
    float4 ax = float4(S.read(p, SL_AUX));
    T = T + pv(Q.comp0, Q.comp1, Q.comp2, Q.comp3) * a;
    ax.y = ax.y * hT / (hT + a);
    stS(S, p, 0, T);
    S.write(half4(ax), p, SL_AUX);
    H.write(float4(h + a), p);
}

// ---------------------------------------------------------------- pencil: graphite on the ground (aux.w), under any paint
struct PenP { int2 org; int2 sz; int2 surf; float2 a; float2 b; float r; float press; float toothAmp; float share; float4 mask; float maskR; float seed; float pad; float4 hole; };
kernel void kPencil(texture2d_array<half, access::read_write> S [[texture(0)]],
                    texture2d<half, access::read> tooth [[texture(1)]],
                    constant PenP& P [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= uint2(P.sz))) return;
    int2 pi = int2(gid) + P.org;
    if (any(pi >= P.surf)) return;
    float2 x = float2(pi) + 0.5;
    if (!inMask(x, P.mask, P.maskR, P.hole)) return;
    float2 ab = P.b - P.a;
    float t = saturate(dot(x - P.a, ab) / max(dot(ab, ab), 1e-6));
    float d = length(x - (P.a + ab * t));
    if (d > P.r) return;
    uint2 p = uint2(pi);
    // a light touch only catches the tops of the weave; pressing harder reaches into the valleys
    float depth = P.toothAmp * (1.0 - float(tooth.read(p).r));
    float reach = 2.0 + P.press * 14.0;
    float c = smoothstep(reach + 1.2, reach - 1.2, depth + 1.5 * (h21(x * 0.71 + 3.1) - 0.5));
    float edge = 1.0 - smoothstep(P.r * 0.45, P.r, d);
    float grain = 0.65 + 0.35 * h21(x * 1.37 + floor(P.seed * 3.0) * 17.0);
    float k = min(0.6 + 0.6 * P.press, 1.0) * c * edge * grain;
    PV T = ldS(S, p, 0), U = ldS(S, p, SL_U);
    float4 ax = float4(S.read(p, SL_AUX));
    k *= exp(-(psum(T) + psum(U) + ax.z) / 3.0);     // the sketch lies under all paint, wet or dry
    ax.w = 1.0 - (1.0 - ax.w) * pow(1.0 - k, P.share);
    S.write(half4(ax), p, SL_AUX);
}

// ---------------------------------------------------------------- drying
struct DryP { int2 size; float hours; float thick0; float shrink; float apply; float2 pad; };
inline float dryRate(PV v, float h, constant Pig& g) {
    if (h < 1e-4) return 0;
    float4 r = v.a * g.dry[0] + v.b * g.dry[1] + v.c * g.dry[2] + v.d * g.dry[3];
    return (r.x + r.y + r.z + r.w) / h;
}
kernel void kDry(texture2d_array<half, access::read_write> S [[texture(0)]],
                 texture2d<float, access::read_write> H [[texture(1)]],
                 texture2d<half, access::read> tooth [[texture(2)]],
                 texture2d<half, access::write> prev [[texture(3)]],
                 constant Pig& g [[buffer(0)]], constant DryP& D [[buffer(1)]],
                 constant float& toothAmp [[buffer(2)]],
                 uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= uint2(D.size))) return;
    uint2 p = gid;
    PV T = ldS(S, p, 0), U = ldS(S, p, SL_U);
    float4 ax = float4(S.read(p, SL_AUX));
    float hT = psum(T), hU = psum(U), hn = hT + hU;
    if (hn < 1e-4) { if (D.apply < 0.5) prev.write(half4(0), p); return; }
    float rT = dryRate(T, hT, g) / (1.0 + hT / D.thick0);
    float rU = dryRate(U, hU, g) / (1.0 + (hU + 2.0 * hT) / D.thick0);   // covered paint dries slower
    float dt = saturate(ax.y + rT * D.hours), du = saturate(ax.x + rU * D.hours);
    float dn = (dt * hT + du * hU) / hn;
    if (D.apply < 0.5) { prev.write(half4(dn, saturate(hn / 20.0), 0, 1), p); return; }
    // each layer sets on its own; they never mix (light strokes over dark ones stay on top)
    ax.x = du; ax.y = dt;
    if (dn >= 0.995) {
        float4 R0 = float4(S.read(p, SL_DRY)), R1 = float4(S.read(p, SL_DRY + 1));
        // graphite stays under the paint: fold it into the ground before the dried layers go on top
        R0 = mix(R0, float4(0.075), ax.w * 0.92); R1 = mix(R1, float4(0.075), ax.w * 0.92); ax.w = 0;
        kmLayer(U, g, R0, R1);
        kmLayer(T, g, R0, R1);
        S.write(half4(R0), p, SL_DRY); S.write(half4(R1), p, SL_DRY + 1);
        ax.z += hn * D.shrink;
        U = pvz(); T = pvz(); ax.x = 0; ax.y = 0;
    }
    stS(S, p, 0, T); stS(S, p, SL_U, U);
    S.write(half4(ax), p, SL_AUX);
    H.write(float4(float(tooth.read(p).r) * toothAmp + ax.z + psum(U) + psum(T)), p);
}

// ---------------------------------------------------------------- display cache (color + normal/height), per dirty rect
struct ShadeP { int2 org; int2 sz; int2 surf; float relief; float k2; float4 mask; float maskR; float glossPaint; float glossGround; float dryGloss; };
kernel void kShade(texture2d_array<half, access::read> S [[texture(0)]],
                   texture2d<float, access::read> H [[texture(1)]],
                   texture2d<half, access::write> col [[texture(2)]],
                   texture2d<half, access::write> nrm [[texture(3)]],
                   constant Pig& g [[buffer(0)]], constant ShadeP& P [[buffer(1)]],
                   uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= uint2(P.sz))) return;
    int2 pi = int2(gid) + P.org;
    if (any(pi >= P.surf)) return;
    uint2 p = uint2(pi);
    PV T = ldR(S, p, 0), U = ldR(S, p, SL_U);
    float4 ax = float4(S.read(p, SL_AUX));
    float4 R0 = float4(S.read(p, SL_DRY)), R1 = float4(S.read(p, SL_DRY + 1));
    R0 = mix(R0, float4(0.075), ax.w * 0.92); R1 = mix(R1, float4(0.075), ax.w * 0.92);     // graphite
    kmLayer(U, g, R0, R1);
    kmLayer(T, g, R0, R1);
    float hT = psum(T), hU = psum(U);
    float hp = hT + hU + ax.z;
    float k2 = P.k2 * smoothstep(0.0, 3.0, hp);   // bare ground has a weaker interface effect
    float3 rgb = max(spec2rgb(saund(R0, k2), saund(R1, k2), g), 0.0);
    int2 m = P.surf - 1;
    float hl = H.read(uint2(clamp(pi - int2(1, 0), int2(0), m))).x, hr = H.read(uint2(clamp(pi + int2(1, 0), int2(0), m))).x;
    float hd = H.read(uint2(clamp(pi - int2(0, 1), int2(0), m))).x, hu = H.read(uint2(clamp(pi + int2(0, 1), int2(0), m))).x;
    float h = H.read(p).x;
    float sc = 0.5 * UNIT_MM / TEXEL_MM * P.relief;
    float wet = saturate((hT * mob(ax.y) + hU * mob(ax.x)) / 4.0);
    float paint = smoothstep(0.5, 6.0, hp);
    float gloss = mix(P.glossGround + 0.3 * ax.w, P.glossPaint * mix(P.dryGloss, 1.0, wet), paint);
    col.write(half4(half3(rgb), gloss), p);
    nrm.write(half4((hr - hl) * sc, (hu - hd) * sc, h * UNIT_MM, wet), p);
}

// ---------------------------------------------------------------- mip chain of the display caches, one level, within a region
kernel void kDown(texture2d<half, access::read> s [[texture(0)]], texture2d<half, access::write> d [[texture(1)]],
                  constant int4& R [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    int2 p = int2(gid) + R.xy;
    if (any(p >= R.zw) || any(p >= int2(d.get_width(), d.get_height()))) return;
    int2 m = int2(s.get_width(), s.get_height()) - 1;
    int2 q = p * 2;
    half4 a = s.read(uint2(min(q, m))) + s.read(uint2(min(q + int2(1, 0), m))) + s.read(uint2(min(q + int2(0, 1), m))) + s.read(uint2(min(q + 1, m)));
    d.write(a * 0.25h, uint2(p));
}

// ---------------------------------------------------------------- per tuft display color for bristle rendering
kernel void kBrushColor(device const float4* res [[buffer(0)]], device const float* cap [[buffer(1)]],
                        device float4* outc [[buffer(2)]], constant Pig& g [[buffer(3)]], constant int& nb [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
    if (int(i) > nb) return;
    PV r = ldRes(res, int(i));
    float v = psum(r);
    float fill = v / cap[i];
    float4 R0 = float4(0.8), R1 = float4(0.8);
    // render a thick coat of the carried mixture
    PV coat = v > 1e-5 ? r * (30.0 / v) : pvz();
    kmLayer(coat, g, R0, R1);
    float3 c = max(spec2rgb(saund(R0, 0.45), saund(R1, 0.45), g), 0.0);
    outc[i] = float4(c, saturate(fill * 6.0));
}

// ================================================================= rendering
struct Frame {
    float4x4 vp;
    float4x4 lvp;
    float4 cam;        // xyz camera position
    float4 ldir;       // xyz to light
    float4 lcol;
    float4 amb;
    float4 params;     // x heat on, y time, z hide handle, w unused
};

constexpr sampler sLin(filter::linear, mip_filter::linear, address::clamp_to_edge, max_anisotropy(8));
constexpr sampler sShadow(coord::normalized, filter::linear, address::clamp_to_edge, compare_func::less_equal);

constexpr sampler sPt(filter::nearest, address::clamp_to_edge);
constant float2 POISSON[16] = { float2(-0.94, -0.40), float2(0.95, -0.77), float2(-0.09, -0.93), float2(0.34, 0.29),
    float2(-0.92, 0.46), float2(-0.38, -0.12), float2(0.48, -0.31), float2(0.12, 0.89), float2(0.65, 0.66), float2(-0.55, 0.81),
    float2(0.79, 0.15), float2(-0.20, 0.40), float2(0.25, -0.60), float2(-0.70, -0.75), float2(0.0, 0.0), float2(0.92, -0.18) };
// soft shadows that harden near contact: the key light is a big window, so the penumbra grows with occluder distance
inline float shadowAt(float3 wp, constant Frame& F, depth2d<float> sm, float bias) {
    float4 l = F.lvp * float4(wp, 1);
    float3 q = l.xyz / l.w;
    float2 uv = float2(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5);
    if (any(uv < 0.0) || any(uv > 1.0)) return 1.0;
    float2 ts = 1.0 / float2(sm.get_width(), sm.get_height());
    float zr = q.z - bias;
    float a = h21(wp.xy * 7.3) * 6.2832;
    float2x2 rot = float2x2(cos(a), sin(a), -sin(a), cos(a));
    float sum = 0, cnt = 0;
    for (int i = 0; i < 16; i++) {
        float d = sm.sample(sPt, uv + rot * POISSON[i] * 48.0 * ts);
        if (d < zr) { sum += d; cnt += 1; }
    }
    if (cnt == 0) return 1.0;
    float distMM = (zr - sum / cnt) * 2200.0;
    float rad = clamp(distMM * 0.45, 1.5, 70.0);    // texels (~0.27 mm each)
    float s = 0;
    for (int i = 0; i < 16; i++) s += sm.sample_compare(sShadow, uv + rot * POISSON[i] * rad * ts, zr);
    s /= 16.0;
    // light bounced around the room fills in shadows of things far above the surface
    return mix(s, 1.0, saturate(distMM / 220.0) * 0.5);
}

// studio environment: warm room, large window toward the key light
inline float3 envCol(float3 r, float rough, constant Frame& F) {
    float3 L = F.ldir.xyz;
    float up = r.z;
    float3 base = mix(float3(0.10, 0.085, 0.07), float3(0.55, 0.52, 0.48), saturate(up * 0.5 + 0.5));
    float k = mix(300.0, 6.0, rough);
    float win = exp(k * (dot(r, L) - 1.0));
    return base * 0.6 + F.lcol.rgb * win * mix(0.9, 0.25, rough);
}

inline float3 lit(float3 P, float3 N, float3 V, float3 alb, float metal, float rough, float sh, float ao, constant Frame& F) {
    float3 L = F.ldir.xyz;
    float3 Hh = normalize(L + V);
    float nl = saturate(dot(N, L)), nv = saturate(dot(N, V)) + 1e-3, nh = saturate(dot(N, Hh));
    float a = max(rough * rough, 0.002), a2 = a * a;
    float D = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    float k = a * 0.5;
    float G = (nl / (nl * (1 - k) + k)) * (nv / (nv * (1 - k) + k));
    float3 f0 = mix(float3(0.04), alb, metal);
    float3 Fr = f0 + (1.0 - f0) * pow(1.0 - saturate(dot(Hh, V)), 5.0);
    float3 spec = D * G * Fr / (4.0 * nv * max(nl, 1e-3));
    float3 diff = alb * (1.0 - metal);
    float3 c = (diff / PI * 3.0 + spec) * F.lcol.rgb * nl * sh;
    float3 skyAmb = F.amb.rgb * (0.65 + 0.35 * N.z);
    c += diff * skyAmb * ao;
    float3 Fe = f0 + (1.0 - f0) * pow(1.0 - nv, 5.0) * (1.0 - rough);
    c += envCol(reflect(-V, N), rough, F) * Fe * ao * mix(1.0, sh, 0.5);
    return c;
}

// ---------------------------------------------------------------- paint surfaces (canvas top / palette top)
struct SurfDraw { float4 orgSize; float4 mask; float4 misc; float4 heat; };  // orgSize: world origin xy (mm), size wh (texels); misc: z, maskR, kind
struct SV { float4 pos [[position]]; float3 wp; float2 tx; };
vertex SV surfVert(uint vid [[vertex_id]], constant Frame& F [[buffer(0)]], constant SurfDraw& D [[buffer(1)]]) {
    float2 c = float2((vid & 1) ? D.mask.z : D.mask.x, (vid & 2) ? D.mask.w : D.mask.y);   // texel coords
    float3 wp = float3(D.orgSize.xy + c * TEXEL_MM, D.misc.x);
    SV o; o.pos = F.vp * float4(wp, 1); o.wp = wp; o.tx = c; return o;
}
fragment float4 surfFrag(SV in [[stage_in]], constant Frame& F [[buffer(0)]], constant SurfDraw& D [[buffer(1)]],
                         texture2d<half> col [[texture(0)]], texture2d<half> nrm [[texture(1)]],
                         depth2d<float> sm [[texture(2)]], texture2d<half> prev [[texture(3)]], depth2d<float> smH [[texture(5)]]) {
    if (!inMask(in.tx, D.mask, D.misc.y, D.heat)) discard_fragment();
    float2 uv = in.tx / D.orgSize.zw;
    float4 c = float4(col.sample(sLin, uv));
    float4 n = float4(nrm.sample(sLin, uv));
    float hb = float(nrm.sample(sLin, uv, bias(4.0)).z);
    float3 N = normalize(float3(-n.x, -n.y, 1.0));
    float3 wp = in.wp + float3(0, 0, n.z);
    float3 V = normalize(F.cam.xyz - wp);
    float ao = saturate(1.0 - (hb - n.z) * 1.6);
    float sh = shadowAt(wp, F, sm, 0.0015) * shadowAt(wp, F, smH, 0.0015);
    float rough = mix(0.85, 0.16, c.a);
    if (F.params.w > 0.5) c.rgb = float3(0.5);
    float3 rgb = lit(wp, N, V, c.rgb, 0.0, rough, sh, ao, F);
    if (F.params.x > 0.5) {
        // drying preview: wet = blue, tacky = yellow, dry = red; bare canvas dimmed
        float4 pv = float4(prev.sample(sLin, uv));
        float d = pv.x;
        float3 hc = mix(float3(0.10, 0.35, 1.0), float3(1.0, 0.85, 0.15), smoothstep(0.1, 0.55, d));
        hc = mix(hc, float3(0.95, 0.15, 0.08), smoothstep(0.55, 0.95, d));
        float lum = dot(rgb, float3(0.3, 0.5, 0.2));
        rgb = mix(float3(lum * 0.5), hc * (0.55 + 0.6 * lum), smoothstep(0.0, 0.15, pv.y));
    }
    return float4(rgb, 1);
}

// ---------------------------------------------------------------- 3D objects
struct Vtx { float4 p; float4 n; };       // p.w = u, n.w = v
struct ObjDraw { float4x4 model; float4 base; float4 mat; float4 extra; };   // mat: metal, rough, mode, fade-near
struct OV { float4 pos [[position]]; float3 wp; float3 n; float2 uv; float3 lp; };
vertex OV objVert(uint vid [[vertex_id]], device const Vtx* V [[buffer(0)]], constant Frame& F [[buffer(1)]], constant ObjDraw& D [[buffer(2)]]) {
    Vtx v = V[vid];
    float4 wp = D.model * float4(v.p.xyz, 1);
    OV o; o.pos = F.vp * wp; o.wp = wp.xyz; o.n = normalize((D.model * float4(v.n.xyz, 0)).xyz); o.uv = float2(v.p.w, v.n.w); o.lp = v.p.xyz;
    return o;
}
vertex float4 shadowVert(uint vid [[vertex_id]], device const Vtx* V [[buffer(0)]], constant Frame& F [[buffer(1)]], constant ObjDraw& D [[buffer(2)]]) {
    return F.lvp * (D.model * float4(V[vid].p.xyz, 1));
}
struct OF { float4 c [[color(0)]]; uint mask [[sample_mask]]; };
fragment OF objFrag(OV in [[stage_in]], constant Frame& F [[buffer(1)]], constant ObjDraw& D [[buffer(2)]],
                    depth2d<float> sm [[texture(2)]], texture2d<half> atlas [[texture(4)]], depth2d<float> smH [[texture(5)]],
                    bool front [[front_facing]]) {
    float3 N = normalize(in.n);
    float3 V = normalize(F.cam.xyz - in.wp);
    if (dot(N, V) < 0 && D.mat.z > 9.5) N = -N;
    float3 alb = D.base.rgb;
    float metal = D.mat.x, rough = D.mat.y;
    int mode = int(D.mat.z + 0.5) % 10;
    float2 uv = in.uv;
    if (mode == 1) {            // tube: label band / aluminium
        float2 lu = float2(uv.y, uv.x);  // u around, v along
        if (uv.y > D.extra.x && uv.y < D.extra.y) {
            float2 a = float2((uv.y - D.extra.x) / (D.extra.y - D.extra.x), fract(uv.x + 0.25));
            float2 auv = float2(a.x, D.extra.z + a.y * D.extra.w);
            alb = float3(atlas.sample(sLin, auv).rgb);
            metal = 0; rough = 0.42;
        } else {
            float cr = vn(float2(lu.x * 30.0, lu.y * 8.0)) - 0.5;
            N = normalize(N + 0.08 * cr * float3(1, 1, 0.3));
            alb = float3(0.86, 0.86, 0.85); metal = 1; rough = 0.28 + 0.1 * cr;
        }
    } else if (mode == 2) {     // wood (palette edge, tray, natural handles)
        float3 c = woodCol(in.lp.xy * D.extra.x + D.extra.yz, D.extra.w, D.base.rgb, D.base.rgb * 0.55);
        alb = c;
    } else if (mode == 3) {     // ribbed plastic cap
        float rib = sin(uv.x * 2.0 * PI * 24.0);
        float3 t = normalize(cross(N, float3(0, 0, 1)) + 1e-4);
        N = normalize(N + t * rib * 0.25);
    } else if (mode == 4) {     // canvas edge fabric
        float2 q = uv * float2(D.extra.x, D.extra.y);
        float w = 0.5 + 0.5 * sin(q.x * 2.0 * PI) * sin(q.y * 2.0 * PI);
        alb *= 0.9 + 0.12 * w;
    } else if (mode == 5) {     // table
        alb = woodCol(in.wp.xy * 0.6, 3.0, float3(0.30, 0.18, 0.10), float3(0.13, 0.075, 0.04));
    }
    float sh = shadowAt(in.wp + N * 0.4, F, sm, 0.004) * (D.mat.w > 0 ? 1.0 : shadowAt(in.wp + N * 0.4, F, smH, 0.004));
    float3 c = lit(in.wp, N, V, alb, metal, rough, sh, 1.0, F);
    OF o; o.c = float4(c, 1); o.mask = 0xF;
    if (D.mat.w > 0) {      // fade when near the camera (brush handle), dithered over MSAA samples
        float dz = F.cam.z - in.wp.z;
        float a = smoothstep(D.mat.w * 0.35, D.mat.w, dz);
        if (F.params.z > 0.5) a = 0;
        int n = int(clamp(floor(a * 4.0 + h21(in.pos.xy)), 0.0, 4.0));
        o.mask = (1u << n) - 1u;
        if (n == 0) discard_fragment();
    }
    return o;
}

// ---------------------------------------------------------------- bristle hairs (camera facing ribbons along cubic curves)
struct Hair { float4 p0; float4 p1; float4 p2; float4 p3; };   // p0.w = tuft id, p1.w = width (mm), p2.w = paint start t, p3.w = shade jitter
struct HV { float4 pos [[position]]; float3 wp; float3 t; float s; float paint [[flat]]; float4 pc [[flat]]; float jit [[flat]]; };
constant int HSEG = 10;
vertex HV hairVert(uint vid [[vertex_id]], uint iid [[instance_id]], device const Hair* Hs [[buffer(0)]],
                   constant Frame& F [[buffer(1)]], device const float4* tc [[buffer(2)]]) {
    Hair h = Hs[iid];
    int seg = vid / 2; float side = (vid & 1) ? 1.0 : -1.0;
    float t = float(seg) / float(HSEG);
    float u = 1.0 - t;
    float3 P = u * u * u * h.p0.xyz + 3 * u * u * t * h.p1.xyz + 3 * u * t * t * h.p2.xyz + t * t * t * h.p3.xyz;
    float3 T = normalize(3 * u * u * (h.p1.xyz - h.p0.xyz) + 6 * u * t * (h.p2.xyz - h.p1.xyz) + 3 * t * t * (h.p3.xyz - h.p2.xyz) + 1e-5);
    float3 V = normalize(F.cam.xyz - P);
    float3 B = normalize(cross(T, V) + 1e-5);
    float w = h.p1.w * mix(1.0, 0.45, t * t);
    P += B * side * w * 0.5;
    HV o; o.pos = F.vp * float4(P, 1); o.wp = P; o.t = T; o.s = t;
    o.paint = h.p2.w; o.pc = tc[int(h.p0.w)]; o.jit = h.p3.w;
    return o;
}
fragment float4 hairFrag(HV in [[stage_in]], constant Frame& F [[buffer(1)]], constant float4& baseCol [[buffer(3)]]) {
    float3 T = normalize(in.t);
    float3 L = F.ldir.xyz, V = normalize(F.cam.xyz - in.wp);
    float3 alb = baseCol.rgb * (0.85 + 0.3 * in.jit);
    float shine = 1.0;
    // paint coats the lower part of the bristles
    float pa = in.pc.a * smoothstep(in.paint - 0.08, in.paint + 0.05, in.s);
    alb = mix(alb, in.pc.rgb, pa);
    shine = mix(1.0, 1.6, pa);
    float tl = dot(T, L), tv = dot(T, V);
    float sl = sqrt(max(1.0 - tl * tl, 0.0)), sv = sqrt(max(1.0 - tv * tv, 0.0));
    float diff = sl;
    float spec = pow(saturate(sl * sv - tl * tv), 40.0) * 0.35 * shine;
    float3 c = alb * (diff * F.lcol.rgb * 0.75 + F.amb.rgb) + spec * F.lcol.rgb;
    return float4(c, 1);
}
