// Paint engine: pigments, paint surfaces, bristle brushes, tubes, undo, drying.
import Metal
import simd
import Foundation

typealias V2 = SIMD2<Float>
typealias V3 = SIMD3<Float>
typealias V4 = SIMD4<Float>
typealias I2 = SIMD2<Int32>

let PPM: Float = 10            // texels per mm
let UNIT: Float = 0.01         // paint thickness unit (mm)
let TILE = 128
let NSLICE = 11
let NCH = 16                   // paint channels per texel (pigments + medium)

// ---------------------------------------------------------------- GPU
final class GPU {
    static let shared = GPU()
    let dev: MTLDevice
    let q: MTLCommandQueue
    let lib: MTLLibrary
    private var cps: [String: MTLComputePipelineState] = [:]
    init() {
        dev = MTLCreateSystemDefaultDevice()!
        q = dev.makeCommandQueue()!
        let url = Bundle.main.url(forResource: "shaders", withExtension: "metallib")
            ?? URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("shaders.metallib")
        lib = try! dev.makeLibrary(URL: url)
    }
    func cp(_ name: String) -> MTLComputePipelineState {
        if let p = cps[name] { return p }
        let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
        cps[name] = p
        return p
    }
    func run(_ enc: MTLComputeCommandEncoder, _ name: String, _ w: Int, _ h: Int = 1) {
        let p = cp(name)
        enc.setComputePipelineState(p)
        let tw = h == 1 ? min(p.maxTotalThreadsPerThreadgroup, 256) : 16
        let th = h == 1 ? 1 : max(1, min(p.maxTotalThreadsPerThreadgroup / 16, 16))
        enc.dispatchThreads(MTLSize(width: max(w, 1), height: max(h, 1), depth: 1), threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
    }
    func buffer(_ n: Int, shared: Bool = true) -> MTLBuffer {
        dev.makeBuffer(length: max(n, 16), options: shared ? .storageModeShared : .storageModePrivate)!
    }
    func tex(_ fmt: MTLPixelFormat, _ w: Int, _ h: Int, slices: Int = 0, mips: Bool = false, rt: Bool = false, shared: Bool = false) -> MTLTexture {
        let d = MTLTextureDescriptor()
        d.textureType = slices > 0 ? .type2DArray : .type2D
        d.pixelFormat = fmt; d.width = w; d.height = h
        if slices > 0 { d.arrayLength = slices }
        if mips { d.mipmapLevelCount = Int(log2(Double(max(w, h)))) + 1 }
        d.usage = rt ? [.renderTarget, .shaderRead] : [.shaderRead, .shaderWrite]
        d.storageMode = shared ? .shared : .private
        return dev.makeTexture(descriptor: d)!
    }
}

// ---------------------------------------------------------------- pigments (8 spectral bands, 410..690 nm)
struct Pigment { let name: String; let code: String; let rinf: [Float]; let s: Float; let dry: Float }

let PIGMENTS: [Pigment] = [
    Pigment(name: "Titanium White", code: "PW6", rinf: [0.86, 0.91, 0.93, 0.94, 0.94, 0.94, 0.935, 0.93], s: 4.0, dry: 0.012),
    Pigment(name: "Ivory Black", code: "PBk9", rinf: [0.034, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040], s: 0.6, dry: 0.010),
    Pigment(name: "Cadmium Yellow Light", code: "PY35", rinf: [0.06, 0.07, 0.13, 0.58, 0.82, 0.85, 0.86, 0.86], s: 2.0, dry: 0.014),
    Pigment(name: "Cadmium Orange", code: "PO20", rinf: [0.06, 0.06, 0.06, 0.08, 0.26, 0.76, 0.83, 0.85], s: 2.0, dry: 0.014),
    Pigment(name: "Yellow Ochre", code: "PY43", rinf: [0.09, 0.11, 0.16, 0.27, 0.44, 0.53, 0.57, 0.59], s: 1.6, dry: 0.022),
    Pigment(name: "Cadmium Red Light", code: "PR108", rinf: [0.07, 0.05, 0.045, 0.045, 0.07, 0.56, 0.82, 0.86], s: 2.0, dry: 0.014),
    Pigment(name: "Alizarin Crimson", code: "PR83", rinf: [0.11, 0.06, 0.03, 0.018, 0.022, 0.19, 0.50, 0.62], s: 0.25, dry: 0.012),
    Pigment(name: "Ultramarine Blue", code: "PB29", rinf: [0.30, 0.35, 0.16, 0.045, 0.022, 0.022, 0.035, 0.09], s: 0.6, dry: 0.016),
    Pigment(name: "Phthalo Blue", code: "PB15", rinf: [0.04, 0.08, 0.06, 0.02, 0.006, 0.003, 0.003, 0.004], s: 0.35, dry: 0.02),
    Pigment(name: "Viridian", code: "PG18", rinf: [0.06, 0.12, 0.30, 0.25, 0.08, 0.03, 0.02, 0.02], s: 0.4, dry: 0.016),
    Pigment(name: "Burnt Sienna", code: "PBr7", rinf: [0.05, 0.05, 0.055, 0.07, 0.12, 0.30, 0.42, 0.48], s: 0.9, dry: 0.03),
    Pigment(name: "Burnt Umber", code: "PBr7", rinf: [0.04, 0.04, 0.045, 0.05, 0.07, 0.11, 0.15, 0.18], s: 1.0, dry: 0.04),
    Pigment(name: "Glazing Medium", code: "alkyd", rinf: [0.25, 0.4, 0.6, 0.8, 0.9, 0.9, 0.9, 0.9], s: 0.0008, dry: 0.05),
    // added later; appended so saved paintings keep their tube numbers
    Pigment(name: "Lemon Yellow", code: "PY175", rinf: [0.05, 0.07, 0.32, 0.80, 0.87, 0.88, 0.88, 0.88], s: 1.2, dry: 0.016),
    Pigment(name: "Quinacridone Magenta", code: "PR122", rinf: [0.30, 0.24, 0.07, 0.018, 0.025, 0.28, 0.66, 0.76], s: 0.3, dry: 0.016),
    Pigment(name: "Cerulean Blue", code: "PB35", rinf: [0.38, 0.55, 0.50, 0.28, 0.11, 0.06, 0.05, 0.06], s: 1.6, dry: 0.016),
]
// order of the tubes in the tray
let TUBE_ORDER = [0, 13, 2, 3, 4, 5, 6, 14, 7, 15, 8, 9, 10, 11, 1, 12]
let MEDIUM = 12
let NPIG = PIGMENTS.count

// band weights: CIE 1931 (Wyman et al. fit) x D65, then to linear sRGB, rows normalised so a flat R=1 is white
func spectralMatrix() -> [[Float]] {
    func g(_ x: Double, _ mu: Double, _ s1: Double, _ s2: Double) -> Double { let t = (x - mu) / (x < mu ? s1 : s2); return exp(-0.5 * t * t) }
    let d65: [Double] = [54.6, 82.75, 91.49, 93.43, 86.68, 104.86, 117.01, 117.81, 114.86, 115.92, 108.81, 109.35, 107.80, 104.79, 107.69, 104.41,
                         104.05, 100.0, 96.33, 95.79, 88.69, 90.01, 89.60, 87.70, 83.29, 83.70, 80.03, 80.21, 82.28, 78.28, 69.72, 71.61, 74.35]
    var W = [[Double]](repeating: [Double](repeating: 0, count: 8), count: 3)
    for l in 390..<710 {
        let x = Double(l)
        let fi = (x - 390) / 10, i0 = Int(fi), fr = fi - Double(i0)
        let e = d65[i0] * (1 - fr) + d65[min(i0 + 1, d65.count - 1)] * fr
        let xb = 1.056 * g(x, 599.8, 37.9, 31.0) + 0.362 * g(x, 442.0, 16.0, 26.7) - 0.065 * g(x, 501.1, 20.4, 26.2)
        let yb = 0.821 * g(x, 568.8, 46.9, 40.5) + 0.286 * g(x, 530.9, 16.3, 31.1)
        let zb = 1.217 * g(x, 437.0, 11.8, 36.0) + 0.681 * g(x, 459.0, 26.0, 13.8)
        let b = (l - 390) / 40
        W[0][b] += e * xb; W[1][b] += e * yb; W[2][b] += e * zb
    }
    let ysum = W[1].reduce(0, +)
    let X2R: [[Double]] = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]
    var M = [[Float]](repeating: [Float](repeating: 0, count: 8), count: 3)
    for r in 0..<3 {
        var row = [Double](repeating: 0, count: 8)
        for b in 0..<8 { for c in 0..<3 { row[b] += X2R[r][c] * W[c][b] / ysum } }
        let s = row.reduce(0, +)
        row = row.map { $0 / s }
        M[r] = row.map { Float($0) }
    }
    return M
}
let SPECM = spectralMatrix()

func rgb2spec(_ c: V3) -> [Float] {
    [c.z, c.z, 0.7 * c.z + 0.3 * c.y, 0.15 * c.z + 0.85 * c.y, c.y, 0.35 * c.y + 0.65 * c.x, c.x, c.x]
}

func pigmentTable() -> [Float] {
    var K = [Float](), S = [Float]()
    for p in PIGMENTS {
        for r in p.rinf { let rr = max(r, 0.001); K.append(p.s * (1 - rr) * (1 - rr) / (2 * rr)); S.append(p.s) }
    }
    var M = [Float]()
    for r in 0..<3 { M += SPECM[r] }
    K += [Float](repeating: 0, count: (NCH - PIGMENTS.count) * 8); S += [Float](repeating: 0, count: (NCH - PIGMENTS.count) * 8)
    var dry = PIGMENTS.map { $0.dry }
    dry += [Float](repeating: 0, count: NCH - dry.count)
    let bristle = rgb2spec(V3(0.80, 0.68, 0.48))
    return K + S + M + dry + bristle
}

func specToRGB(_ R: [Float]) -> V3 {
    var c = V3.zero
    for b in 0..<8 { c += V3(SPECM[0][b], SPECM[1][b], SPECM[2][b]) * R[b] }
    return c
}
// CPU KM: masstone of a pigment mix (fractions) over white at given thickness, for UI swatches
func mixColor(_ v: [Float], thick: Float = 40, ground: Float = 0.9) -> V3 {
    var R = [Float]()
    for b in 0..<8 {
        var kx: Float = 0, sx: Float = 0
        for (i, p) in PIGMENTS.enumerated() where i < v.count {
            let rr = max(p.rinf[b], 0.001)
            kx += v[i] * thick * p.s * (1 - rr) * (1 - rr) / (2 * rr); sx += v[i] * thick * p.s
        }
        let s = max(sx, 1e-4), a = 1 + kx / s, bb = sqrt(max(a * a - 1, 1e-8)), x = min(max(bb * s, 1e-5), 30)
        let cth = (1 + exp(-2 * x)) / (1 - exp(-2 * x))
        let r = (1 - ground * (a - bb * cth)) / (a - ground + bb * cth)
        let k2: Float = 0.45
        R.append(0.96 * (1 - k2) * r / (1 - k2 * r))
    }
    return simd_clamp(specToRGB(R), V3(repeating: 0), V3(repeating: 1))
}
func linToSRGB(_ x: Float) -> Float { x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1 / 2.4) - 0.055 }

// ---------------------------------------------------------------- shared param structs (mirror shaders.metal)
struct InitP { var size: I2; var kind: Int32; var period: Float; var g0: V4; var g1: V4; var toothAmp: Float; var seed: Float; var org: V2 }
struct BrG { var seg: V4; var p0: V4; var p1: V4 }
struct FPP { var vp: V2; var org: I2; var tsz: I2; var toothAmp: Float; var groove: Float; var conform: Float; var coat: Float; var pad: V2 = .zero }
struct StepP { var org: I2; var fpsz: I2; var surf: I2; var nb: Int32; var pad0: Int32 = 0; var alpha: Float; var rho: Float; var ratePass: Float; var mixPass: Float; var mask: V4; var maskR: Float; var toothAmp: Float; var eps: Float; var film: Float; var sinkRep: Float; var groove: Float; var seed: Float; var pad1: Float = 0; var hole: V4 }
struct ApplyP { var nb: Int32; var feed: Float; var wick: Float; var clean: Float; var nz: Int32; var pad: SIMD3<Int32> = .zero }
struct SqzP { var org: I2; var sz: I2; var surf: I2; var c: V2; var comp0: V4; var comp1: V4; var comp2: V4; var comp3: V4; var rad: Float; var amt: Float; var hmax: Float; var toothAmp: Float; var mask: V4; var maskR: Float; var seed: Float; var dir: V2; var hole: V4 }
struct PenP { var org: I2; var sz: I2; var surf: I2; var a: V2; var b: V2; var r: Float; var press: Float; var toothAmp: Float; var share: Float; var mask: V4; var maskR: Float; var seed: Float; var pad: Float; var hole: V4 }
struct DryP { var size: I2; var hours: Float; var thick0: Float; var shrink: Float; var apply: Float; var pad: V2 = .zero }
struct ShadeP { var org: I2; var sz: I2; var surf: I2; var relief: Float; var k2: Float; var mask: V4; var maskR: Float; var glossPaint: Float; var glossGround: Float; var dryGloss: Float }

struct IRect {
    var x0, y0, x1, y1: Int
    var empty: Bool { x1 <= x0 || y1 <= y0 }
    func union(_ o: IRect) -> IRect { empty ? o : (o.empty ? self : IRect(x0: min(x0, o.x0), y0: min(y0, o.y0), x1: max(x1, o.x1), y1: max(y1, o.y1))) }
    func clip(_ w: Int, _ h: Int) -> IRect { IRect(x0: max(x0, 0), y0: max(y0, 0), x1: min(x1, w), y1: min(y1, h)) }
    static let none = IRect(x0: 0, y0: 0, x1: 0, y1: 0)
}

// ---------------------------------------------------------------- paint surface
final class Surface {
    let id: Int
    let kind: Int          // 0 canvas, 1 palette
    let w: Int, h: Int
    var origin: V2         // world mm of texel (0,0)
    var z: Float           // world height of the top face (mm)
    let grainOrg: V2       // where the wood grain / weave pattern is anchored
    var mask: V4           // valid region in texels
    var maskR: Float
    var hole = V4.zero     // thumb hole (x, y, r) in texels
    let texture: Int       // canvas: 0 cotton, 2 linen, 3 smooth panel
    let toothAmp: Float
    let state, height, tooth, color, nrm, prev: MTLTexture
    let colorLevels: [MTLTexture], nrmLevels: [MTLTexture]
    var dirty = IRect.none
    var painted: [Bool]
    var packed: [Data?]    // encoded tiles as last saved/loaded (nil once changed), so saving only re-encodes what changed
    var tilesX: Int { (w + TILE - 1) / TILE }
    var tilesY: Int { (h + TILE - 1) / TILE }

    init(id: Int, kind: Int, widthMM: Float, heightMM: Float, origin: V2, z: Float, cornerMM: Float, texture: Int = 0) {
        self.id = id; self.kind = kind; self.origin = origin; self.z = z; self.texture = texture; grainOrg = origin
        w = Int(widthMM * PPM); h = Int(heightMM * PPM)
        mask = V4(0, 0, Float(w), Float(h)); maskR = cornerMM * PPM
        toothAmp = kind == 1 ? 1.5 : [11, 0, 8, 3][texture]
        let g = GPU.shared
        state = g.tex(.rgba16Float, w, h, slices: NSLICE)
        height = g.tex(.r32Float, w, h)
        tooth = g.tex(.r16Float, w, h)
        prev = g.tex(.rgba16Float, w, h)
        func levels(_ t: MTLTexture) -> [MTLTexture] { (0..<t.mipmapLevelCount).map { t.makeTextureView(pixelFormat: .rgba16Float, textureType: .type2D, levels: $0..<($0 + 1), slices: 0..<1)! } }
        let c = g.tex(.rgba16Float, w, h, mips: true), n = g.tex(.rgba16Float, w, h, mips: true)
        color = c; nrm = n
        colorLevels = levels(c); nrmLevels = levels(n)
        painted = [Bool](repeating: false, count: ((w + TILE - 1) / TILE) * ((h + TILE - 1) / TILE))
        packed = [Data?](repeating: nil, count: painted.count)
    }
    func reset(_ cmd: MTLCommandBuffer) {
        let gesso = [Float]([0.80, 0.85, 0.87, 0.88, 0.88, 0.88, 0.875, 0.87])
        var p = InitP(size: I2(Int32(w), Int32(h)), kind: Int32(kind == 1 ? 1 : (texture == 0 ? 0 : texture)), period: texture == 2 ? 4.2 : 5.6,
                      g0: V4(gesso[0], gesso[1], gesso[2], gesso[3]), g1: V4(gesso[4], gesso[5], gesso[6], gesso[7]),
                      toothAmp: toothAmp, seed: Float(id) * 13.7 + 1.3, org: grainOrg)
        let e = cmd.makeComputeCommandEncoder()!
        e.setTexture(state, index: 0); e.setTexture(tooth, index: 1); e.setTexture(height, index: 2)
        e.setBytes(&p, length: MemoryLayout<InitP>.stride, index: 0)
        GPU.shared.run(e, "kInit", w, h)
        e.endEncoding()
        dirty = IRect(x0: 0, y0: 0, x1: w, y1: h)
        painted = painted.map { _ in false }
        packed = packed.map { _ in nil }
    }
    func toTexel(_ p: V2) -> V2 { (p - origin) * PPM }
    func toWorld(_ t: V2) -> V2 { origin + t / PPM }
    func contains(_ world: V2) -> Bool {
        let t = toTexel(world)
        let c = V2((mask.x + mask.z) / 2, (mask.y + mask.w) / 2), hh = V2((mask.z - mask.x) / 2, (mask.w - mask.y) / 2)
        let q = simd_abs(t - c) - hh + V2(repeating: maskR)
        if hole.z > 0 && simd_distance(t, V2(hole.x, hole.y)) < hole.z { return false }
        return simd_length(simd_max(q, .zero)) + min(max(q.x, q.y), 0) - maskR <= 0
    }
    // within `mm` of the paintable area (a brush centred just off the edge still reaches it)
    func near(_ world: V2, _ mm: Float) -> Bool {
        let t = toTexel(world)
        let c = V2((mask.x + mask.z) / 2, (mask.y + mask.w) / 2), hh = V2((mask.z - mask.x) / 2, (mask.w - mask.y) / 2)
        let q = simd_abs(t - c) - hh + V2(repeating: maskR)
        return simd_length(simd_max(q, .zero)) + min(max(q.x, q.y), 0) - maskR <= mm * PPM
    }
    func markPainted(_ r: IRect) {
        let c = r.clip(w, h); if c.empty { return }
        for ty in (c.y0 / TILE)...((c.y1 - 1) / TILE) { for tx in (c.x0 / TILE)...((c.x1 - 1) / TILE) { painted[ty * tilesX + tx] = true } }
    }
}

// ---------------------------------------------------------------- bristle brush
func noise1(_ x: Float, _ seed: Float) -> Float {
    func h(_ n: Float) -> Float { let s = sin(n * 127.1 + seed * 311.7) * 43758.5453; return s - floor(s) }
    let i = floor(x), f = x - i, u = f * f * (3 - 2 * f)
    return h(i) * (1 - u) + h(i + 1) * u
}

struct Rng {
    var s: UInt64
    mutating func next() -> Float { s = s &* 6364136223846793005 &+ 1442695040888963407; return Float((s >> 40) & 0xFFFFFF) / Float(0x1000000) }
    mutating func gauss() -> Float { let u = max(next(), 1e-7), v = next(); return sqrt(-2 * log(u)) * cos(2 * .pi * v) }
}

final class Brush {
    let name: String
    let round: Bool
    let knife: Bool             // palette knife: a rigid blade, simulated as tightly packed stiff tufts
    let pencil: Bool            // graphite pencil: no bristles, marks the ground directly
    let width: Float, thick: Float, length: Float
    let n: Int
    let tuftR: Float
    var offs = [V2](), lens = [Float](), dirj = [V2](), rads = [Float]()
    var clump = [V2]()          // per tuft: offset of its clump centre (wet bristles gather into clumps)
    var clumpLag = [V2]()
    var clumpOf = [Int]()
    var clumpiness: Float = 0.55
    var clumpJit = [V2]()
    var travel: Float = 0           // distance the brush has moved (drives bristle wander)
    var foot: Float = 0             // mean length of bristle contact along the surface (sets the substep size)
    var slant: Float = 1            // rms sine between touching bristles and their motion
    var tip = [V2](), tipZ = [Float](), touching = [Bool](), bendPt = [V2](), eq = [V2](), lastTip = [V2]()
    var root = [V3]()
    let res, dres, cap, colors, zoneBuf, flow: MTLBuffer
    let nz: Int                 // reservoir zones across the brush
    let capTuft: Float
    // pose
    var ferrule = V3(0, 0, 30), axis = V3(0, 0, -1), e1 = V3(1, 0, 0), e2 = V3(0, 1, 0)
    var baseZ: Float = 0     // world z of the surface the bristle positions are relative to
    var handleColor: V3
    var handleLen: Float
    var hairColor: V3
    var stiffness: Float

    var lie = [Float](), pressJ = [Float]()
    var pressDepth: Float = 0.62     // how far full pressure pushes the bristles down, as a fraction of their length

    init(name: String, round: Bool, width: Float, thick: Float, length: Float, n: Int, tuftR: Float, handle: V3, seed: UInt64, knife: Bool = false, pencil: Bool = false) {
        self.name = name; self.round = round; self.width = width; self.thick = thick; self.length = length; self.n = n; self.tuftR = tuftR; self.knife = knife; self.pencil = pencil
        handleColor = handle; handleLen = knife ? 120 : (pencil ? 160 : 150 + width * 3); hairColor = V3(0.82, 0.70, 0.50); stiffness = knife ? 20 : 1
        var rng = Rng(s: seed)
        for i in 0..<n {
            var o: V2
            if round {
                let a = Float(i) * 2.39996323, r = sqrt((Float(i) + 0.5) / Float(n))
                o = V2(cos(a), sin(a)) * r * width * 0.5 + V2(rng.gauss(), rng.gauss()) * 0.04 * width
                let rf = simd_length(o) / (width * 0.5)
                lens.append(length * (1 - 0.32 * rf * rf) * (1 + 0.035 * rng.gauss()))
            } else if knife {
                o = V2((rng.next() - 0.5) * width, (rng.next() - 0.5) * thick)
                let ex = abs(o.x) / (width * 0.5)
                lens.append(length * (1 - 0.55 * ex * ex))       // trowel blade: pointed tip
            } else {
                o = V2((rng.next() - 0.5) * width, (rng.next() - 0.5) * thick)
                let ex = abs(o.x) / (width * 0.5)
                lens.append(length * (1 - 0.06 * pow(ex, 6)) * (1 + 0.03 * rng.gauss()))
            }
            offs.append(o)
            dirj.append(knife ? .zero : V2(rng.gauss(), rng.gauss()) * 0.006)
            rads.append(tuftR * (0.85 + 0.3 * rng.next()))
        }
        tip = [V2](repeating: .zero, count: n); tipZ = [Float](repeating: 10, count: n)
        touching = [Bool](repeating: false, count: n); bendPt = tip; eq = tip; lastTip = tip
        lie = [Float](repeating: 0, count: n); pressJ = lie
        root = [V3](repeating: .zero, count: n)
        // zones: rounds = core + 6 sectors, flats = 10 strips across the width
        var zones = [Int32]()
        for o in offs {
            if round {
                let r = simd_length(o) / (width * 0.5)
                zones.append(r < 0.35 ? 0 : Int32(1 + Int((atan2(o.y, o.x) + .pi) / (2 * .pi) * 6) % 6))
            } else { zones.append(Int32(min(max(Int((o.x / width + 0.5) * 10), 0), 9))) }
        }
        nz = round ? 7 : 10
        let g = GPU.shared
        res = g.buffer((n + nz) * NCH * 4); dres = g.buffer((n + 1) * NCH * 4); cap = g.buffer((n + nz) * 4); colors = g.buffer((n + 1) * 16)
        flow = g.buffer(n * NCH * 4)
        zoneBuf = g.dev.makeBuffer(bytes: zones, length: n * 4, options: .storageModeShared)!
        memset(res.contents(), 0, res.length); memset(dres.contents(), 0, dres.length)
        // paint the bundle can hold: ~30% of its volume (mm^3 -> unit*texel^2 is x1e4); a quarter sits on the tips
        let vol = round ? Float.pi * width * width / 4 * length * 0.55 : width * thick * length * 0.7
        let total = vol * 0.45 * 1e4
        capTuft = total * 0.25 / Float(n)
        let cp = cap.contents().bindMemory(to: Float.self, capacity: n + nz)
        for i in 0..<n { cp[i] = capTuft }
        for z in 0..<nz { cp[n + z] = 0 }
        for i in 0..<n { cp[n + Int(zones[i])] += capTuft * 3 }
        for z in 0..<nz { cp[n + z] = max(cp[n + z], 1) }
        reclump(seed: seed)
    }

    // the reservoir is GPU state: clear it in command order, after any strokes already encoded
    func clean(_ cmd: MTLCommandBuffer) { let e = cmd.makeBlitCommandEncoder()!; e.fill(buffer: res, range: 0..<res.length, value: 0); e.endEncoding() }

    // bristles part into a new set of clumps on every stroke
    func reclump(seed: UInt64) {
        var rng = Rng(s: seed &* 2654435761 &+ 99)
        let k = round ? max(4, Int(width * 2.4)) : max(4, Int(width * 1.7))
        var centers = [V2]()
        for _ in 0..<k {
            if round { let a = rng.next() * 2 * .pi, r = sqrt(rng.next()) * width * 0.5; centers.append(V2(cos(a), sin(a)) * r) }
            else { centers.append(V2((rng.next() - 0.5) * width, (rng.next() - 0.5) * thick * 0.6)) }
        }
        clump = []; clumpOf = []
        for o in offs {
            var best = 0, bd = Float.greatestFiniteMagnitude
            for (i, c) in centers.enumerated() {
                let d = round ? simd_length_squared(o - c) : pow(o.x - c.x, 2) + 0.15 * pow(o.y - c.y, 2)
                if d < bd { bd = d; best = i }
            }
            clump.append(centers[best]); clumpOf.append(best)
        }
        clumpLag = [V2](repeating: .zero, count: k)
        clumpJit = (0..<k).map { _ in knife ? .zero : V2(rng.gauss(), rng.gauss()) * 0.03 }
        clumpiness = knife ? 0 : 0.45 + 0.3 * rng.next()
    }

    // place the brush: tips' rest centroid over `target` (surface-local mm), pressed by `pressure`
    func pose(target: V2, pressure: Float, tilt: V2, roll: Float) {
        var tv = tilt
        let tl = simd_length(tv); if tl > 0.85 { tv *= 0.85 / tl }
        axis = simd_normalize(V3(-tv.x, -tv.y, -sqrt(max(1 - simd_length_squared(tv), 0.05))))
        var r = V3(cos(roll), sin(roll), 0)
        r -= axis * simd_dot(r, axis)
        e1 = simd_normalize(r); e2 = simd_cross(axis, e1)
        let zt = pencil ? 0.3 * (1 - min(pressure * 4, 1)) : 0.35 - pressure * pressDepth * length * abs(axis.z)
        ferrule = V3(target.x, target.y, zt) - axis * length
    }

    // advance bristle physics one substep; returns GPU bristle records (surface texel coords relative to org)
    func simulate(surface: Surface?, moved: Float, out: UnsafeMutablePointer<BrG>?, bbox: inout IRect) -> Int {
        var meanLag = V2.zero, nt = 0
        // free bristles converge to the tip; pressing flattens the belly and fans them out
        let pressFan = min(max((0.35 - (ferrule.z + axis.z * length)) / (length * abs(axis.z) * 0.5), 0), 1)
        let conv: Float = knife ? 0 : (round ? 0.82 : 0.15) * (1 - 0.8 * pressFan)
        for j in 0..<n {
            let o = offs[j] + (clump[j] - offs[j]) * clumpiness
            let R = ferrule + e1 * o.x + e2 * o.y
            root[j] = R
            let off3 = e1 * o.x + e2 * o.y
            let L = lens[j]
            let jt = dirj[j] + clumpJit[clumpOf[j]]
            let d = simd_normalize(axis * L - off3 * conv + (e1 * jt.x + e2 * jt.y) * L)
            let F = R + d * L
            if F.z >= 0 {
                tip[j] = V2(F.x, F.y); tipZ[j] = F.z; touching[j] = false; bendPt[j] = tip[j]; eq[j] = tip[j]
                continue
            }
            let c = min(-F.z / max(L * abs(d.z), 0.3), 1.6)
            var outw = V2(off3.x, off3.y)
            let ol = simd_length(outw)
            outw = ol > 1e-4 ? outw / ol : simd_normalize(dirj[j] + V2(1e-3, 0))
            let dxy = V2(d.x, d.y)
            let splay: Float = knife ? 0 : (round ? 0.45 : 0.25) * (ol / max(width * 0.5, 0.1))
            var dh = dxy + outw * splay * c
            if simd_length(dh) < 1e-3 { dh = outw }
            dh = simd_normalize(dh)
            let reach = max(L * simd_length(dxy), 0.88 * sqrt(max(L * L - R.z * R.z, 0)))
            // clumps and stray bristles wander sideways as they drag; edge bristles more
            let edge = min(ol / max(width * 0.5, 0.1), 1)
            let cw = noise1(travel * 0.45 + Float(clumpOf[j]) * 7.3, 1) - 0.5, tw = noise1(travel * 1.3 + Float(j) * 3.7, 2) - 0.5
            let wander = knife ? 0 : cw * (0.12 + 0.35 * edge * edge) * c + tw * 0.08
            let E = V2(R.x, R.y) + dh * reach + V2(-dh.y, dh.x) * wander
            eq[j] = E
            if !touching[j] { tip[j] = E; lastTip[j] = E } else {
                let dd = tip[j] - E
                let lag = 0.55 * min(c, 1) * L / stiffness
                let dl = simd_length(dd)
                if dl > lag { tip[j] = E + dd / dl * lag * 0.93 }
            }
            touching[j] = true; tipZ[j] = 0
            meanLag += tip[j] - E; nt += 1
        }
        // cohesion: bristles drag as clumps, clumps loosely as a bundle
        if nt > 0 {
            let m = meanLag / Float(nt)
            var cs = [V2](repeating: .zero, count: clumpLag.count), cn = [Float](repeating: 0, count: clumpLag.count)
            for j in 0..<n where touching[j] { cs[clumpOf[j]] += tip[j] - eq[j]; cn[clumpOf[j]] += 1 }
            for k in cs.indices where cn[k] > 0 { cs[k] = cs[k] / cn[k] * 0.6 + m * 0.4 }
            for j in 0..<n where touching[j] { tip[j] = eq[j] + simd_mix(tip[j] - eq[j], cs[clumpOf[j]], V2(repeating: 0.6)) }
        }
        // the part of each touching bristle that lies on the surface
        var fs: Float = 0
        for j in 0..<n where touching[j] {
            let L = lens[j]
            let toRoot = V2(root[j].x, root[j].y) - tip[j]
            let dl = simd_length(toRoot)
            let F = root[j] + simd_normalize(axis * L) * L
            let c = min(-F.z / max(L * abs(axis.z), 0.3), 1.6)
            lie[j] = min(L * min(max(c * (knife ? 2.5 : 1.1), 0.04), 0.85), dl)
            bendPt[j] = dl > 1e-5 ? tip[j] + toRoot / dl * lie[j] : tip[j]
            pressJ[j] = min(c * (knife ? 8 : 1.6), 1)
            fs += lie[j] + 2 * rads[j]
        }
        foot = nt > 0 ? fs / Float(nt) : 0
        travel += moved
        guard let s = surface, let out = out else { for j in 0..<n { lastTip[j] = tip[j] }; return 0 }
        var cnt = 0, sl: Float = 0, sn: Float = 0
        var bb = IRect(x0: Int.max, y0: Int.max, x1: Int.min, y1: Int.min)
        for j in 0..<n {
            let L = lens[j]
            var a: V2, b: V2, press: Float = 0, z: Float = 0, len: Float
            if touching[j] {
                a = bendPt[j]; b = tip[j]; press = pressJ[j]; len = lie[j]
            } else if tipZ[j] < 3 {
                a = tip[j] - V2(axis.x, axis.y) * 0.6; b = tip[j]; z = tipZ[j] / UNIT; len = 0.6
            } else { lastTip[j] = tip[j]; continue }
            let r = rads[j] * (1 + 0.45 * press)
            let mv = simd_length(tip[j] - lastTip[j])
            if touching[j] && mv > 1e-4 && len > 1e-3 {
                let u = (b - a) / max(simd_length(b - a), 1e-4), m = (tip[j] - lastTip[j]) / mv
                let sn2 = u.x * m.y - u.y * m.x; sl += sn2 * sn2; sn += 1
            }
            let rate = (max(mv, 0.25 * moved) + 0.002) / max(len + 2 * r, 0.25)
            lastTip[j] = tip[j]
            let ta = s.toTexel(a), tb = s.toTexel(b), rt = r * PPM
            out[cnt] = BrG(seg: V4(ta.x, ta.y, tb.x, tb.y), p0: V4(rt, press, z, rate), p1: V4(2 * rads[j] * L * PPM * PPM, Float(j), 0, 0))
            cnt += 1
            let R = Int(rt * 2.2) + 3
            bb.x0 = min(bb.x0, Int(min(ta.x, tb.x)) - R); bb.y0 = min(bb.y0, Int(min(ta.y, tb.y)) - R)
            bb.x1 = max(bb.x1, Int(max(ta.x, tb.x)) + R); bb.y1 = max(bb.y1, Int(max(ta.y, tb.y)) + R)
        }
        if sn > 0 { slant = sqrt(sl / sn) }       // rms: the most slanted bristles decide
        bbox = cnt > 0 ? bb : .none
        return cnt
    }

    func hover() { for j in 0..<n { touching[j] = false } }
}

// ---------------------------------------------------------------- tubes
final class Tube {
    let pigment: Int
    var left: Float = 1      // remaining fraction
    init(_ p: Int) { pigment = p }
    var name: String { PIGMENTS[pigment].name }
}

// ---------------------------------------------------------------- operations (history / replay)
struct Sample: Codable { var x: Float; var y: Float; var p: Float; var tx: Float; var ty: Float; var roll: Float; var t: Float; var s: Int8 }
enum Op: Codable {
    case stroke(brush: Int, samples: [Sample])
    case squeeze(tube: Int, samples: [Sample])
    case dry(hours: Float, palette: Bool)
    case clean(brush: Int)
    case scrape
    case wipe
    case crop(rect: [Float])
    case movePalette(x: Float, y: Float)
    var empty: Op {
        switch self {
        case .stroke(let b, _): return .stroke(brush: b, samples: [])
        case .squeeze(let t, _): return .squeeze(tube: t, samples: [])
        default: return self
        }
    }
}

final class UndoRec {
    var tiles: [(s: Int, tx: Int, ty: Int, buf: MTLBuffer)] = []
    var keys = Set<Int>()
    var brushes: [(b: Int, buf: MTLBuffer)] = []
    var tubes: [Float] = []
    var crop: V4 = .zero
    var pal: V2 = .zero
    var bytes: Int = 0
}

// ---------------------------------------------------------------- engine
final class Engine {
    let g = GPU.shared
    var surfaces: [Surface] = []
    var canvas: Surface { surfaces[0] }
    var palette: Surface { surfaces[1] }
    var brushes: [Brush] = []
    var tubes: [Tube] = []
    var active = 0                      // active brush
    let pigBuf: MTLBuffer
    let fpTex: MTLTexture, fpDepth: MTLTexture
    let fpRPS: MTLRenderPipelineState
    let fpDSS: MTLDepthStencilState
    var arena: [MTLBuffer] = []
    var arenaIdx = 0, arenaOff = 0
    let FPMAX = 640

    // sim tuning
    var alpha: Float = 0.7, rho: Float = 0.75, ratePass: Float = 0.75, mixPass: Float = 0.35, eps: Float = 3.5
    var feed: Float = 0.5, wick: Float = 0.004, groove: Float = 9, conform: Float = 1.1, coat: Float = 0.7
    var film: Float = 12, sinkRep: Float = 0.8
    var relief: Float = 1.0
    var medium = 0                      // 0 oil, 1 acrylic, 2 gouache
    var dullDry = true { didSet { for s in surfaces { s.dirty = IRect(x0: 0, y0: 0, x1: s.w, y1: s.h) } } }   // dried paint loses a little gloss

    // input
    var cur: (op: Op, done: Int)?
    var lastSample: Sample?
    var undoRec: UndoRec?
    var undoStack: [UndoRec] = [], redoStack: [UndoRec] = []
    var history: [Op] = []
    var redoOps: [Op] = []
    let undoBudget = 3_000_000_000
    var onChange: (() -> Void)?        // the painting was edited

    init() {
        let dev = g.dev
        let tbl = pigmentTable()
        pigBuf = dev.makeBuffer(bytes: tbl, length: tbl.count * 4, options: .storageModeShared)!
        fpTex = g.tex(.rgba32Float, FPMAX, FPMAX, rt: true)
        fpDepth = { let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: 640, height: 640, mipmapped: false)
            d.usage = .renderTarget; d.storageMode = .private; return dev.makeTexture(descriptor: d)! }()
        let rp = MTLRenderPipelineDescriptor()
        rp.vertexFunction = g.lib.makeFunction(name: "fpVert"); rp.fragmentFunction = g.lib.makeFunction(name: "fpFrag")
        rp.colorAttachments[0].pixelFormat = .rgba32Float; rp.depthAttachmentPixelFormat = .depth32Float
        fpRPS = try! dev.makeRenderPipelineState(descriptor: rp)
        let ds = MTLDepthStencilDescriptor(); ds.depthCompareFunction = .less; ds.isDepthWriteEnabled = true
        fpDSS = dev.makeDepthStencilState(descriptor: ds)!
        for _ in 0..<3 { arena.append(g.buffer(64 << 20)) }

        surfaces.append(Engine.makeCanvas(400, 300, texture: 0))
        let pal = Surface(id: 1, kind: 1, widthMM: 340, heightMM: 260, origin: Engine.paletteHome, z: -14, cornerMM: 34)
        pal.hole = V4(38 * PPM, 222 * PPM, 13 * PPM, 0)
        surfaces.append(pal)
        let specs: [(String, Bool, Float, Float, Float, Int, Float, V3)] = [
            ("Round 2", true, 2.2, 2.2, 9, 70, 0.14, V3(0.50, 0.08, 0.06)),
            ("Round 6", true, 4.6, 4.6, 15, 190, 0.17, V3(0.50, 0.08, 0.06)),
            ("Round 12", true, 8.2, 8.2, 23, 380, 0.21, V3(0.50, 0.08, 0.06)),
            ("Flat 4", false, 6.5, 1.8, 9.5, 170, 0.15, V3(0.05, 0.05, 0.05)),
            ("Flat 8", false, 11.5, 2.6, 14, 340, 0.18, V3(0.05, 0.05, 0.05)),
            ("Flat 14", false, 19, 3.6, 19, 640, 0.22, V3(0.05, 0.05, 0.05)),
        ]
        for (i, s) in specs.enumerated() {
            brushes.append(Brush(name: s.0, round: s.1, width: s.2, thick: s.3, length: s.4, n: s.5, tuftR: s.6, handle: s.7, seed: UInt64(1000 + i * 77)))
        }
        brushes.append(Brush(name: "Palette knife", round: false, width: 22, thick: 0.5, length: 32, n: 260, tuftR: 0.13, handle: V3(0.42, 0.26, 0.13), seed: 9999, knife: true))
        // detail brushes (appended so saved paintings keep their brush numbers)
        brushes.append(Brush(name: "Round 0", round: true, width: 0.9, thick: 0.9, length: 7, n: 30, tuftR: 0.07, handle: V3(0.50, 0.08, 0.06), seed: 1700))
        brushes.append(Brush(name: "Rigger 1", round: true, width: 1.2, thick: 1.2, length: 24, n: 50, tuftR: 0.07, handle: V3(0.12, 0.20, 0.12), seed: 1800))
        brushes.append(Brush(name: "Pencil HB", round: true, width: 0.6, thick: 0.6, length: 14, n: 1, tuftR: 0.3, handle: V3(0.06, 0.20, 0.12), seed: 1900, pencil: true))
        brushes[7].pressDepth = 0.35; brushes[8].pressDepth = 0.18; brushes[6].pressDepth = 0.1
        tubes = (0..<NPIG).map { Tube($0) }
        let cmd = g.q.makeCommandBuffer()!
        for s in surfaces { s.reset(cmd) }
        cmd.commit()
    }

    // canvases sit with their bottom-right corner at a fixed spot next to the palette and brushes
    static func makeCanvas(_ wmm: Float, _ hmm: Float, texture: Int) -> Surface {
        Surface(id: 0, kind: 0, widthMM: wmm, heightMM: hmm, origin: V2(200 - wmm, -150), z: 0, cornerMM: 0, texture: texture)
    }
    func newCanvas(_ cmd: MTLCommandBuffer, _ wmm: Float, _ hmm: Float, texture: Int) {
        let c = Engine.makeCanvas(wmm, hmm, texture: texture)
        c.reset(cmd)
        surfaces[0] = c
        placePalette(palette.origin); paletteStart = palette.origin
        history = []; redoOps = []; undoStack = []; redoStack = []
    }

    func resetAll(_ cmd: MTLCommandBuffer) {
        for s in surfaces { s.reset(cmd); s.mask = V4(0, 0, Float(s.w), Float(s.h)) }
        for b in brushes { b.clean(cmd) }
        for t in tubes { t.left = 1 }
        placePalette(paletteStart)
    }

    // ------------------------------------------------ arena for per-substep bristle data
    func beginFrame() { arenaIdx = (arenaIdx + 1) % arena.count; arenaOff = 0 }
    func alloc(_ bytes: Int) -> (MTLBuffer, Int)? {
        let a = (bytes + 255) & ~255
        if arenaOff + a > arena[arenaIdx].length { return nil }
        defer { arenaOff += a }
        return (arena[arenaIdx], arenaOff)
    }

    // ------------------------------------------------ ops
    func begin(_ op: Op) {
        let r = newRec()
        undoRec = r
        switch op {
        case .stroke(let b, _):
            snapBrush(r, b)
            brushes[b].reclump(seed: UInt64(history.count + 1) &* 7919 &+ UInt64(b))
        case .clean(let b): snapBrush(r, b)
        default: break
        }
        cur = (op, 0)
        lastSample = nil
    }
    func add(_ s: Sample) {
        guard var c = cur else { return }
        switch c.op {
        case .stroke(let b, var ss): ss.append(s); c.op = .stroke(brush: b, samples: ss)
        case .squeeze(let t, var ss): ss.append(s); c.op = .squeeze(tube: t, samples: ss)
        default: break
        }
        cur = c
    }
    func end(_ cmd: MTLCommandBuffer) {
        guard let c = cur else { return }
        process(cmd)
        if case .stroke(let b, _) = c.op {
            liftOff(cmd, brushes[b])
            if brushes[b].knife { brushes[b].clean(cmd) }     // wipe the knife on the rag
        }
        cur = nil
        commit(c.op)
    }
    var backlog: Int {      // samples of the current op not yet simulated
        guard let c = cur else { return 0 }
        switch c.op { case .stroke(_, let ss), .squeeze(_, let ss): return ss.count - c.done; default: return 0 }
    }
    func newRec() -> UndoRec {
        let r = UndoRec(); r.tubes = tubes.map { $0.left }; r.crop = canvas.mask; r.pal = palette.origin
        return r
    }
    func commit(_ op: Op) {
        if let r = undoRec {
            undoStack.append(r)
            redoStack.removeAll()
            trimUndo()
        }
        undoRec = nil
        history.append(op); redoOps.removeAll()
        onChange?()
    }
    func trimUndo() {
        var total = undoStack.reduce(0) { $0 + $1.bytes }
        while total > undoBudget && undoStack.count > 1 { total -= undoStack.removeFirst().bytes }
    }

    // process pending samples of the current op
    func process(_ cmd: MTLCommandBuffer) {
        flushSnaps(cmd)
        guard var c = cur else { return }
        switch c.op {
        case .stroke(let bi, let ss):
            while c.done < ss.count && arenaOff < 32 << 20 { strokeTo(cmd, brushes[bi], ss[c.done]); c.done += 1 }
        case .squeeze(let ti, let ss):
            while c.done < ss.count { squeezeTo(cmd, tubes[ti], ss[c.done]); c.done += 1 }
        default: break
        }
        cur = c
    }

    func surfaceFor(_ s: Sample) -> Surface? { s.s >= 0 && Int(s.s) < surfaces.count ? surfaces[Int(s.s)] : nil }

    // ------------------------------------------------ brush strokes
    let STEP: Float = 0.1   // minimum mm per substep
    let COARSE: Float = 0.45 // scales the adaptive substep (1 = fastest; smaller is finer and slower)
    func strokeTo(_ cmd: MTLCommandBuffer, _ b: Brush, _ s: Sample) {
        guard let l = lastSample else {
            lastSample = s
            b.hover()
            substep(cmd, b, s, moved: 0)
            return
        }
        let d = simd_length(V2(s.x - l.x, s.y - l.y))
        let dt = max(s.t - l.t, 0)
        // transfer is normalized by distance moved, so a substep only has to be short relative to the bristles'
        // contact (and sideways, relative to a bristle's thickness, so slanted bristles don't stamp)
        let step = b.pencil ? 0.4 : b.foot > 0 ? min(max(STEP, COARSE * min(0.25 * b.foot, 3 * b.tuftR / max(b.slant, 0.3))), 1.2 * COARSE) : 1.0
        var n = max(Int(ceil(d / step)), Int(ceil(dt / 0.008)), 1)
        n = min(n, 600)
        for k in 1...n {
            let f = Float(k) / Float(n)
            let m = Sample(x: l.x + (s.x - l.x) * f, y: l.y + (s.y - l.y) * f, p: l.p + (s.p - l.p) * f,
                           tx: l.tx + (s.tx - l.tx) * f, ty: l.ty + (s.ty - l.ty) * f, roll: l.roll + (s.roll - l.roll) * f,
                           t: l.t + (s.t - l.t) * f, s: f < 0.5 ? l.s : s.s)
            substep(cmd, b, m, moved: d / Float(n))
        }
        lastSample = s
    }

    func liftOff(_ cmd: MTLCommandBuffer, _ b: Brush) {
        b.hover()
        lastSample = nil
    }
    // flats turn so their width lies across the direction of travel, or along it (edgewise, for thin lines)
    static func followRoll(_ roll: Float, dir: V2, edge: Bool = false) -> Float {
        let target = atan2(dir.y, dir.x) + (edge ? 0 : .pi / 2)
        var d = (target - roll).truncatingRemainder(dividingBy: .pi)
        if d > .pi / 2 { d -= .pi } else if d < -.pi / 2 { d += .pi }
        return roll + d * 0.18
    }
    // a mouse has no pressure falloff: emulate the hand lifting while still moving (recorded like real samples)
    static func liftSamples(_ ss: [Sample]) -> [Sample] {
        guard let l = ss.last else { return [] }
        let back = ss.last(where: { l.t - $0.t > 0.03 }) ?? ss.first!
        let dt = max(l.t - back.t, 1e-3)
        var v = V2(l.x - back.x, l.y - back.y) / dt
        let sp = simd_length(v)
        if sp > 250 { v *= 250 / sp }
        var out = [Sample]()
        for k in 1...8 {
            let f = Float(k) / 8, t = f * 0.06
            var m = l
            m.x += v.x * t; m.y += v.y * t; m.t += t
            m.p = l.p * (1 - f) * (1 - f)
            out.append(m)
        }
        return out
    }

    func substep(_ cmd: MTLCommandBuffer, _ b: Brush, _ s: Sample, moved: Float) {
        let surf = surfaceFor(s)
        b.baseZ = surf?.z ?? -20
        b.pose(target: V2(s.x, s.y), pressure: s.p, tilt: V2(s.tx, s.ty), roll: s.roll)
        if b.pencil { pencilStep(cmd, b, surf, s, moved: moved); return }
        guard let (buf, off) = alloc(b.n * MemoryLayout<BrG>.stride) else { return }
        let ptr = (buf.contents() + off).bindMemory(to: BrG.self, capacity: b.n)
        var bb = IRect.none
        let cnt = b.simulate(surface: surf, moved: moved, out: ptr, bbox: &bb)
        guard let sf = surf, cnt > 0 else { return }
        // footprint region
        var org = I2(Int32(bb.x0), Int32(bb.y0))
        var w = bb.x1 - bb.x0, h = bb.y1 - bb.y0
        if w > FPMAX || h > FPMAX {
            let cx = (bb.x0 + bb.x1) / 2, cy = (bb.y0 + bb.y1) / 2
            w = min(w, FPMAX); h = min(h, FPMAX); org = I2(Int32(cx - w / 2), Int32(cy - h / 2))
        }
        for i in 0..<cnt {
            ptr[i].seg -= V4(Float(org.x), Float(org.y), Float(org.x), Float(org.y))
        }
        let region = IRect(x0: Int(org.x), y0: Int(org.y), x1: Int(org.x) + w, y1: Int(org.y) + h).clip(sf.w, sf.h)
        if region.empty { return }
        snapTiles(cmd, sf, region)
        sf.dirty = sf.dirty.union(IRect(x0: region.x0 - 2, y0: region.y0 - 2, x1: region.x1 + 2, y1: region.y1 + 2).clip(sf.w, sf.h))
        sf.markPainted(region)

        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = fpTex; rp.colorAttachments[0].loadAction = .clear
        rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: -1, alpha: 0); rp.colorAttachments[0].storeAction = .store
        rp.depthAttachment.texture = fpDepth; rp.depthAttachment.loadAction = .clear; rp.depthAttachment.clearDepth = 1; rp.depthAttachment.storeAction = .dontCare
        let re = cmd.makeRenderCommandEncoder(descriptor: rp)!
        re.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(w), height: Double(h), znear: 0, zfar: 1))
        re.setRenderPipelineState(fpRPS); re.setDepthStencilState(fpDSS)
        let gr: Float = b.knife ? 0 : (sf.kind == 0 ? groove : groove * 0.6)
        var fpp = FPP(vp: V2(Float(w), Float(h)), org: org, tsz: I2(Int32(sf.w), Int32(sf.h)), toothAmp: sf.toothAmp,
                      groove: gr, conform: b.knife ? 0.3 : conform, coat: b.knife ? 0.2 : coat)
        re.setVertexBuffer(buf, offset: off, index: 0)
        re.setVertexBytes(&fpp, length: MemoryLayout<FPP>.stride, index: 1)
        re.setFragmentBytes(&fpp, length: MemoryLayout<FPP>.stride, index: 1)
        re.setFragmentTexture(sf.tooth, index: 0)
        re.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: cnt)
        re.endEncoding()

        var sp = StepP(org: org, fpsz: I2(Int32(w), Int32(h)), surf: I2(Int32(sf.w), Int32(sf.h)), nb: Int32(b.n),
                       alpha: alpha, rho: b.knife ? rho * 0.3 : rho, ratePass: ratePass, mixPass: b.knife ? 0.7 : mixPass, mask: sf.mask, maskR: sf.maskR, toothAmp: sf.toothAmp, eps: eps, film: film, sinkRep: sinkRep, groove: gr, seed: b.knife ? 0.5 : Float(history.count % 997) * 0.731, pad1: b.knife ? 1 : 0, hole: sf.hole)
        let ce = cmd.makeComputeCommandEncoder()!
        ce.setTexture(fpTex, index: 0); ce.setTexture(sf.state, index: 1); ce.setTexture(sf.height, index: 2); ce.setTexture(sf.tooth, index: 3)
        ce.setBuffer(b.res, offset: 0, index: 0); ce.setBuffer(b.cap, offset: 0, index: 1); ce.setBuffer(b.dres, offset: 0, index: 2)
        ce.setBytes(&sp, length: MemoryLayout<StepP>.stride, index: 3)
        g.run(ce, "kTransfer", w, h)
        applyRes(ce, b, moved: moved)
        ce.endEncoding()
    }

    // graphite: the point sweeps a capsule from its last position, catching the tops of the weave under light pressure
    func pencilStep(_ cmd: MTLCommandBuffer, _ b: Brush, _ surf: Surface?, _ s: Sample, moved: Float) {
        let pt = V2(s.x, s.y), last = b.tip[0]
        b.tip[0] = pt
        guard let sf = surf, s.p > 0.01, moved > 0 else { return }
        let r = (0.28 + 0.12 * s.p) * PPM
        let a = sf.toTexel(last), c = sf.toTexel(pt)
        let R = Int(r) + 2
        let region = IRect(x0: Int(min(a.x, c.x)) - R, y0: Int(min(a.y, c.y)) - R, x1: Int(max(a.x, c.x)) + R, y1: Int(max(a.y, c.y)) + R).clip(sf.w, sf.h)
        if region.empty { return }
        snapTiles(cmd, sf, region)
        sf.dirty = sf.dirty.union(region)
        sf.markPainted(region)
        let L = simd_length(c - a)
        var q = PenP(org: I2(Int32(region.x0), Int32(region.y0)), sz: I2(Int32(region.x1 - region.x0), Int32(region.y1 - region.y0)),
                     surf: I2(Int32(sf.w), Int32(sf.h)), a: a, b: c, r: r, press: s.p, toothAmp: sf.toothAmp, share: L / (L + 2 * r),
                     mask: sf.mask, maskR: sf.maskR, seed: b.travel, pad: 0, hole: sf.hole)
        b.travel += moved
        let e = cmd.makeComputeCommandEncoder()!
        e.setTexture(sf.state, index: 0); e.setTexture(sf.tooth, index: 1)
        e.setBytes(&q, length: MemoryLayout<PenP>.stride, index: 0)
        g.run(e, "kPencil", region.x1 - region.x0, region.y1 - region.y0)
        e.endEncoding()
    }

    // fold accumulated deltas into bristle reservoirs, feed tufts from the belly
    func applyRes(_ ce: MTLComputeCommandEncoder, _ b: Brush, moved: Float) {
        var ap = ApplyP(nb: Int32(b.n), feed: 1 - pow(1 - feed, max(moved, 0.02)), wick: 1 - pow(1 - wick, max(moved, 0.02)), clean: 0, nz: Int32(b.nz))
        ce.setComputePipelineState(g.cp("kApply"))
        ce.setBuffer(b.res, offset: 0, index: 0); ce.setBuffer(b.dres, offset: 0, index: 1); ce.setBuffer(b.cap, offset: 0, index: 2)
        ce.setBytes(&ap, length: MemoryLayout<ApplyP>.stride, index: 3)
        ce.setBuffer(b.zoneBuf, offset: 0, index: 4); ce.setBuffer(b.flow, offset: 0, index: 5)
        ce.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    // ------------------------------------------------ tube squeeze
    func squeezeTo(_ cmd: MTLCommandBuffer, _ t: Tube, _ s: Sample) {
        let l = lastSample ?? s
        lastSample = s
        guard let sf = surfaceFor(s) else { return }      // tubes flatten as they're used but never run dry
        let d = simd_length(V2(s.x - l.x, s.y - l.y)), dt = max(s.t - l.t, 0)
        let n = max(Int(ceil(d / 0.25)), 1)
        let rad: Float = 2.0 * PPM                                   // nozzle radius in texels
        let rope: Float = 270                                        // rope height (units)
        // peak deposit per substep so consecutive gaussians sum to a rope of height `rope`, plus a slow pile-up when held still
        let amt = s.p * (rope * d * PPM / (rad * 1.144) + 500 * dt) / Float(n)
        let dir = d > 1e-4 ? simd_normalize(V2(s.x - l.x, s.y - l.y)) : V2(1, 0)
        var comp = [Float](repeating: 0, count: NCH); comp[t.pigment] = 1
        for k in 1...n {
            let f = Float(k) / Float(n)
            let c = sf.toTexel(V2(l.x + (s.x - l.x) * f, l.y + (s.y - l.y) * f))
            let R = Int(rad * 1.7)
            let region = IRect(x0: Int(c.x) - R, y0: Int(c.y) - R, x1: Int(c.x) + R, y1: Int(c.y) + R).clip(sf.w, sf.h)
            if region.empty { continue }
            snapTiles(cmd, sf, region)
            sf.dirty = sf.dirty.union(IRect(x0: region.x0 - 2, y0: region.y0 - 2, x1: region.x1 + 2, y1: region.y1 + 2).clip(sf.w, sf.h))
            sf.markPainted(region)
            t.left = max(0.12, t.left - amt * rad * rad * 1.31 * UNIT / (PPM * PPM) / 21000 * 2)
            var q = SqzP(org: I2(Int32(region.x0), Int32(region.y0)), sz: I2(Int32(region.x1 - region.x0), Int32(region.y1 - region.y0)),
                         surf: I2(Int32(sf.w), Int32(sf.h)), c: c, comp0: V4(comp[0], comp[1], comp[2], comp[3]),
                         comp1: V4(comp[4], comp[5], comp[6], comp[7]), comp2: V4(comp[8], comp[9], comp[10], comp[11]), comp3: V4(comp[12], comp[13], comp[14], comp[15]),
                         rad: rad, amt: amt, hmax: rope * 1.15, toothAmp: sf.toothAmp, mask: sf.mask, maskR: sf.maskR,
                         seed: Float(t.pigment) * 3.1, dir: dir, hole: sf.hole)
            let e = cmd.makeComputeCommandEncoder()!
            e.setTexture(sf.state, index: 0); e.setTexture(sf.height, index: 1); e.setTexture(sf.tooth, index: 2)
            e.setBytes(&q, length: MemoryLayout<SqzP>.stride, index: 0)
            g.run(e, "kSqueeze", region.x1 - region.x0, region.y1 - region.y0)
            e.endEncoding()
        }
    }

    // ------------------------------------------------ drying
    func dryParams(_ hours: Float, apply: Bool, _ s: Surface) -> DryP {
        let speed: Float = medium == 0 ? 1 : 60
        return DryP(size: I2(Int32(s.w), Int32(s.h)), hours: hours * speed, thick0: 25, shrink: medium == 1 ? 0.75 : 1, apply: apply ? 1 : 0)
    }
    func preview(_ cmd: MTLCommandBuffer, hours: Float, palette: Bool) {
        for s in surfaces where s.kind == 0 || palette { dryKernel(cmd, s, dryParams(hours, apply: false, s)) }
    }
    func dry(_ cmd: MTLCommandBuffer, hours: Float, palette: Bool) {
        let r = newRec()
        undoRec = r
        for s in surfaces where s.kind == 0 || palette {
            for ty in 0..<s.tilesY { for tx in 0..<s.tilesX where s.painted[ty * s.tilesX + tx] {
                snapTiles(cmd, s, IRect(x0: tx * TILE, y0: ty * TILE, x1: tx * TILE + 1, y1: ty * TILE + 1))
            } }
            dryKernel(cmd, s, dryParams(hours, apply: true, s))
            s.dirty = IRect(x0: 0, y0: 0, x1: s.w, y1: s.h)
        }
        commit(.dry(hours: hours, palette: palette))
    }
    func dryKernel(_ cmd: MTLCommandBuffer, _ s: Surface, _ p: DryP) {
        var p = p
        var ta = s.toothAmp
        let e = cmd.makeComputeCommandEncoder()!
        e.setTexture(s.state, index: 0); e.setTexture(s.height, index: 1); e.setTexture(s.tooth, index: 2); e.setTexture(s.prev, index: 3)
        e.setBuffer(pigBuf, offset: 0, index: 0); e.setBytes(&p, length: MemoryLayout<DryP>.stride, index: 1)
        e.setBytes(&ta, length: 4, index: 2)
        g.run(e, "kDry", s.w, s.h)
        e.endEncoding()
    }

    // ------------------------------------------------ misc ops
    func cleanBrush(_ cmd: MTLCommandBuffer, _ i: Int) {
        let r = newRec()
        snapBrush(r, i); flushSnaps(cmd)
        undoRec = r
        brushes[i].clean(cmd)
        commit(.clean(brush: i))
    }
    // scrape the palette clean / wipe the canvas back to fresh gesso
    func clearSurface(_ cmd: MTLCommandBuffer, palette pal: Bool) {
        let r = newRec()
        undoRec = r
        let s = pal ? palette : canvas
        for ty in 0..<s.tilesY { for tx in 0..<s.tilesX where s.painted[ty * s.tilesX + tx] {
            snapTiles(cmd, s, IRect(x0: tx * TILE, y0: ty * TILE, x1: tx * TILE + 1, y1: ty * TILE + 1))
        } }
        let m = s.mask
        s.reset(cmd); s.mask = m
        commit(pal ? .scrape : .wipe)
    }
    // the palette rests on the table, or is held just above whatever it is moved over
    static let paletteHome = V2(225, -140)
    var paletteStart = Engine.paletteHome       // palette position when the history began (replay starts there)
    func placePalette(_ o: V2) {
        let p = palette, c = canvas
        p.origin = simd_clamp(o, V2(repeating: -3000), V2(repeating: 3000))
        let pl = p.origin, ph = pl + V2(Float(p.w), Float(p.h)) / PPM
        let cl = c.toWorld(V2(c.mask.x, c.mask.y)), ch = c.toWorld(V2(c.mask.z, c.mask.w))
        // things it would rest on: the canvas, the brushes and the tube tray
        let under = [V4(cl.x, cl.y, ch.x, ch.y), V4(-200, -440, 90, -150), V4(238, -264, 512, -150)]
        let over = under.contains { pl.x < $0.z && ph.x > $0.x && pl.y < $0.w && ph.y > $0.y }
        p.z = over ? c.z + 7 : -14
    }
    func movePalette(_ o: V2) {
        undoRec = newRec()
        placePalette(o)
        commit(.movePalette(x: o.x, y: o.y))
    }
    func crop(_ rect: V4) {
        let r = newRec()
        undoRec = r
        canvas.mask = rect
        canvas.dirty = IRect(x0: 0, y0: 0, x1: canvas.w, y1: canvas.h)
        commit(.crop(rect: [rect.x, rect.y, rect.z, rect.w]))
    }

    // ------------------------------------------------ undo (tile snapshots)
    var tileBytes: Int { TILE * TILE * (8 * NSLICE + 4) }
    func snapTiles(_ cmd: MTLCommandBuffer, _ s: Surface, _ r: IRect) {
        guard let rec = undoRec else {
            for ty in (r.y0 / TILE)...((r.y1 - 1) / TILE) { for tx in (r.x0 / TILE)...((r.x1 - 1) / TILE) { s.packed[ty * s.tilesX + tx] = nil } }
            return
        }
        var todo: [(Int, Int)] = []
        for ty in (r.y0 / TILE)...((r.y1 - 1) / TILE) {
            for tx in (r.x0 / TILE)...((r.x1 - 1) / TILE) {
                let key = (s.id << 20) | (ty << 10) | tx
                s.packed[ty * s.tilesX + tx] = nil
                if !rec.keys.contains(key) { rec.keys.insert(key); todo.append((tx, ty)) }
            }
        }
        if todo.isEmpty { return }
        let bl = cmd.makeBlitCommandEncoder()!
        for (tx, ty) in todo {
            let buf = g.buffer(tileBytes, shared: false)
            copyTile(bl, s, tx, ty, buf, toBuffer: true)
            rec.tiles.append((s.id, tx, ty, buf)); rec.bytes += tileBytes
        }
        bl.endEncoding()
    }
    func copyTile(_ bl: MTLBlitCommandEncoder, _ s: Surface, _ tx: Int, _ ty: Int, _ buf: MTLBuffer, toBuffer: Bool, offset: Int = 0) {
        let o = MTLOrigin(x: tx * TILE, y: ty * TILE, z: 0)
        let sz = MTLSize(width: min(TILE, s.w - tx * TILE), height: min(TILE, s.h - ty * TILE), depth: 1)
        for sl in 0..<NSLICE {
            let off = offset + sl * TILE * TILE * 8
            if toBuffer {
                bl.copy(from: s.state, sourceSlice: sl, sourceLevel: 0, sourceOrigin: o, sourceSize: sz, to: buf, destinationOffset: off, destinationBytesPerRow: TILE * 8, destinationBytesPerImage: TILE * TILE * 8)
            } else {
                bl.copy(from: buf, sourceOffset: off, sourceBytesPerRow: TILE * 8, sourceBytesPerImage: TILE * TILE * 8, sourceSize: sz, to: s.state, destinationSlice: sl, destinationLevel: 0, destinationOrigin: o)
            }
        }
        let off = offset + NSLICE * TILE * TILE * 8
        if toBuffer {
            bl.copy(from: s.height, sourceSlice: 0, sourceLevel: 0, sourceOrigin: o, sourceSize: sz, to: buf, destinationOffset: off, destinationBytesPerRow: TILE * 4, destinationBytesPerImage: TILE * TILE * 4)
        } else {
            bl.copy(from: buf, sourceOffset: off, sourceBytesPerRow: TILE * 4, sourceBytesPerImage: TILE * TILE * 4, sourceSize: sz, to: s.height, destinationSlice: 0, destinationLevel: 0, destinationOrigin: o)
        }
    }
    // brush snapshots are copied on the GPU, in order with the strokes (begin() has no command buffer, so the copy waits for the next one)
    var snaps: [(MTLBuffer, MTLBuffer)] = []
    func snapBrush(_ r: UndoRec, _ i: Int) {
        let b = brushes[i], buf = g.buffer(b.res.length, shared: false)
        r.brushes.append((i, buf)); r.bytes += buf.length
        snaps.append((b.res, buf))
    }
    func flushSnaps(_ cmd: MTLCommandBuffer) {
        guard !snaps.isEmpty else { return }
        let e = cmd.makeBlitCommandEncoder()!
        for (a, b) in snaps { e.copy(from: a, sourceOffset: 0, to: b, destinationOffset: 0, size: a.length) }
        e.endEncoding(); snaps.removeAll()
    }
    // swap the recorded state with the current one (undo and redo are the same operation)
    func swap(_ cmd: MTLCommandBuffer, _ r: UndoRec) {
        if !r.tiles.isEmpty {
            var fresh: [MTLBuffer] = []
            let b1 = cmd.makeBlitCommandEncoder()!
            for t in r.tiles { let nb = g.buffer(tileBytes, shared: false); copyTile(b1, surfaces[t.s], t.tx, t.ty, nb, toBuffer: true); fresh.append(nb) }
            b1.endEncoding()
            let b2 = cmd.makeBlitCommandEncoder()!
            for t in r.tiles { copyTile(b2, surfaces[t.s], t.tx, t.ty, t.buf, toBuffer: false) }
            b2.endEncoding()
            for i in r.tiles.indices {
                r.tiles[i].buf = fresh[i]
                let t = r.tiles[i], s = surfaces[t.s]
                s.dirty = s.dirty.union(IRect(x0: t.tx * TILE - 2, y0: t.ty * TILE - 2, x1: t.tx * TILE + TILE + 2, y1: t.ty * TILE + TILE + 2).clip(s.w, s.h))
                s.painted[t.ty * s.tilesX + t.tx] = true; s.packed[t.ty * s.tilesX + t.tx] = nil
            }
        }
        flushSnaps(cmd)
        if !r.brushes.isEmpty {
            let e = cmd.makeBlitCommandEncoder()!
            for i in r.brushes.indices {
                let b = brushes[r.brushes[i].b], tmp = g.buffer(b.res.length, shared: false)
                e.copy(from: b.res, sourceOffset: 0, to: tmp, destinationOffset: 0, size: tmp.length)
                e.copy(from: r.brushes[i].buf, sourceOffset: 0, to: b.res, destinationOffset: 0, size: tmp.length)
                r.brushes[i].buf = tmp
            }
            e.endEncoding()
        }
        let t = tubes.map { $0.left }
        for (i, v) in r.tubes.enumerated() { tubes[i].left = v }
        r.tubes = t
        let c = canvas.mask; canvas.mask = r.crop; r.crop = c
        if c != canvas.mask { canvas.dirty = IRect(x0: 0, y0: 0, x1: canvas.w, y1: canvas.h) }
        let po = palette.origin; placePalette(r.pal); r.pal = po
    }
    func undo(_ cmd: MTLCommandBuffer) {
        guard cur == nil, let r = undoStack.popLast() else { return }
        swap(cmd, r)
        redoStack.append(r)
        if let op = history.popLast() { redoOps.append(op) }
        onChange?()
    }
    func redo(_ cmd: MTLCommandBuffer) {
        guard cur == nil, let r = redoStack.popLast() else { return }
        swap(cmd, r)
        undoStack.append(r)
        if let op = redoOps.popLast() { history.append(op) }
        onChange?()
    }

    // ------------------------------------------------ tile codec (lossless)
    // A tile's 44 half-float channels become 16-bit planes, plus two for the height, stored as the bit difference from the
    // height recomputed out of tooth and paint (nearly always tiny). Each plane is predicted from its neighbours (LOCO-I's
    // median predictor; the height residual is noise and skips it) and split into high and low byte planes before LZFSE,
    // so empty or smooth channels cost almost nothing. Bit-exact; about a quarter smaller than compressing the raw tile.
    static let NPLANE = NSLICE * 4 + 2
    @inline(__always) static func medPredict(_ q: UnsafeMutablePointer<UInt16>, _ i: Int) -> UInt16 {
        let x = i % TILE
        if i >= TILE && x > 0 {
            let a = Int(q[i - 1]), c = Int(q[i - TILE]), d = Int(q[i - TILE - 1])
            return UInt16(truncatingIfNeeded: max(min(a, c), min(max(a, c), a + c - d)))
        }
        return i == 0 ? 0 : q[x > 0 ? i - 1 : i - TILE]
    }
    static func heightGuess(_ u: UnsafePointer<UInt16>, _ tooth: UnsafePointer<Float16>, _ amp: Float, _ i: Int) -> UInt32 {
        let n = TILE * TILE
        func f(_ sl: Int, _ c: Int) -> Float { Float(Float16(bitPattern: u[(sl * n + i) * 4 + c])) }
        var paint: Float = 0
        for sl in 0..<8 { for c in 0..<4 { paint += f(sl, c) } }
        return (Float(tooth[i]) * amp + f(8, 2) + paint).bitPattern
    }
    // version 3 compresses with zlib (raw deflate, which browsers decode natively); 1-2 used LZFSE
    static func algo(_ version: Int) -> NSData.CompressionAlgorithm { version >= 3 ? .zlib : .lzfse }
    static func packTile(_ src: UnsafeRawPointer, _ tooth: UnsafePointer<Float16>, _ amp: Float) -> Data {
        let n = TILE * TILE, u = src.bindMemory(to: UInt16.self, capacity: NSLICE * n * 4 + 2 * n)
        let pl = UnsafeMutablePointer<UInt16>.allocate(capacity: NPLANE * n), out = UnsafeMutablePointer<UInt8>.allocate(capacity: NPLANE * n * 2)
        defer { pl.deallocate(); out.deallocate() }
        for sl in 0..<NSLICE { for c in 0..<4 { for i in 0..<n { pl[(sl * 4 + c) * n + i] = u[(sl * n + i) * 4 + c] } } }
        let hb = NSLICE * n * 4
        let hp = (src + 2 * hb).bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n {
            let r = hp[i] &- heightGuess(u, tooth, amp, i)
            pl[NSLICE * 4 * n + i] = UInt16(r >> 16); pl[(NSLICE * 4 + 1) * n + i] = UInt16(r & 0xFFFF)
        }
        for p in 0..<NPLANE {
            let q = pl + p * n, o = out + 2 * p * n
            for i in 0..<n { let d = q[i] &- (p < NSLICE * 4 ? medPredict(q, i) : 0); o[i] = UInt8(d >> 8); o[n + i] = UInt8(d & 255) }
        }
        return (try? (Data(bytesNoCopy: out, count: NPLANE * n * 2, deallocator: .none) as NSData).compressed(using: .zlib) as Data) ?? Data()
    }
    static func unpackTile(_ d: Data, _ dst: UnsafeMutableRawPointer, _ tooth: UnsafePointer<Float16>, _ amp: Float, version: Int) -> Bool {
        let n = TILE * TILE
        guard let raw = try? (d as NSData).decompressed(using: algo(version)) as Data, raw.count == NPLANE * n * 2 else { return false }
        let pl = UnsafeMutablePointer<UInt16>.allocate(capacity: NPLANE * n); defer { pl.deallocate() }
        raw.withUnsafeBytes { (r: UnsafeRawBufferPointer) in
            for p in 0..<NPLANE {
                let q = pl + p * n, o = 2 * p * n
                for i in 0..<n { q[i] = (p < NSLICE * 4 ? medPredict(q, i) : 0) &+ (UInt16(r[o + i]) << 8 | UInt16(r[o + n + i])) }
            }
        }
        let u = dst.bindMemory(to: UInt16.self, capacity: NSLICE * n * 4 + 2 * n), hb = NSLICE * n * 4
        for sl in 0..<NSLICE { for c in 0..<4 { for i in 0..<n { u[(sl * n + i) * 4 + c] = pl[(sl * 4 + c) * n + i] } } }
        let hp = (dst + 2 * hb).bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n { hp[i] = heightGuess(u, tooth, amp, i) &+ (UInt32(pl[NSLICE * 4 * n + i]) << 16 | UInt32(pl[(NSLICE * 4 + 1) * n + i])) }
        return true
    }
    // history as a flat list of floats: op tag, its parameters, and for strokes/squeezes the sample count and samples
    static func packHistory(_ ops: [Op]) -> Data {
        var f = [Float]()
        for op in ops {
            switch op {
            case .stroke(let b, let ss), .squeeze(let b, let ss):
                if case .stroke = op { f.append(0) } else { f.append(1) }
                f.append(Float(b)); f.append(Float(ss.count))
                for m in ss { f += [m.x, m.y, m.p, m.tx, m.ty, m.roll, m.t, Float(m.s)] }
            case .dry(let h, let p): f += [2, h, p ? 1 : 0]
            case .clean(let b): f += [3, Float(b)]
            case .scrape: f.append(4)
            case .wipe: f.append(5)
            case .crop(let r): f += [6] + r
            case .movePalette(let x, let y): f += [7, x, y]
            }
        }
        return f.withUnsafeBytes { (try? (Data($0) as NSData).compressed(using: .zlib) as Data) ?? Data() }
    }
    static func unpackHistory(_ d: Data, version: Int) -> [Op]? {
        guard let raw = try? (d as NSData).decompressed(using: algo(version)) as Data, raw.count % 4 == 0 else { return nil }
        let f = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        var ops = [Op](), i = 0
        func take(_ k: Int) -> ArraySlice<Float>? { guard i + k <= f.count else { return nil }; defer { i += k }; return f[i..<(i + k)] }
        while i < f.count {
            guard let tag = take(1)?.first else { return nil }
            switch tag {
            case 0, 1:
                guard let h = take(2), let b = h.first, let c = h.last, c >= 0, let v = take(Int(c) * 8) else { return nil }
                let ss = stride(from: v.startIndex, to: v.endIndex, by: 8).map { Sample(x: f[$0], y: f[$0 + 1], p: f[$0 + 2], tx: f[$0 + 3], ty: f[$0 + 4], roll: f[$0 + 5], t: f[$0 + 6], s: Int8(f[$0 + 7])) }
                ops.append(tag == 0 ? .stroke(brush: Int(b), samples: ss) : .squeeze(tube: Int(b), samples: ss))
            case 2: guard let v = take(2) else { return nil }; ops.append(.dry(hours: v.first!, palette: v.last! > 0))
            case 3: guard let v = take(1) else { return nil }; ops.append(.clean(brush: Int(v.first!)))
            case 4: ops.append(.scrape)
            case 5: ops.append(.wipe)
            case 6: guard let v = take(4) else { return nil }; ops.append(.crop(rect: Array(v)))
            case 7: guard let v = take(2) else { return nil }; ops.append(.movePalette(x: v.first!, y: v.last!))
            default: return nil
            }
        }
        return ops
    }

    // ------------------------------------------------ save / load
    // file: "PAINTSIM", then length-prefixed blocks: JSON header, canvas tiles, palette tiles, brush reservoirs, and
    // (version 2+) the packed history. Version 3 compresses with zlib, 2 with LZFSE; version 1 files (raw LZFSE tiles,
    // history inside the JSON) still load.
    struct Header: Codable { var version = 3; var history: [Op]; var tubes: [Float]; var crop: [Float]; var medium: Int; var canvas: [Float]?; var palette: [Float]? }
    // the (regenerated) tooth of these tiles, TILE×TILE halves each
    func toothTiles(_ s: Surface, _ tiles: [Int]) -> MTLBuffer {
        let buf = g.buffer(tiles.count * TILE * TILE * 2)
        let cmd = g.q.makeCommandBuffer()!, bl = cmd.makeBlitCommandEncoder()!
        for (k, t) in tiles.enumerated() {
            let tx = t % s.tilesX, ty = t / s.tilesX
            bl.copy(from: s.tooth, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: tx * TILE, y: ty * TILE, z: 0),
                    sourceSize: MTLSize(width: min(TILE, s.w - tx * TILE), height: min(TILE, s.h - ty * TILE), depth: 1),
                    to: buf, destinationOffset: k * TILE * TILE * 2, destinationBytesPerRow: TILE * 2, destinationBytesPerImage: TILE * TILE * 2)
        }
        bl.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        return buf
    }
    // painted tiles only (the rest is regenerated); tiles unchanged since the last save or load reuse their encoding
    func surfaceData(_ s: Surface) -> Data {
        let tiles = (0..<s.painted.count).filter { s.painted[$0] }
        guard !tiles.isEmpty else { return Data(count: 4) }
        let todo = tiles.filter { s.packed[$0] == nil }
        for c in stride(from: 0, to: todo.count, by: 256) {
            let chunk = Array(todo[c..<min(c + 256, todo.count)])
            let buf = g.buffer(chunk.count * tileBytes)
            let cmd = g.q.makeCommandBuffer()!, bl = cmd.makeBlitCommandEncoder()!
            for (k, t) in chunk.enumerated() { copyTile(bl, s, t % s.tilesX, t / s.tilesX, buf, toBuffer: true, offset: k * tileBytes) }
            bl.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
            var packed = [Data](repeating: Data(), count: chunk.count)
            let src = buf.contents(), tb = tileBytes, amp = s.toothAmp
            let th = toothTiles(s, chunk).contents().bindMemory(to: Float16.self, capacity: chunk.count * TILE * TILE)
            packed.withUnsafeMutableBufferPointer { pk in
                let pkp = pk
                DispatchQueue.concurrentPerform(iterations: chunk.count) { k in pkp[k] = Engine.packTile(src + k * tb, th + k * TILE * TILE, amp) }
            }
            for (k, t) in chunk.enumerated() { s.packed[t] = packed[k] }
        }
        var out = Data()
        var n = UInt32(tiles.count); out.append(Data(bytes: &n, count: 4))
        for t in tiles {
            let d = s.packed[t]!
            var ti = UInt32(t), ln = UInt32(d.count)
            out.append(Data(bytes: &ti, count: 4)); out.append(Data(bytes: &ln, count: 4)); out.append(d)
        }
        return out
    }
    func loadSurface(_ s: Surface, _ d: Data, version: Int) -> Bool {
        let cmd0 = g.q.makeCommandBuffer()!
        let m = s.mask; s.reset(cmd0); s.mask = m
        cmd0.commit(); cmd0.waitUntilCompleted()
        func u32(_ o: Int) -> Int { Int(d.subdata(in: o..<(o + 4)).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }) }
        guard d.count >= 4 else { return false }
        let n = u32(0)
        var o = 4
        var entries = [(Int, Data)]()
        for _ in 0..<n {
            guard o + 8 <= d.count else { return false }
            let t = u32(o), ln = u32(o + 4); o += 8
            guard o + ln <= d.count, t < s.painted.count else { return false }
            entries.append((t, d.subdata(in: o..<(o + ln)))); o += ln
        }
        let tb = tileBytes
        for c in stride(from: 0, to: entries.count, by: 256) {
            let chunk = Array(entries[c..<min(c + 256, entries.count)])
            let buf = g.buffer(chunk.count * tb), dst = buf.contents(), amp = s.toothAmp
            let th = toothTiles(s, chunk.map { $0.0 }).contents().bindMemory(to: Float16.self, capacity: chunk.count * TILE * TILE)
            var ok = [Bool](repeating: false, count: chunk.count)
            ok.withUnsafeMutableBufferPointer { okb in
                let okp = okb
                DispatchQueue.concurrentPerform(iterations: chunk.count) { k in
                    if version >= 2 { okp[k] = Engine.unpackTile(chunk[k].1, dst + k * tb, th + k * TILE * TILE, amp, version: version); return }
                    guard let raw = try? (chunk[k].1 as NSData).decompressed(using: .lzfse) as Data, raw.count == tb else { return }
                    raw.withUnsafeBytes { _ = memcpy(dst + k * tb, $0.baseAddress!, tb) }
                    okp[k] = true
                }
            }
            guard !ok.contains(false) else { return false }
            let cmd = g.q.makeCommandBuffer()!, bl = cmd.makeBlitCommandEncoder()!
            for (k, e) in chunk.enumerated() {
                copyTile(bl, s, e.0 % s.tilesX, e.0 / s.tilesX, buf, toBuffer: false, offset: k * tb)
                s.painted[e.0] = true
                if version >= 3 { s.packed[e.0] = e.1 }     // older blocks are re-encoded on the next save
            }
            bl.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        }
        s.dirty = IRect(x0: 0, y0: 0, x1: s.w, y1: s.h)
        return true
    }
    // wait for the GPU to finish what's in flight, so brush reservoirs can be read or written on the CPU
    func idle() { let c = g.q.makeCommandBuffer()!; flushSnaps(c); c.commit(); c.waitUntilCompleted() }
    func encode() throws -> Data {
        idle()
        let h = Header(history: [], tubes: tubes.map { $0.left }, crop: [canvas.mask.x, canvas.mask.y, canvas.mask.z, canvas.mask.w], medium: medium,
                       canvas: [Float(canvas.w) / PPM, Float(canvas.h) / PPM, Float(canvas.texture)],
                       palette: [palette.origin.x, palette.origin.y, paletteStart.x, paletteStart.y])
        var br = Data()
        for b in brushes { br.append(Data(bytes: b.res.contents(), count: b.res.length)) }
        let blocks = [try JSONEncoder().encode(h), surfaceData(canvas), surfaceData(palette),
                      try (br as NSData).compressed(using: .zlib) as Data, Engine.packHistory(history)]
        var out = Data("PAINTSIM".utf8)
        out.reserveCapacity(blocks.reduce(16) { $0 + $1.count + 8 })
        for b in blocks { var n = UInt64(b.count); out.append(Data(bytes: &n, count: 8)); out.append(b) }
        return out
    }
    func save(to url: URL) throws { try encode().write(to: url, options: .atomic) }
    func load(from url: URL) throws { try decode(try Data(contentsOf: url)) }
    func decode(_ d: Data) throws {
        guard d.count > 16, String(data: d.prefix(8), encoding: .utf8) == "PAINTSIM" else { throw CocoaError(.fileReadCorruptFile) }
        var blocks = [Data](), o = 8
        while o + 8 <= d.count {
            let n = Int(d.subdata(in: o..<(o + 8)).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }); o += 8
            guard n >= 0, o + n <= d.count else { throw CocoaError(.fileReadCorruptFile) }
            blocks.append(d.subdata(in: o..<(o + n))); o += n
        }
        guard blocks.count >= 4 else { throw CocoaError(.fileReadCorruptFile) }
        idle()
        let h = try JSONDecoder().decode(Header.self, from: blocks[0])
        let v = h.version
        var ops = h.history
        if v >= 2 {
            guard blocks.count == 5, let o = Engine.unpackHistory(blocks[4], version: v), let br = try? (blocks[3] as NSData).decompressed(using: Engine.algo(v)) as Data else { throw CocoaError(.fileReadCorruptFile) }
            ops = o; blocks[3] = br
        }
        if let c = h.canvas, c.count == 3 { surfaces[0] = Engine.makeCanvas(c[0], c[1], texture: Int(c[2])) }
        guard loadSurface(canvas, blocks[1], version: v), loadSurface(palette, blocks[2], version: v) else { throw CocoaError(.fileReadCorruptFile) }
        var off = 0
        for b in brushes where off + b.res.length <= blocks[3].count {
            blocks[3].withUnsafeBytes { _ = memcpy(b.res.contents(), $0.baseAddress! + off, b.res.length) }
            off += b.res.length
        }
        for (i, v) in h.tubes.enumerated() where i < tubes.count { tubes[i].left = v }
        if h.crop.count == 4 { canvas.mask = V4(h.crop[0], h.crop[1], h.crop[2], h.crop[3]) }
        let pp = h.palette ?? [], home = Engine.paletteHome
        placePalette(pp.count == 4 ? V2(pp[0], pp[1]) : home); paletteStart = pp.count == 4 ? V2(pp[2], pp[3]) : home
        medium = h.medium
        history = ops; redoOps = []; undoStack = []; redoStack = []
        cur = nil; lastSample = nil; undoRec = nil
    }

    // test helper: v1 vs v2 tile sizes, and that v2 round-trips bit-exact
    func saveReport() {
        let s = canvas, tiles = (0..<s.painted.count).filter { s.painted[$0] }
        let buf = g.buffer(tiles.count * tileBytes)
        let cmd = g.q.makeCommandBuffer()!, bl = cmd.makeBlitCommandEncoder()!
        for (k, t) in tiles.enumerated() { copyTile(bl, s, t % s.tilesX, t / s.tilesX, buf, toBuffer: true, offset: k * tileBytes) }
        bl.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        var v1 = 0, v2 = 0, exact = true
        let th = toothTiles(s, tiles).contents().bindMemory(to: Float16.self, capacity: tiles.count * TILE * TILE)
        let back = UnsafeMutableRawPointer.allocate(byteCount: tileBytes, alignment: 16); defer { back.deallocate() }
        for k in 0..<tiles.count {
            let src = buf.contents() + k * tileBytes
            v1 += ((try? (Data(bytes: src, count: tileBytes) as NSData).compressed(using: .lzfse)) as Data?)?.count ?? 0
            let d = Engine.packTile(src, th + k * TILE * TILE, s.toothAmp); v2 += d.count
            memset(back, 0xAB, tileBytes)
            exact = exact && Engine.unpackTile(d, back, th + k * TILE * TILE, s.toothAmp, version: 3) && memcmp(src, back, tileBytes) == 0
        }
        print("tiles v1 \(v1 >> 20) MB, v2 \(v2 >> 20) MB, bit-exact \(exact); history v1 json \((try! JSONEncoder().encode(history)).count >> 10) KB, v2 \(Engine.packHistory(history).count >> 10) KB")
    }

    // ------------------------------------------------ debug readback
    func brushLoad(_ b: Brush) -> (tufts: Float, belly: Float) {
        let p = b.res.contents().bindMemory(to: Float.self, capacity: (b.n + b.nz) * NCH)
        var t: Float = 0, bl: Float = 0
        for j in 0..<b.n { for k in 0..<NCH { t += p[j * NCH + k] } }
        for j in b.n..<(b.n + b.nz) { for k in 0..<NCH { bl += p[j * NCH + k] } }
        return (t, bl)
    }
    func readHeights(_ s: Surface, _ x0: Int, _ y0: Int, _ w: Int, _ h: Int) -> [Float] {
        let buf = g.buffer(w * h * 4)
        let cmd = g.q.makeCommandBuffer()!, bl = cmd.makeBlitCommandEncoder()!
        bl.copy(from: s.height, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: x0, y: y0, z: 0), sourceSize: MTLSize(width: w, height: h, depth: 1),
                to: buf, destinationOffset: 0, destinationBytesPerRow: w * 4, destinationBytesPerImage: w * h * 4)
        bl.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        return Array(UnsafeBufferPointer(start: buf.contents().bindMemory(to: Float.self, capacity: w * h), count: w * h))
    }

    // ------------------------------------------------ display caches
    func shade(_ cmd: MTLCommandBuffer) {
        for s in surfaces where !s.dirty.empty {
            let r = s.dirty
            var sp = ShadeP(org: I2(Int32(r.x0), Int32(r.y0)), sz: I2(Int32(r.x1 - r.x0), Int32(r.y1 - r.y0)), surf: I2(Int32(s.w), Int32(s.h)),
                            relief: relief, k2: medium == 2 ? 0.2 : 0.45, mask: s.mask, maskR: s.maskR,
                            glossPaint: medium == 0 ? 1 : (medium == 1 ? 0.75 : 0.25), glossGround: s.kind == 0 ? 0.05 : 0.45, dryGloss: dullDry ? 0.85 : 1)
            let e = cmd.makeComputeCommandEncoder()!
            e.setTexture(s.state, index: 0); e.setTexture(s.height, index: 1); e.setTexture(s.color, index: 2); e.setTexture(s.nrm, index: 3)
            e.setBuffer(pigBuf, offset: 0, index: 0); e.setBytes(&sp, length: MemoryLayout<ShadeP>.stride, index: 1)
            g.run(e, "kShade", r.x1 - r.x0, r.y1 - r.y0)
            e.endEncoding()
            downsample(cmd, s, r)
            s.dirty = .none
        }
        for b in brushes { brushColors(cmd, b) }
    }
    // rebuild the mip chain of the display caches only where they changed
    func downsample(_ cmd: MTLCommandBuffer, _ s: Surface, _ r: IRect) {
        let e = cmd.makeComputeCommandEncoder()!
        var rr = r
        for l in 1..<s.color.mipmapLevelCount {
            rr = IRect(x0: rr.x0 >> 1, y0: rr.y0 >> 1, x1: (rr.x1 + 1) >> 1, y1: (rr.y1 + 1) >> 1)
            var R = SIMD4<Int32>(Int32(rr.x0), Int32(rr.y0), Int32(rr.x1), Int32(rr.y1))
            e.setBytes(&R, length: 16, index: 0)
            for t in [s.colorLevels, s.nrmLevels] {
                e.setTexture(t[l - 1], index: 0); e.setTexture(t[l], index: 1)
                g.run(e, "kDown", rr.x1 - rr.x0, rr.y1 - rr.y0)
            }
        }
        e.endEncoding()
    }
    func brushColors(_ cmd: MTLCommandBuffer, _ b: Brush) {
        var nb = Int32(b.n)
        let e = cmd.makeComputeCommandEncoder()!
        e.setBuffer(b.res, offset: 0, index: 0); e.setBuffer(b.cap, offset: 0, index: 1); e.setBuffer(b.colors, offset: 0, index: 2)
        e.setBuffer(pigBuf, offset: 0, index: 3); e.setBytes(&nb, length: 4, index: 4)
        g.run(e, "kBrushColor", b.n + 1)
        e.endEncoding()
    }
}
