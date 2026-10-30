// Scene: desk, canvas, palette, tube tray, brushes. Meshes are generated procedurally.
import Metal
import simd
import CoreGraphics
import CoreText
import Foundation

struct Vtx { var p: V4; var n: V4 }
struct Frame { var vp: float4x4; var lvp: float4x4; var cam: V4; var ldir: V4; var lcol: V4; var amb: V4; var params: V4 }
struct SurfDraw { var orgSize: V4; var mask: V4; var misc: V4; var heat: V4 }
struct ObjDraw { var model: float4x4; var base: V4; var mat: V4; var extra: V4 }
struct Hair { var p0: V4; var p1: V4; var p2: V4; var p3: V4 }

let TABLE_Z: Float = -20
// order of the brushes on the desk (left to right), also the 1-9, 0 keys
let BRUSH_ORDER = [8, 7, 0, 1, 2, 3, 4, 5, 6, 9]

// ---------------------------------------------------------------- math
func perspective(_ fovy: Float, _ aspect: Float, _ near: Float, _ far: Float) -> float4x4 {
    let y = 1 / tan(fovy * 0.5), x = y / aspect, z = far / (near - far)
    return float4x4(columns: (V4(x, 0, 0, 0), V4(0, y, 0, 0), V4(0, 0, z, -1), V4(0, 0, z * near, 0)))
}
func ortho(_ l: Float, _ r: Float, _ b: Float, _ t: Float, _ n: Float, _ f: Float) -> float4x4 {
    float4x4(columns: (V4(2 / (r - l), 0, 0, 0), V4(0, 2 / (t - b), 0, 0), V4(0, 0, -1 / (f - n), 0), V4(-(r + l) / (r - l), -(t + b) / (t - b), -n / (f - n), 1)))
}
func lookAt(_ eye: V3, _ c: V3, _ up: V3) -> float4x4 {
    let f = simd_normalize(c - eye), s = simd_normalize(simd_cross(f, up)), u = simd_cross(s, f)
    return float4x4(columns: (V4(s.x, u.x, -f.x, 0), V4(s.y, u.y, -f.y, 0), V4(s.z, u.z, -f.z, 0), V4(-simd_dot(s, eye), -simd_dot(u, eye), simd_dot(f, eye), 1)))
}
func frame(_ x: V3, _ y: V3, _ z: V3, _ o: V3) -> float4x4 { float4x4(columns: (V4(x, 0), V4(y, 0), V4(z, 0), V4(o, 1))) }
func translate(_ t: V3) -> float4x4 { frame(V3(1, 0, 0), V3(0, 1, 0), V3(0, 0, 1), t) }
func scale(_ s: V3) -> float4x4 { float4x4(diagonal: V4(s, 1)) }
func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float { let t = min(max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t) }

// ---------------------------------------------------------------- meshes
final class Mesh {
    let buf: MTLBuffer, count: Int
    init(_ v: [Vtx]) { count = v.count; buf = GPU.shared.dev.makeBuffer(bytes: v, length: max(v.count, 1) * MemoryLayout<Vtx>.stride, options: .storageModeShared)! }
}

// parametric surface (u around/across, v along) -> triangle list with finite-difference normals
func sweep(_ nu: Int, _ nv: Int, flip: Bool = false, _ f: (Float, Float) -> V3) -> [Vtx] {
    var grid = [[Vtx]]()
    for j in 0...nv {
        var row = [Vtx]()
        for i in 0...nu {
            let u = Float(i) / Float(nu), v = Float(j) / Float(nv)
            let p = f(u, v)
            let e: Float = 1e-3
            let du = f(u + e, v) - f(u - e, v)
            var dv = f(u, min(v + e, 1)) - f(u, max(v - e, 0))
            if simd_length(dv) < 1e-7 { dv = f(u, min(v + 0.02, 1)) - f(u, max(v - 0.02, 0)) }
            var n = simd_cross(du, dv)
            n = simd_length(n) > 1e-9 ? simd_normalize(n) : V3(0, 0, 1)
            if flip { n = -n }
            row.append(Vtx(p: V4(p, u), n: V4(n, v)))
        }
        grid.append(row)
    }
    var out = [Vtx]()
    for j in 0..<nv { for i in 0..<nu {
        let a = grid[j][i], b = grid[j][i + 1], c = grid[j + 1][i], d = grid[j + 1][i + 1]
        out += flip ? [a, c, b, b, c, d] : [a, b, c, b, d, c]
    } }
    return out
}
func lathe(_ nu: Int, _ prof: [(Float, Float)]) -> [Vtx] {     // (radius, z) pairs
    sweep(nu, prof.count - 1) { u, v in
        let fi = v * Float(prof.count - 1), i = min(Int(fi), prof.count - 2), t = fi - Float(i)
        let r = prof[i].0 + (prof[i + 1].0 - prof[i].0) * t, z = prof[i].1 + (prof[i + 1].1 - prof[i].1) * t
        let a = u * 2 * .pi
        return V3(cos(a) * r, sin(a) * r, z)
    }
}
func disk(_ nu: Int, _ r: Float, _ z: Float, up: Bool) -> [Vtx] {
    var out = [Vtx]()
    let n = V4(0, 0, up ? 1 : -1, 0)
    for i in 0..<nu {
        let a0 = Float(i) / Float(nu) * 2 * .pi, a1 = Float(i + 1) / Float(nu) * 2 * .pi
        let c = Vtx(p: V4(0, 0, z, 0), n: n), p0 = Vtx(p: V4(cos(a0) * r, sin(a0) * r, z, 0), n: n), p1 = Vtx(p: V4(cos(a1) * r, sin(a1) * r, z, 0), n: n)
        out += up ? [c, p0, p1] : [c, p1, p0]
    }
    return out
}
func box(_ lo: V3, _ hi: V3) -> [Vtx] {
    var out = [Vtx]()
    func quad(_ a: V3, _ b: V3, _ c: V3, _ d: V3, _ n: V3, _ uv: V2) {
        let va = Vtx(p: V4(a, 0), n: V4(n, 0)), vb = Vtx(p: V4(b, uv.x), n: V4(n, 0)), vc = Vtx(p: V4(c, uv.x), n: V4(n, uv.y)), vd = Vtx(p: V4(d, 0), n: V4(n, uv.y))
        out += [va, vb, vc, va, vc, vd]
    }
    let (x0, y0, z0, x1, y1, z1) = (lo.x, lo.y, lo.z, hi.x, hi.y, hi.z)
    quad(V3(x0, y0, z1), V3(x1, y0, z1), V3(x1, y1, z1), V3(x0, y1, z1), V3(0, 0, 1), V2(x1 - x0, y1 - y0))
    quad(V3(x0, y0, z0), V3(x0, y1, z0), V3(x1, y1, z0), V3(x1, y0, z0), V3(0, 0, -1), V2(y1 - y0, x1 - x0))
    quad(V3(x0, y0, z0), V3(x1, y0, z0), V3(x1, y0, z1), V3(x0, y0, z1), V3(0, -1, 0), V2(x1 - x0, z1 - z0))
    quad(V3(x1, y1, z0), V3(x0, y1, z0), V3(x0, y1, z1), V3(x1, y1, z1), V3(0, 1, 0), V2(x1 - x0, z1 - z0))
    quad(V3(x0, y1, z0), V3(x0, y0, z0), V3(x0, y0, z1), V3(x0, y1, z1), V3(-1, 0, 0), V2(y1 - y0, z1 - z0))
    quad(V3(x1, y0, z0), V3(x1, y1, z0), V3(x1, y1, z1), V3(x1, y0, z1), V3(1, 0, 0), V2(y1 - y0, z1 - z0))
    return out
}
func roundedOutline(_ x0: Float, _ y0: Float, _ x1: Float, _ y1: Float, _ r: Float) -> [V2] {
    var pts = [V2]()
    let cs = [(V2(x1 - r, y0 + r), -Float.pi / 2), (V2(x1 - r, y1 - r), 0), (V2(x0 + r, y1 - r), Float.pi / 2), (V2(x0 + r, y0 + r), Float.pi)]
    for (c, a0) in cs { for k in 0...10 { let a = a0 + Float(k) / 10 * .pi / 2; pts.append(c + V2(cos(a), sin(a)) * r) } }
    pts.append(pts[0])
    return pts
}
// side walls of a rounded rectangle slab
func roundedWall(_ x0: Float, _ y0: Float, _ x1: Float, _ y1: Float, _ r: Float, _ z0: Float, _ z1: Float) -> [Vtx] {
    let pts = roundedOutline(x0, y0, x1, y1, r)
    var out = [Vtx]()
    var acc: Float = 0
    for i in 0..<(pts.count - 1) {
        let a = pts[i], b = pts[i + 1]
        let d = simd_normalize(b - a), n = V3(d.y, -d.x, 0)
        let l = simd_length(b - a)
        let va = Vtx(p: V4(a.x, a.y, z0, acc), n: V4(n, 0)), vb = Vtx(p: V4(b.x, b.y, z0, acc + l), n: V4(n, 0))
        let vc = Vtx(p: V4(b.x, b.y, z1, acc + l), n: V4(n, 1)), vd = Vtx(p: V4(a.x, a.y, z1, acc), n: V4(n, 1))
        out += [va, vb, vc, va, vc, vd]
        acc += l
    }
    return out
}

// brush handle + ferrule in brush local space: z up along the handle, bristles exit at z = 0
func brushMeshes(_ b: Brush) -> (ferrule: [Vtx], handle: [Vtx]) {
    let rf: Float = b.round ? b.width * 0.5 + 0.35 : b.width * 0.24 + 1.6
    let lf: Float = b.round ? 14 + b.width * 1.5 : 16 + b.width * 0.9
    let p: Float = b.round ? 2 : 3.5
    let ferrule = sweep(48, 24) { u, v in
        let a = u * 2 * .pi
        let s = smoothstep(0, 0.55, v)
        let ax = b.round ? (b.width * 0.5 + 0.3) * (1 - s) + rf * s : (b.width * 0.5 + 0.25) * (1 - s) + rf * s
        let ay = b.round ? ax : (b.thick * 0.5 + 0.3) * (1 - s) + rf * s
        let pe = b.round ? 2 : p * (1 - s) + 2 * s
        let c = cos(a), sn = sin(a)
        let cx = (c < 0 ? -1 : 1) * pow(abs(c), 2 / pe), cy = (sn < 0 ? -1 : 1) * pow(abs(sn), 2 / pe)
        let crimp: Float = (v > 0.6 && v < 0.66) || (v > 0.7 && v < 0.74) ? 0.94 : 1
        return V3(cx * ax * crimp, cy * ay * crimp, v * lf)
    }
    let hl = b.handleLen, rh = rf * 0.98
    let prof: [(Float, Float)] = [(rh * 0.9, lf - 1), (rh, lf), (rh * 1.18, lf + hl * 0.12), (rh * 1.28, lf + hl * 0.28), (rh * 1.12, lf + hl * 0.5),
                                  (rh * 0.85, lf + hl * 0.8), (rh * 0.7, lf + hl * 0.97), (rh * 0.5, lf + hl), (0.01, lf + hl + 0.5)]
    return (ferrule + disk(48, rf, lf, up: true), lathe(40, prof))
}
func ferruleLen(_ b: Brush) -> Float { b.pencil ? 0 : b.knife ? 10 : (b.round ? 14 + b.width * 1.5 : 16 + b.width * 0.9) }

// palette knife: wooden handle, short collar, thin trowel blade running from the collar to the tip (local -z)
func knifeMeshes(_ b: Brush) -> (ferrule: [Vtx], handle: [Vtx], blade: [Vtx]) {
    let collar = lathe(32, [(2.0, -0.5), (3.6, 1.5), (4.2, 6), (4.2, 10)]) + disk(32, 4.2, 10, up: true)
    let hl = b.handleLen
    let handle = lathe(40, [(4.2, 9.5), (4.6, 12), (5.6, 10 + hl * 0.35), (5.2, 10 + hl * 0.8), (3.8, 10 + hl), (0.01, 10 + hl + 1)])
    let blade = sweep(24, 40) { u, v in
        let a = u * 2 * .pi
        let w: Float = v < 0.12 ? 1.6 : b.width * 0.5 * (1 - 0.55 * pow(abs(v - 0.45) / 0.55, 2)) * smoothstep(0.1, 0.3, v) + 1.6 * (1 - smoothstep(0.1, 0.3, v))
        let tip = 1 - smoothstep(0.9, 1.0, v) * 0.85
        return V3(cos(a) * w * tip, sin(a) * 0.35, -v * b.length)
    }
    return (collar, handle, blade)
}

// pencil: hexagonal lacquered body from z = 0 up, sharpened wood cone down to the graphite point at z = -length
func pencilMeshes(_ b: Brush) -> (wood: [Vtx], body: [Vtx], lead: [Vtx]) {
    let R: Float = 4, L = b.length, lead: Float = 2.5
    func hex(_ a: Float) -> Float { let s = Float.pi / 3; return R * cos(s / 2) / cos((a.truncatingRemainder(dividingBy: s)) - s / 2) }
    let body = sweep(48, 1) { u, v in let a = u * 2 * .pi; return V3(cos(a) * hex(a), sin(a) * hex(a), v * b.handleLen) }
        + disk(6, R * 0.98, b.handleLen, up: true)
    // the knife cuts a cone; where it meets the flats it leaves a scalloped edge
    let wood = sweep(48, 12) { u, v in
        let a = u * 2 * .pi, r = min(0.55 + (R - 0.55) * v, hex(a))
        return V3(cos(a) * r, sin(a) * r, -(L - lead) * (1 - v))
    }
    let tip = lathe(24, [(0.05, -L), (0.3, -L + 0.6), (0.55, -L + lead)])
    return (wood, body, tip)
}

// paint tube: nozzle at z=0 pointing -z, crimp end at z = 112
let TUBE_R: Float = 9.5
func tubeBody(left: Float, seed: Float) -> [Vtx] {
    let f0 = 0.5 + 0.43 * left
    return sweep(48, 60) { u, v in
        let a = u * 2 * .pi
        let z = 6 + v * 106
        let s = smoothstep(f0, 0.93, v)
        let dent = (1 - left) * 0.12 * sin(v * 23 + seed) * sin(a * 3 + seed * 2)
        let ax = TUBE_R * (1 + 0.55 * s), ay = max(TUBE_R * (1 - 0.95 * s) * (1 + dent), 0.5)
        return V3(cos(a) * ax, sin(a) * ay, z)
    }
}
func tubeShoulder() -> [Vtx] { lathe(48, [(3.6, -6), (4.0, -5.5), (4.0, -1), (4.4, -0.5), (6.5, 1.8), (8.6, 4.2), (TUBE_R, 6.2)]) + disk(24, 3.6, -6, up: false) }
func tubeCap() -> [Vtx] { lathe(64, [(6.4, -17), (6.9, -16.6), (7.0, -2), (6.6, -1.2)]) + disk(32, 6.4, -17, up: false) }

// ---------------------------------------------------------------- labels (CoreText into an atlas)
let LABEL_W = 512, LABEL_H = 160
func makeLabels() -> MTLTexture {
    let W = LABEL_W, H = LABEL_H * NPIG
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4, space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    func font(_ name: String, _ size: CGFloat) -> CTFont { CTFontCreateWithName(name as CFString, size, nil) }
    func text(_ s: String, _ f: CTFont, _ c: CGColor, _ x: CGFloat, _ y: CGFloat) {
        let a = NSAttributedString(string: s, attributes: [NSAttributedString.Key(kCTFontAttributeName as String): f, NSAttributedString.Key(kCTForegroundColorAttributeName as String): c])
        let line = CTLineCreateWithAttributedString(a)
        ctx.textPosition = CGPoint(x: x, y: y)
        CTLineDraw(line, ctx)
    }
    for (i, p) in PIGMENTS.enumerated() {
        let y0 = CGFloat(H - (i + 1) * LABEL_H)
        ctx.setFillColor(CGColor(srgbRed: 0.93, green: 0.915, blue: 0.87, alpha: 1))
        ctx.fill(CGRect(x: 0, y: y0, width: CGFloat(W), height: CGFloat(LABEL_H)))
        var v = [Float](repeating: 0, count: NPIG); v[i] = 1
        let mass = mixColor(v, thick: 60)
        var tv = [Float](repeating: 0, count: NPIG); tv[i] = 0.3; tv[0] += 0.7
        let tint = mixColor(tv, thick: 60)
        func cg(_ c: V3) -> CGColor { CGColor(srgbRed: CGFloat(linToSRGB(c.x)), green: CGFloat(linToSRGB(c.y)), blue: CGFloat(linToSRGB(c.z)), alpha: 1) }
        // colour band near the shoulder: masstone with a strip of tint
        ctx.setFillColor(cg(mass)); ctx.fill(CGRect(x: 0, y: y0, width: 150, height: CGFloat(LABEL_H)))
        ctx.setFillColor(cg(tint)); ctx.fill(CGRect(x: 150, y: y0, width: 14, height: CGFloat(LABEL_H)))
        let ink = CGColor(srgbRed: 0.12, green: 0.11, blue: 0.10, alpha: 1)
        ctx.setFillColor(ink); ctx.fill(CGRect(x: 172, y: y0 + 112, width: 330, height: 1.5)); ctx.fill(CGRect(x: 172, y: y0 + 46, width: 330, height: 1.5))
        let name = p.name.uppercased()
        var fs: CGFloat = 26
        while fs > 12 {
            let a = NSAttributedString(string: name, attributes: [NSAttributedString.Key(kCTFontAttributeName as String): font("AvenirNext-DemiBold", fs)])
            if CTLineGetTypographicBounds(CTLineCreateWithAttributedString(a), nil, nil, nil) < 325 { break }
            fs -= 1
        }
        text(name, font("AvenirNext-DemiBold", fs), ink, 174, y0 + 72)
        text("ARTISTS' OIL COLOUR", font("AvenirNext-Medium", 13), ink, 174, y0 + 122)
        text("\(p.code)   ·   37 ml   ·   series \(1 + i % 4)", font("AvenirNext-Regular", 13), ink, 174, y0 + 24)
    }
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm_srgb, width: W, height: H, mipmapped: true)
    d.usage = .shaderRead
    let t = GPU.shared.dev.makeTexture(descriptor: d)!
    t.replace(region: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0, withBytes: ctx.data!, bytesPerRow: W * 4)
    let cmd = GPU.shared.q.makeCommandBuffer()!, bl = cmd.makeBlitCommandEncoder()!
    bl.generateMipmaps(for: t); bl.endEncoding(); cmd.commit()
    return t
}

// ---------------------------------------------------------------- camera
struct Camera {
    var center = V2(90, -45)
    var dist: Float = 980
    let fov: Float = 26 * .pi / 180
    var eye: V3 { V3(center.x, center.y, dist) }
    func viewProj(_ aspect: Float) -> float4x4 {
        perspective(fov, aspect, max(dist * 0.02, 2), dist + 400) * lookAt(eye, V3(center.x, center.y, 0), V3(0, 1, 0))
    }
    // ray through normalised device coords
    func ray(_ ndc: V2, _ aspect: Float) -> (V3, V3) {
        let inv = viewProj(aspect).inverse
        let a = inv * V4(ndc.x, ndc.y, 0, 1), b = inv * V4(ndc.x, ndc.y, 1, 1)
        let p0 = V3(a.x, a.y, a.z) / a.w, p1 = V3(b.x, b.y, b.z) / b.w
        return (p0, simd_normalize(p1 - p0))
    }
}
func hitPlane(_ r: (V3, V3), _ z: Float) -> V2 { let t = (z - r.0.z) / r.1.z; return V2(r.0.x + r.1.x * t, r.0.y + r.1.y * t) }

// ---------------------------------------------------------------- scene
enum Held: Equatable { case brush(Int), tube(Int), none }

final class Scene {
    let g = GPU.shared
    let eng: Engine
    var cam = Camera()
    let surfRPS, objRPS, hairRPS, shadowRPS: MTLRenderPipelineState
    let dss, dssShadow: MTLDepthStencilState
    let shadowTex: MTLTexture, shadowHeld: MTLTexture
    let labels: MTLTexture
    var meshes: [String: Mesh] = [:]
    var brushMesh: [[Mesh]] = []
    var tubeMesh: [Mesh] = []
    var tubeLeft: [Float] = []
    var hairBufs: [MTLBuffer] = []
    var hairIdx = 0
    let HPT = 3     // hairs per tuft

    // interaction state (set by the app)
    private(set) var held: Held = .brush(1)
    // put the previously held brush back in its slot
    func setHeld(_ h: Held) {
        if h == held { return }
        if case .brush(let i) = held { restPose(eng.brushes[i], i) }
        held = h
    }
    var cursor = V3(0, 0, 0)        // world point under the cursor
    var tubePress: Float = 0
    var heat = false
    var hideMode = 0          // held brush: 0 whole, 1 without handle, 2 hidden
    var time: Float = 0
    var debug: Float = 0
    var exportView: (vp: float4x4, eye: V3)?        // orthographic top-down view of the canvas for export

    let lightDir = simd_normalize(V3(-0.42, 0.52, 0.74))

    init(_ e: Engine, sampleCount: Int, colorFormat: MTLPixelFormat) {
        eng = e
        let gp = GPU.shared, dev = gp.dev
        func pipe(_ v: String, _ f: String?, a2c: Bool = false, shadow: Bool = false) -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = gp.lib.makeFunction(name: v)
            if let f = f { d.fragmentFunction = gp.lib.makeFunction(name: f) }
            if shadow { d.depthAttachmentPixelFormat = .depth32Float; d.rasterSampleCount = 1 } else {
                d.colorAttachments[0].pixelFormat = colorFormat; d.depthAttachmentPixelFormat = .depth32Float
                d.rasterSampleCount = sampleCount; d.isAlphaToCoverageEnabled = a2c
            }
            return try! dev.makeRenderPipelineState(descriptor: d)
        }
        surfRPS = pipe("surfVert", "surfFrag")
        objRPS = pipe("objVert", "objFrag")
        hairRPS = pipe("hairVert", "hairFrag")
        shadowRPS = pipe("shadowVert", nil, shadow: true)
        let d = MTLDepthStencilDescriptor(); d.depthCompareFunction = .lessEqual; d.isDepthWriteEnabled = true
        dss = dev.makeDepthStencilState(descriptor: d)!
        dssShadow = dev.makeDepthStencilState(descriptor: d)!
        let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: 4096, height: 4096, mipmapped: false)
        sd.usage = [.renderTarget, .shaderRead]; sd.storageMode = .private
        shadowTex = dev.makeTexture(descriptor: sd)!
        shadowHeld = dev.makeTexture(descriptor: sd)!
        labels = makeLabels()
        buildMeshes()
        for _ in 0..<3 { hairBufs.append(g.buffer(4000 * HPT * MemoryLayout<Hair>.stride)) }
        for (i, b) in e.brushes.enumerated() { restPose(b, i) }
    }

    func buildMeshes() {
        meshes["table"] = Mesh(box(V3(-2500, -2500, TABLE_Z - 30), V3(2500, 2500, TABLE_Z)))
        meshes["unitSides"] = Mesh(sweep(4, 1) { u, v in
            // unit square outline (x,y in 0..1) extruded z -1..0
            let t = u * 4, k = min(Int(t), 3), f = t - Float(k)
            let c: [V2] = [V2(0, 0), V2(1, 0), V2(1, 1), V2(0, 1), V2(0, 0)]
            let p = c[k] + (c[k + 1] - c[k]) * f
            return V3(p.x, p.y, v - 1)
        }.map { var x = $0; x.n = V4(-x.n.x, -x.n.y, -x.n.z, x.n.w); return x })
        meshes["unitTop"] = Mesh(box(V3(0, 0, -1), V3(1, 1, 0)))
        let p = eng.palette
        let x1 = Float(p.w) / PPM, y1 = Float(p.h) / PPM, th: Float = 5.5
        let hc = V2(p.hole.x, p.hole.y) / PPM, hr = p.hole.z / PPM
        let holeWall = sweep(48, 1, flip: true) { u, v in V3(hc.x + cos(u * 2 * .pi) * hr, hc.y + sin(u * 2 * .pi) * hr, -th + v * th) }
        meshes["palette"] = Mesh(roundedWall(0, 0, x1, y1, p.maskR / PPM, -th, 0) + holeWall)
        // shadow caster: the top face with the thumb hole cut out (ring between the outline and the hole, by angle around it)
        let o = roundedOutline(0, 0, x1, y1, p.maskR / PPM)
        var cap = [Vtx](), up = V4(0, 0, 1, 0)
        for i in 0..<(o.count - 1) {
            let k = max(1, Int(simd_distance(o[i], o[i + 1]) / 4))
            for j in 0..<k {
                let a = o[i] + (o[i + 1] - o[i]) * Float(j) / Float(k), b = o[i] + (o[i + 1] - o[i]) * Float(j + 1) / Float(k)
                let ca = hc + simd_normalize(a - hc) * hr, cb = hc + simd_normalize(b - hc) * hr
                cap += [V2](arrayLiteral: ca, a, b, ca, b, cb).map { Vtx(p: V4($0.x, $0.y, 0, 0), n: up) }
            }
        }
        meshes["paletteCap"] = Mesh(cap)
        // tube tray
        let tx0: Float = 238, tx1: Float = 512, ty0: Float = -264, ty1: Float = -150
        var tray = box(V3(tx0, ty0, TABLE_Z), V3(tx1, ty1, TABLE_Z + 3))
        tray += box(V3(tx0, ty0, TABLE_Z), V3(tx1, ty0 + 6, TABLE_Z + 16)) + box(V3(tx0, ty1 - 6, TABLE_Z), V3(tx1, ty1, TABLE_Z + 16))
        tray += box(V3(tx0, ty0, TABLE_Z), V3(tx0 + 6, ty1, TABLE_Z + 16)) + box(V3(tx1 - 6, ty0, TABLE_Z), V3(tx1, ty1, TABLE_Z + 16))
        meshes["tray"] = Mesh(tray)
        meshes["shoulder"] = Mesh(tubeShoulder())
        meshes["cap"] = Mesh(tubeCap())
        for (i, t) in eng.tubes.enumerated() { tubeMesh.append(Mesh(tubeBody(left: t.left, seed: Float(i)))); tubeLeft.append(t.left) }
        for b in eng.brushes {
            if b.knife { let m = knifeMeshes(b); brushMesh.append([Mesh(m.ferrule), Mesh(m.handle), Mesh(m.blade)]) }
            else if b.pencil { let m = pencilMeshes(b); brushMesh.append([Mesh(m.wood), Mesh(m.body), Mesh(m.lead)]) }
            else { let m = brushMeshes(b); brushMesh.append([Mesh(m.ferrule), Mesh(m.handle)]) }
        }
    }

    // ------------------------------------------------ placement
    let TS: Float = 0.72     // small (21 ml) tubes
    func tubeRest(_ i: Int) -> float4x4 {
        let x: Float = 250 + Float(TUBE_ORDER.firstIndex(of: i) ?? i) * 16.6
        let p = V3(x, -175, TABLE_Z + 3 + TUBE_R * 0.92 * TS)
        return frame(V3(-TS, 0, 0), V3(0, 0, -TS), V3(0, -TS, 0), p)    // caps toward the palette, label facing up
    }
    func tubeHeld(_ at: V3) -> float4x4 {
        let ax = simd_normalize(V3(0.32, -0.42, 0.85))
        let x = simd_normalize(simd_cross(V3(0, 0, 1), ax)), y = simd_cross(ax, x)
        return frame(x * TS, y * TS, ax * TS, at + ax * 2)
    }
    // brushes lie below the canvas with their bristles pointing at it
    func brushRestPos(_ i: Int) -> V3 { V3(-175 + Float(BRUSH_ORDER.firstIndex(of: i) ?? i) * 26, -176 - eng.brushes[i].length, eng.brushes[i].knife ? 6 : (eng.brushes[i].pencil ? 3.5 : 4.6)) }    // z relative to the table
    func restPose(_ b: Brush, _ i: Int) {
        b.baseZ = TABLE_Z
        b.axis = V3(0, 1, 0); b.e1 = V3(1, 0, 0); b.e2 = simd_cross(b.axis, b.e1)
        b.ferrule = brushRestPos(i)
        var bb = IRect.none
        _ = b.simulate(surface: nil, moved: 0, out: nil, bbox: &bb)
    }
    func brushModel(_ b: Brush) -> float4x4 { frame(b.e1, -b.e2, -b.axis, b.ferrule + V3(0, 0, b.baseZ)) }

    // tube body regenerates as paint gets used
    func refreshTubes() {
        for (i, t) in eng.tubes.enumerated() where abs(t.left - tubeLeft[i]) > 0.004 {
            tubeMesh[i] = Mesh(tubeBody(left: t.left, seed: Float(i))); tubeLeft[i] = t.left
        }
    }

    // render the (cropped) canvas straight from above at one pixel per texel
    func export(scale: Float = 1) -> CGImage? {
        let m = eng.canvas.mask, o = eng.canvas.origin
        let w = Int((m.z - m.x) * scale), h = Int((m.w - m.y) * scale)
        guard w > 0, h > 0, w <= 16384, h <= 16384 else { return nil }
        let x0 = o.x + m.x / PPM, y0 = o.y + m.y / PPM, x1 = o.x + m.z / PPM, y1 = o.y + m.w / PPM
        let eye = V3((x0 + x1) / 2, (y0 + y1) / 2, 3000)
        exportView = (ortho((x0 - x1) / 2, (x1 - x0) / 2, (y0 - y1) / 2, (y1 - y0) / 2, 100, 4000) * lookAt(eye, V3(eye.x, eye.y, 0), V3(0, 1, 0)), eye)
        defer { exportView = nil }
        let off = Offscreen(w, h)
        let cmd = g.q.makeCommandBuffer()!
        eng.shade(cmd)
        encode(cmd, off.pass, aspect: Float(w) / Float(h))
        cmd.commit(); cmd.waitUntilCompleted()
        return off.image()
    }

    // frame the canvas, palette, tube tray and the bristle ends of the brushes
    func fitDesk(_ aspect: Float) {
        let c = eng.canvas
        let lo = V2(min(c.origin.x - 15, -195), -272), hi = V2(eng.palette.origin.x + Float(eng.palette.w) / PPM + 12, max(c.origin.y + Float(c.h) / PPM, 120) + 15)
        fit(lo, hi, aspect)
    }
    func fitCanvas(_ aspect: Float) {
        let c = eng.canvas
        let lo = c.toWorld(V2(c.mask.x, c.mask.y)), hi = c.toWorld(V2(c.mask.z, c.mask.w))
        fit(lo - 12, hi + 12, aspect)
    }
    func fit(_ lo: V2, _ hi: V2, _ aspect: Float) {
        let t = tan(cam.fov / 2)
        cam.center = (lo + hi) / 2
        cam.dist = max((hi.x - lo.x) / (2 * t * aspect), (hi.y - lo.y) / (2 * t))
    }

    // world point -> view point (for synthesized input in the self test)
    func project(_ w: V3, _ size: CGSize) -> CGPoint {
        let c = cam.viewProj(Float(size.width / size.height)) * V4(w, 1)
        return CGPoint(x: CGFloat((c.x / c.w * 0.5 + 0.5)) * size.width, y: CGFloat((c.y / c.w * 0.5 + 0.5)) * size.height)
    }

    // ------------------------------------------------ picking
    // what a click picks up; clicking the empty slot of the held thing puts it down (.none)
    func pick(_ r: (V3, V3)) -> Held? {
        let pt = hitPlane(r, TABLE_Z + 10)
        for i in 0..<eng.tubes.count {
            let c = tubeRest(i).columns.3
            if abs(pt.x - c.x) < 7.5, pt.y > c.y - 82, pt.y < c.y + 13 { return held == .tube(i) ? Held.none : .tube(i) }
        }
        let pb = hitPlane(r, TABLE_Z + 4)
        for i in 0..<brushMesh.count {        // brushes defined at runtime have no model to pick up
            let p = brushRestPos(i), b = eng.brushes[i]
            if abs(pb.x - p.x) < (b.knife ? 11 : 6.5), pb.y < p.y + b.length, pb.y > p.y - ferruleLen(b) - b.handleLen { return held == .brush(i) ? Held.none : .brush(i) }
        }
        return nil
    }

    // ------------------------------------------------ render
    func hairs(_ b: Brush, into ptr: UnsafeMutablePointer<Hair>) -> Int {
        var k = 0
        let z = b.baseZ
        var rng = Rng(s: 7)
        for j in 0..<b.n {
            let R = b.root[j] + V3(0, 0, z)
            let L = b.lens[j]
            var p1: V3, p2: V3, p3: V3
            if b.touching[j] {
                p3 = V3(b.tip[j].x, b.tip[j].y, z + 0.08)
                let bp = V3(b.bendPt[j].x, b.bendPt[j].y, z + 0.1)
                p1 = R + b.axis * L * 0.35
                p2 = bp + simd_normalize(p1 - bp) * min(simd_length(p1 - bp) * 0.3, L * 0.2)
                p2.z = max(p2.z, z + 0.1)
            } else {
                p3 = V3(b.tip[j].x, b.tip[j].y, b.tipZ[j] + z)
                p1 = R + (p3 - R) * 0.33; p2 = R + (p3 - R) * 0.66
            }
            for _ in 0..<HPT {
                let o = (b.e1 * rng.gauss() + b.e2 * rng.gauss()) * b.rads[j] * 0.6
                let ot = (b.e1 * rng.gauss() + b.e2 * rng.gauss()) * b.rads[j] * 0.9
                ptr[k] = Hair(p0: V4(R + o, Float(j)), p1: V4(p1 + o * 0.8, b.rads[j] * 0.9), p2: V4(p2 + ot * 0.6, 0.5), p3: V4(p3 + ot * 0.4, rng.next()))
                k += 1
            }
        }
        return k
    }

    func frameUniforms(_ aspect: Float) -> Frame {
        let cv = eng.canvas, pl = eng.palette
        let lo = simd_min(simd_min(cv.origin, pl.origin), V2(-200, -290))
        let hi = simd_max(simd_max(cv.origin + V2(Float(cv.w), Float(cv.h)) / PPM, pl.origin + V2(Float(pl.w), Float(pl.h)) / PPM), V2(570, 130))
        let m = (lo + hi) / 2, ext = max(560, simd_reduce_max(hi - lo) * 0.5 + 60)
        let c = V3(m.x, m.y, TABLE_Z)
        let lv = lookAt(c + lightDir * 1500, c, V3(0, 1, 0))
        let lp = ortho(-ext, ext, -ext, ext, 400, 2600)
        return Frame(vp: exportView?.vp ?? cam.viewProj(aspect), lvp: lp * lv, cam: V4(exportView?.eye ?? cam.eye, 1), ldir: V4(lightDir, 0),
                     lcol: V4(1.0, 0.95, 0.88, 0) * 1.12, amb: V4(0.30, 0.31, 0.34, 0) * 0.62,
                     params: V4(heat ? 1 : 0, time, 0, debug))
    }

    struct Draw { let mesh: Mesh; var d: ObjDraw; let shadow: Bool; var main = true; var isHeld = false }
    func drawList() -> [Draw] {
        var list = [Draw]()
        func add(_ m: Mesh, _ model: float4x4, _ base: V3, metal: Float = 0, rough: Float = 0.5, mode: Float = 0, fade: Float = 0, extra: V4 = .zero, shadow: Bool = true, main: Bool = true) {
            list.append(Draw(mesh: m, d: ObjDraw(model: model, base: V4(base, 1), mat: V4(metal, rough, mode, fade), extra: extra), shadow: shadow, main: main, isHeld: fade > 0))
        }
        add(meshes["table"]!, matrix_identity_float4x4, V3(0.4, 0.25, 0.15), rough: 0.55, mode: 5, shadow: false)
        let cv = eng.canvas, m = cv.mask
        let cx0 = cv.origin.x + m.x / PPM, cy0 = cv.origin.y + m.y / PPM, cw = (m.z - m.x) / PPM, ch = (m.w - m.y) / PPM
        add(meshes["unitSides"]!, translate(V3(cx0, cy0, 0)) * scale(V3(cw, ch, -TABLE_Z)), V3(0.86, 0.84, 0.79), rough: 0.9, mode: 4, extra: V4(cw * 0.75, 15, 0, 0))
        add(meshes["unitTop"]!, translate(V3(cx0, cy0, 0)) * scale(V3(cw, ch, -TABLE_Z)), .zero, main: false)
        let pm = translate(V3(eng.palette.origin.x, eng.palette.origin.y, eng.palette.z))
        add(meshes["palette"]!, pm, V3(0.74, 0.55, 0.34), rough: 0.4, mode: 2, extra: V4(1, 0, 0, 2))
        add(meshes["paletteCap"]!, pm, .zero, main: false)
        add(meshes["tray"]!, matrix_identity_float4x4, V3(0.55, 0.40, 0.26), rough: 0.6, mode: 2, extra: V4(1, 30, 0, 5))
        for i in 0..<eng.tubes.count {
            var model = tubeRest(i), inTray = true
            if case .tube(let h) = held, h == i { model = tubeHeld(cursor); inTray = false }
            let fade: Float = inTray ? 0 : 160
            let lab = V4(0.16, 0.84, Float(i) / Float(NPIG), 1 / Float(NPIG))
            add(tubeMesh[i], model, V3(0.85, 0.85, 0.85), metal: 1, rough: 0.3, mode: 1, fade: fade, extra: lab)
            add(meshes["shoulder"]!, model, V3(0.86, 0.86, 0.85), metal: 1, rough: 0.25, fade: fade)
            if inTray { add(meshes["cap"]!, model, V3(0.03, 0.03, 0.03), rough: 0.35, mode: 3) }
            else { add(meshes["cap"]!, tubeRest(i), V3(0.03, 0.03, 0.03), rough: 0.35, mode: 3) }
        }
        for (i, b) in eng.brushes.enumerated() where i < brushMesh.count {
            var fade: Float = 0, hide = 0
            if case .brush(let h) = held, h == i { fade = 140; hide = hideMode }
            let model = brushModel(b)
            if hide < 2 && b.pencil {
                add(brushMesh[i][0], model, V3(0.78, 0.60, 0.42), rough: 0.8, fade: fade * 0.6)
                add(brushMesh[i][2], model, V3(0.09, 0.09, 0.10), metal: 0.6, rough: 0.35, fade: fade > 0 ? 1 : 0)
            } else if hide < 2 {
                add(brushMesh[i][0], model, V3(0.80, 0.78, 0.74), metal: 1, rough: 0.22, fade: fade * 0.6)
                if b.knife { add(brushMesh[i][2], model, V3(0.82, 0.82, 0.83), metal: 1, rough: 0.18, mode: 10, fade: fade > 0 ? 1 : 0) }
            }
            if hide < 1 { add(brushMesh[i][1], model, b.handleColor, rough: b.knife ? 0.45 : (b.pencil ? 0.3 : 0.18), fade: fade) }
        }
        return list
    }

    func encodeShadows(_ cmd: MTLCommandBuffer, _ F: Frame, _ list: [Draw]) {
        shadowPass(cmd, F, list.filter { !$0.isHeld }, shadowTex)
        shadowPass(cmd, F, list.filter { $0.isHeld }, shadowHeld)
    }
    func shadowPass(_ cmd: MTLCommandBuffer, _ F: Frame, _ list: [Draw], _ tex: MTLTexture) {
        let rp = MTLRenderPassDescriptor()
        rp.depthAttachment.texture = tex; rp.depthAttachment.loadAction = .clear; rp.depthAttachment.clearDepth = 1; rp.depthAttachment.storeAction = .store
        let e = cmd.makeRenderCommandEncoder(descriptor: rp)!
        e.setRenderPipelineState(shadowRPS); e.setDepthStencilState(dssShadow)
        e.setDepthBias(0.5, slopeScale: 2, clamp: 0.01)
        var F = F
        e.setVertexBytes(&F, length: MemoryLayout<Frame>.stride, index: 1)
        for d in list where d.shadow {
            var od = d.d
            e.setVertexBuffer(d.mesh.buf, offset: 0, index: 0)
            e.setVertexBytes(&od, length: MemoryLayout<ObjDraw>.stride, index: 2)
            e.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: d.mesh.count)
        }
        e.endEncoding()
    }

    func encode(_ cmd: MTLCommandBuffer, _ rp: MTLRenderPassDescriptor, aspect: Float) {
        refreshTubes()
        var F = frameUniforms(aspect)
        let list = exportView == nil ? drawList() : []
        encodeShadows(cmd, F, list)
        let e = cmd.makeRenderCommandEncoder(descriptor: rp)!
        e.setDepthStencilState(dss)
        e.setCullMode(.back); e.setFrontFacing(.counterClockwise)
        e.setVertexBytes(&F, length: MemoryLayout<Frame>.stride, index: 1)
        e.setFragmentBytes(&F, length: MemoryLayout<Frame>.stride, index: 1)
        e.setFragmentTexture(shadowTex, index: 2)
        e.setFragmentTexture(shadowHeld, index: 5)
        e.setFragmentTexture(labels, index: 4)
        // objects
        e.setRenderPipelineState(objRPS)
        for d in list where d.main {
            var od = d.d
            e.setVertexBuffer(d.mesh.buf, offset: 0, index: 0)
            e.setVertexBytes(&od, length: MemoryLayout<ObjDraw>.stride, index: 2)
            e.setFragmentBytes(&od, length: MemoryLayout<ObjDraw>.stride, index: 2)
            e.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: d.mesh.count)
        }
        // paint surfaces
        e.setCullMode(.none)
        e.setRenderPipelineState(surfRPS)
        e.setVertexBytes(&F, length: MemoryLayout<Frame>.stride, index: 0)
        e.setFragmentBytes(&F, length: MemoryLayout<Frame>.stride, index: 0)
        for s in eng.surfaces where exportView == nil || s.kind == 0 {
            var sd = SurfDraw(orgSize: V4(s.origin.x, s.origin.y, Float(s.w), Float(s.h)), mask: s.mask, misc: V4(s.z, s.maskR, Float(s.kind), 0), heat: s.hole)
            e.setVertexBytes(&sd, length: MemoryLayout<SurfDraw>.stride, index: 1)
            e.setFragmentBytes(&sd, length: MemoryLayout<SurfDraw>.stride, index: 1)
            e.setFragmentTexture(s.color, index: 0); e.setFragmentTexture(s.nrm, index: 1); e.setFragmentTexture(s.prev, index: 3)
            e.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        // bristles
        hairIdx = (hairIdx + 1) % hairBufs.count
        let hb = hairBufs[hairIdx]
        var off = 0
        e.setRenderPipelineState(hairRPS)
        e.setVertexBytes(&F, length: MemoryLayout<Frame>.stride, index: 1)
        e.setFragmentBytes(&F, length: MemoryLayout<Frame>.stride, index: 1)
        for (i, b) in eng.brushes.enumerated() where exportView == nil && !b.knife && !b.pencil && (i < brushMesh.count || held == .brush(i)) {
            if case .brush(let h) = held, h == i, hideMode == 2 { continue }
            if off + b.n * HPT * MemoryLayout<Hair>.stride > hb.length { continue }
            let ptr = (hb.contents() + off).bindMemory(to: Hair.self, capacity: b.n * HPT)
            let n = hairs(b, into: ptr)
            var base = V4(b.hairColor, 1)
            e.setVertexBuffer(hb, offset: off, index: 0)
            e.setVertexBuffer(b.colors, offset: 0, index: 2)
            e.setFragmentBytes(&base, length: 16, index: 3)
            e.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 22, instanceCount: n)
            off += n * MemoryLayout<Hair>.stride
        }
        e.endEncoding()
    }
}
