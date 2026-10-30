// App shell: window, input, HUD, replay, save/load/export, and headless test renders.
import AppKit
import MetalKit
import simd
import UniformTypeIdentifiers

// ---------------------------------------------------------------- offscreen rendering (tests, export)
final class Offscreen {
    let w: Int, h: Int
    let msaa: MTLTexture, depth: MTLTexture, resolve: MTLTexture
    init(_ w: Int, _ h: Int) {
        self.w = w; self.h = h
        let dev = GPU.shared.dev
        let m = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm_srgb, width: w, height: h, mipmapped: false)
        m.textureType = .type2DMultisample; m.sampleCount = 4; m.usage = .renderTarget; m.storageMode = .private
        msaa = dev.makeTexture(descriptor: m)!
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: w, height: h, mipmapped: false)
        d.textureType = .type2DMultisample; d.sampleCount = 4; d.usage = .renderTarget; d.storageMode = .private
        depth = dev.makeTexture(descriptor: d)!
        let r = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm_srgb, width: w, height: h, mipmapped: false)
        r.usage = [.renderTarget, .shaderRead]; r.storageMode = .shared
        resolve = dev.makeTexture(descriptor: r)!
    }
    var pass: MTLRenderPassDescriptor {
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = msaa; rp.colorAttachments[0].resolveTexture = resolve
        rp.colorAttachments[0].loadAction = .clear; rp.colorAttachments[0].storeAction = .multisampleResolve
        rp.colorAttachments[0].clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1)
        rp.depthAttachment.texture = depth; rp.depthAttachment.loadAction = .clear; rp.depthAttachment.clearDepth = 1; rp.depthAttachment.storeAction = .dontCare
        return rp
    }
    func image() -> CGImage {
        var px = [UInt8](repeating: 0, count: w * h * 4)
        resolve.getBytes(&px, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
        return ctx.makeImage()!
    }
}
func writePNG(_ img: CGImage, _ path: String) {
    let url = URL(fileURLWithPath: path) as CFURL
    let d = CGImageDestinationCreateWithURL(url, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(d, img, nil); CGImageDestinationFinalize(d)
}

// ---------------------------------------------------------------- scripted painting (tests)
final class Script {
    let e: Engine
    var t: Float = 0
    init(_ e: Engine) { self.e = e }
    func run(_ f: (MTLCommandBuffer) -> Void) {
        e.beginFrame()
        let cmd = GPU.shared.q.makeCommandBuffer()!
        let t0 = CACurrentMediaTime()
        f(cmd)
        cpu += CACurrentMediaTime() - t0
        cmd.commit(); cmd.waitUntilCompleted()
        gpu += cmd.gpuEndTime - cmd.gpuStartTime
    }
    var cpu = 0.0, gpu = 0.0
    func path(_ pts: [V3], speed: Float) -> [(V2, Float)] {     // (x,y,pressure) polyline -> samples at 120 Hz
        var out = [(V2, Float)]()
        let dt: Float = 1 / 120
        for i in 0..<(pts.count - 1) {
            let a = pts[i], b = pts[i + 1]
            let n = max(1, Int(simd_length(V2(b.x - a.x, b.y - a.y)) / (speed * dt)))
            for k in 0..<n { let f = Float(k) / Float(n); let p = a + (b - a) * f; out.append((V2(p.x, p.y), p.z)) }
        }
        out.append((V2(pts.last!.x, pts.last!.y), pts.last!.z))
        return out
    }
    var roll: [Float] = [Float](repeating: 0, count: 12)
    func stroke(_ brush: Int, _ s: Int8, _ pts: [V3], speed: Float = 100, tilt: V2 = V2(0.3, -0.3), fixedRoll: Float? = nil) {
        e.begin(.stroke(brush: brush, samples: []))
        let ps = path(pts, speed: speed)
        var i = 0
        while i < ps.count {
            run { cmd in
                for _ in 0..<8 where i < ps.count {
                    t += 1 / 120
                    if let r = fixedRoll { roll[brush] = r } else if i > 0, !e.brushes[brush].round {
                        let d = ps[i].0 - ps[i - 1].0
                        if simd_length(d) > 0.05 { roll[brush] = Engine.followRoll(roll[brush], dir: d) }
                    }
                    e.add(Sample(x: ps[i].0.x, y: ps[i].0.y, p: ps[i].1, tx: tilt.x, ty: tilt.y, roll: roll[brush], t: t, s: s)); i += 1
                }
                e.process(cmd)
            }
        }
        if case .stroke(_, let ss) = e.cur!.op { for m in Engine.liftSamples(ss) { e.add(m) } }
        run { cmd in e.end(cmd) }
        t += 0.3
    }
    func squeeze(_ tube: Int, _ pts: [V3], speed: Float = 25) {
        e.begin(.squeeze(tube: tube, samples: []))
        let ps = path(pts, speed: speed)
        run { cmd in
            for p in ps { t += 1 / 120; e.add(Sample(x: p.0.x, y: p.0.y, p: p.1, tx: 0, ty: 0, roll: 0, t: t, s: 1)) }
            e.process(cmd)
        }
        run { cmd in e.end(cmd) }
    }
}

func runTest(_ name: String, _ out: String) {
    let e = Engine()
    let sc = Scene(e, sampleCount: 4, colorFormat: .bgra8Unorm_srgb)
    let s = Script(e)
    let pal = e.palette.origin
    func P(_ x: Float, _ y: Float) -> V2 { pal + V2(x, y) }
    func render(_ center: V2, _ dist: Float, _ w: Int, _ h: Int, _ path: String) {
        sc.cam.center = center; sc.cam.dist = dist
        let off = Offscreen(w, h)
        s.run { cmd in e.shade(cmd); sc.encode(cmd, off.pass, aspect: Float(w) / Float(h)) }
        writePNG(off.image(), path)
    }
    switch name {
    case "swatch":
        let W = 900, H = 640
        let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(gray: 0.5, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: W, height: H))
        func cg(_ c: V3) -> CGColor { CGColor(srgbRed: CGFloat(linToSRGB(c.x)), green: CGFloat(linToSRGB(c.y)), blue: CGFloat(linToSRGB(c.z)), alpha: 1) }
        // per pigment: masstone, glaze over white, 50% / 10% / 2% tints
        for (col, i) in TUBE_ORDER.filter({ $0 != MEDIUM }).enumerated() {
            let x = 10 + col * 58
            var v = [Float](repeating: 0, count: NPIG)
            v[i] = 1
            ctx.setFillColor(cg(mixColor(v, thick: 200))); ctx.fill(CGRect(x: x, y: H - 70, width: 52, height: 60))
            ctx.setFillColor(cg(mixColor(v, thick: 3))); ctx.fill(CGRect(x: x, y: H - 140, width: 52, height: 60))
            for (k, f) in [Float(0.5), 0.1, 0.02].enumerated() {
                var m = [Float](repeating: 0, count: NPIG); m[i] = f; m[0] += 1 - f
                ctx.setFillColor(cg(mixColor(m, thick: 200))); ctx.fill(CGRect(x: x, y: H - 210 - k * 70, width: 52, height: 60))
            }
        }
        // pair mixes 50/50
        let pairs = [(13, 8), (13, 15), (2, 8), (14, 7), (14, 8), (6, 7), (14, 13), (14, 2), (15, 4), (13, 9), (10, 7), (2, 1)]
        for (k, pr) in pairs.enumerated() {
            var m = [Float](repeating: 0, count: NPIG); m[pr.0] = 0.5; m[pr.1] = 0.5
            ctx.setFillColor(cg(mixColor(m, thick: 200))); ctx.fill(CGRect(x: 10 + k * 73, y: 10, width: 66, height: 60))
            m[0] = 1
            ctx.setFillColor(cg(mixColor(m.map { $0 / 2 }, thick: 200))); ctx.fill(CGRect(x: 10 + k * 73, y: 80, width: 66, height: 60))
        }
        writePNG(ctx.makeImage()!, out)
    case "strokes":
        s.squeeze(0, [V3(P(30, 150), 1), V3(P(55, 150), 1)])
        s.squeeze(7, [V3(P(30, 120), 1), V3(P(50, 118), 1)])
        s.squeeze(2, [V3(P(30, 90), 1), V3(P(48, 92), 1)])
        s.squeeze(6, [V3(P(30, 60), 1), V3(P(46, 60), 1)])
        // load round 12 with ultramarine
        s.stroke(2, 1, [V3(P(36, 112), 0.8), V3(P(40, 125), 0.9), V3(P(46, 112), 0.9), V3(P(52, 124), 0.8)], speed: 40)
        s.stroke(2, 0, [V3(-100, 60, 0.5), V3(-20, 70, 0.5), V3(60, 55, 0.5)], speed: 120)
        s.stroke(2, 0, [V3(-100, 30, 0.2), V3(-20, 38, 0.25), V3(60, 25, 0.2)], speed: 140)
        s.stroke(2, 0, [V3(-100, 0, 0.9), V3(-20, 8, 0.9), V3(60, -5, 0.9)], speed: 100)
        // flat 8: white with a touch of yellow, across the wet blue
        s.stroke(4, 1, [V3(P(30, 152), 0.8), V3(P(55, 149), 0.8), V3(P(30, 151), 0.8)], speed: 40)
        s.stroke(4, 1, [V3(P(36, 95), 0.5), V3(P(40, 90), 0.5)], speed: 30)
        s.stroke(4, 0, [V3(-60, 80, 0.5), V3(-55, -30, 0.5)], speed: 90)
        s.stroke(4, 0, [V3(20, 80, 0.3), V3(25, -30, 0.3)], speed: 120)
        s.stroke(4, 0, [V3(-90, -50, 0.6), V3(80, -55, 0.6)], speed: 120)
        let cs = e.readHeights(e.canvas, 480, 1300, 300, 1)
        print("yellow x-section", stride(from: 0, to: 300, by: 4).map { String(format: "%.0f", cs[$0]) }.joined(separator: " "))
        render(V2(-20, 15), 260, 1600, 1200, out)
        render(V2(P(45, 105).x, P(45, 105).y), 160, 1200, 1200, out.replacingOccurrences(of: ".png", with: "_pal.png"))
        render(V2(-55, 40), 90, 1400, 1000, out.replacingOccurrences(of: ".png", with: "_zoom.png"))
        sc.debug = 1
        render(V2(-55, 40), 90, 1400, 1000, out.replacingOccurrences(of: ".png", with: "_relief.png"))
    case "debug":
        s.squeeze(7, [V3(P(30, 120), 1), V3(P(50, 118), 1)])
        let hp = e.readHeights(e.palette, Int(30 * PPM), Int(105 * PPM), 1, 300)
        print("blob profile max", hp.max()!, "tooth", hp.min()!)
        print("load before", e.brushLoad(e.brushes[2]))
        s.stroke(2, 1, [V3(P(36, 112), 0.8), V3(P(40, 125), 0.9)], speed: 40)
        print("load after 1", e.brushLoad(e.brushes[2]), "cap tufts", e.brushes[2].capTuft * Float(e.brushes[2].n))
        s.stroke(2, 1, [V3(P(46, 112), 0.9), V3(P(52, 124), 0.8)], speed: 40)
        print("load after 2", e.brushLoad(e.brushes[2]))
        let hp2 = e.readHeights(e.palette, Int(30 * PPM), Int(105 * PPM), 1, 300)
        print("blob after max", hp2.max()!)
        s.stroke(2, 0, [V3(-100, 60, 0.5), V3(-20, 70, 0.5)], speed: 120)
        print("load after canvas", e.brushLoad(e.brushes[2]))
        let hc = e.readHeights(e.canvas, Int(60 * PPM), Int(140 * PPM), 1, 300)
        print("canvas cross-section", hc.enumerated().filter { $0.offset % 10 == 0 }.map { String(format: "%.1f", $0.element) }.joined(separator: " "))
    case "perf":
        s.squeeze(0, [V3(P(30, 150), 1), V3(P(55, 150), 1)])
        s.stroke(5, 1, [V3(P(30, 152), 0.8), V3(P(55, 149), 0.8)], speed: 40)
        for (bi, spd) in [(5, Float(200)), (2, Float(200)), (0, Float(100)), (5, Float(1500)), (2, Float(1500)), (8, Float(1500))] {
            let t0 = CACurrentMediaTime()
            s.cpu = 0; s.gpu = 0
            s.stroke(bi, 0, [V3(-100, -60, 0.6), V3(100, 60, 0.6)], speed: spd)
            let dt = CACurrentMediaTime() - t0
            print(String(format: "  cpu %.2f gpu %.2f", s.cpu, s.gpu))
            print("brush \(e.brushes[bi].name) \(e.brushes[bi].n) tufts, 233mm at \(spd)mm/s: sim \(String(format: "%.2f", dt))s for \(String(format: "%.2f", 233 / spd))s of painting")
        }
    case "dry":
        s.squeeze(7, [V3(P(30, 120), 1), V3(P(50, 118), 1)])
        s.squeeze(11, [V3(P(30, 90), 1), V3(P(48, 92), 1)])
        s.squeeze(0, [V3(P(30, 150), 1), V3(P(55, 150), 1)])
        s.stroke(2, 1, [V3(P(36, 112), 0.8), V3(P(40, 125), 0.9), V3(P(46, 112), 0.9)], speed: 40)
        s.stroke(2, 0, [V3(-80, 30, 0.3), V3(40, 35, 0.3)], speed: 80)          // thin blue
        s.stroke(5, 1, [V3(P(34, 96), 0.9), V3(P(44, 86), 0.9), V3(P(36, 92), 0.9)], speed: 40)
        s.stroke(5, 0, [V3(-80, -10, 0.25), V3(40, -5, 0.25)], speed: 70)       // thick umber (light touch)
        // preview a 20 h dry
        s.run { cmd in e.preview(cmd, hours: 20, palette: false) }
        sc.heat = true
        render(V2(-20, 10), 220, 1400, 1000, out.replacingOccurrences(of: ".png", with: "_heat.png"))
        sc.heat = false
        s.run { cmd in e.dry(cmd, hours: 20, palette: false) }
        // white strokes across both: blue (dried thin) should resist, umber (still wet) should smear
        s.run { e.brushes[4].clean($0) }
        s.stroke(4, 1, [V3(P(30, 152), 0.8), V3(P(55, 149), 0.8), V3(P(30, 151), 0.8)], speed: 40)
        s.stroke(4, 0, [V3(-60, 50, 0.5), V3(-58, -30, 0.5)], speed: 80)
        render(V2(-20, 10), 220, 1400, 1000, out)
        s.run { cmd in e.undo(cmd) }
        render(V2(-20, 10), 220, 1400, 1000, out.replacingOccurrences(of: ".png", with: "_undo1.png"))
        s.run { cmd in e.undo(cmd); e.undo(cmd) }
        render(V2(-20, 10), 220, 1400, 1000, out.replacingOccurrences(of: ".png", with: "_undo3.png"))
        s.run { cmd in e.redo(cmd); e.redo(cmd); e.redo(cmd) }
        render(V2(-20, 10), 220, 1400, 1000, out.replacingOccurrences(of: ".png", with: "_redo.png"))
    case "save":
        s.squeeze(7, [V3(P(30, 120), 1), V3(P(50, 118), 1)])
        s.stroke(2, 1, [V3(P(36, 112), 0.8), V3(P(40, 125), 0.9), V3(P(46, 112), 0.9)], speed: 40)
        s.stroke(2, 0, [V3(-80, 30, 0.5), V3(40, 35, 0.5)], speed: 80)
        e.crop(V4(200, 200, 2000, 1500))
        let url = URL(fileURLWithPath: out.replacingOccurrences(of: ".png", with: ".pirl"))
        let t0 = CACurrentMediaTime()
        try! e.save(to: url)
        print("saved", (try! Data(contentsOf: url)).count / 1024, "KB in", CACurrentMediaTime() - t0)
        let t1 = CACurrentMediaTime()
        HUD.setThumbnail(sc, url)
        let t2 = CACurrentMediaTime()
        HUD.setThumbnail(sc, url, wait: true)
        var r = CGRect(x: 0, y: 0, width: 256, height: 256)
        if let i = NSWorkspace.shared.icon(forFile: url.path).cgImage(forProposedRect: &r, context: nil, hints: nil) { writePNG(i, out.replacingOccurrences(of: ".png", with: "_icon.png")) }
        print(String(format: "thumbnail: %.1f ms on the main thread, %.0f ms in all", (t2 - t1) * 1000, (CACurrentMediaTime() - t2) * 1000))
        let e2 = Engine()
        try! e2.load(from: url)
        let sc2 = Scene(e2, sampleCount: 4, colorFormat: .bgra8Unorm_srgb)
        writePNG(sc2.export(scale: 0.5)!, out)
        func px(_ i: CGImage) -> Data { i.dataProvider!.data! as Data }
        e.crop(V4(0, 0, Float(e.canvas.w), Float(e.canvas.h))); e2.crop(V4(0, 0, Float(e2.canvas.w), Float(e2.canvas.h)))
        let a = sc.export(scale: 0.5)!, b = sc2.export(scale: 0.5)!
        print("reloaded painting identical:", px(a) == px(b), "painted tiles", e2.canvas.painted.filter { $0 }.count)
        print("history", e2.history.count)
        // a version-1 file (raw LZFSE tiles, history in the JSON header) still loads
        func v1(_ sf: Surface) -> Data {
            let tiles = (0..<sf.painted.count).filter { sf.painted[$0] }
            let buf = GPU.shared.buffer(max(tiles.count, 1) * e.tileBytes)
            let cmd = GPU.shared.q.makeCommandBuffer()!, bl = cmd.makeBlitCommandEncoder()!
            for (k, t) in tiles.enumerated() { e.copyTile(bl, sf, t % sf.tilesX, t / sf.tilesX, buf, toBuffer: true, offset: k * e.tileBytes) }
            bl.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
            var d = Data(); var n = UInt32(tiles.count); d.append(Data(bytes: &n, count: 4))
            for (k, t) in tiles.enumerated() {
                let z = try! (Data(bytes: buf.contents() + k * e.tileBytes, count: e.tileBytes) as NSData).compressed(using: .lzfse) as Data
                var ti = UInt32(t), ln = UInt32(z.count); d.append(Data(bytes: &ti, count: 4)); d.append(Data(bytes: &ln, count: 4)); d.append(z)
            }
            return d
        }
        var hdr = Engine.Header(history: e.history, tubes: e.tubes.map { $0.left }, crop: [0, 0, Float(e.canvas.w), Float(e.canvas.h)], medium: 0,
                                canvas: [Float(e.canvas.w) / PPM, Float(e.canvas.h) / PPM, 0], palette: nil)
        hdr.version = 1
        var br = Data(); for b in e.brushes { br.append(Data(bytes: b.res.contents(), count: b.res.length)) }
        var old = Data("PAINTSIM".utf8)
        for blk in [try! JSONEncoder().encode(hdr), v1(e.canvas), v1(e.palette), br] { var n = UInt64(blk.count); old.append(Data(bytes: &n, count: 8)); old.append(blk) }
        let e3 = Engine(); try! e3.decode(old)
        let sc3 = Scene(e3, sampleCount: 4, colorFormat: .bgra8Unorm_srgb)
        print("version-1 file loads identically:", px(sc3.export(scale: 0.5)!) == px(a), "history", e3.history.count)
    case "dull":
        // the same strokes wet and fully dried (compare the two renders)
        for (k, c) in [4, 10, 7, 13].enumerated() {
            s.squeeze(c, [V3(P(25, 160 - Float(k) * 22), 1), V3(P(50, 160 - Float(k) * 22), 1)])
            s.stroke(5, 1, [V3(P(30, 160 - Float(k) * 22), 0.8), V3(P(48, 162 - Float(k) * 22), 0.8)], speed: 40)
            s.stroke(5, 0, [V3(-60, 40 - Float(k) * 22, 0.6), V3(60, 40 - Float(k) * 22, 0.6)], speed: 100)
        }
        // light strokes laid over the wet dark ones
        s.squeeze(0, [V3(P(70, 160), 1), V3(P(95, 160), 1)])
        s.run { e.brushes[2].clean($0) }
        for k in 0..<3 {
            s.stroke(2, 1, [V3(P(75, 160), 0.8), V3(P(90, 162), 0.8)], speed: 40)
            s.stroke(2, 0, [V3(-40 + Float(k) * 30, 50, 0.5), V3(-40 + Float(k) * 30, -40, 0.5)], speed: 80)
        }
        render(V2(0, 5), 150, 1200, 900, out.replacingOccurrences(of: ".png", with: "_wet.png"))
        s.run { cmd in e.dry(cmd, hours: 2000, palette: false) }
        render(V2(0, 5), 150, 1200, 900, out)
    case "replay":
        // colours with brush cleans in between, replayed all in one command buffer (as full speed does) and one op per frame
        for (k, c) in [4, 10, 7, 13, 0].enumerated() {
            let y = 160 - Float(k) * 18
            s.squeeze(c, [V3(P(25, y), 1), V3(P(50, y), 1)])
            s.stroke(5, 1, [V3(P(30, y), 0.8), V3(P(48, y + 2), 0.8)], speed: 40)
            s.stroke(5, 0, [V3(-60, 40 - Float(k) * 18, 0.6), V3(60, 40 - Float(k) * 18, 0.6)], speed: 100)
            s.run { e.cleanBrush($0, 5) }
            if k % 2 == 1 { s.stroke(6, 1, [V3(P(30, y), 0.7), V3(P(45, y), 0.7)], speed: 30); s.stroke(6, 0, [V3(-60 + Float(k) * 25, -60, 0.7), V3(-50 + Float(k) * 25, -20, 0.7)], speed: 40) }
        }
        // undoing a clean gives the brush its paint back
        s.stroke(5, 1, [V3(P(30, 160), 0.8), V3(P(48, 162), 0.8)], speed: 40)
        let load = Data(bytes: e.brushes[5].res.contents(), count: e.brushes[5].res.length)
        s.run { e.cleanBrush($0, 5) }; s.run { e.undo($0) }
        print("undo clean restores the brush:", load == Data(bytes: e.brushes[5].res.contents(), count: e.brushes[5].res.length), load.contains { $0 != 0 })
        s.run { e.redo($0) }
        let url = URL(fileURLWithPath: out.replacingOccurrences(of: ".png", with: ".pirl"))
        try! e.save(to: url); HUD.setThumbnail(sc, url, wait: true)
        var r = CGRect(x: 0, y: 0, width: 256, height: 256)
        if let i = NSWorkspace.shared.icon(forFile: url.path).cgImage(forProposedRect: &r, context: nil, hints: nil) { writePNG(i, out.replacingOccurrences(of: ".png", with: "_icon.png")) }
        func px(_ i: CGImage) -> [UInt8] { [UInt8](i.dataProvider!.data! as Data) }
        let ref = px(sc.export(scale: 0.5)!)
        for fast in [true, false] {
            let e2 = Engine(), sc2 = Scene(e2, sampleCount: 4, colorFormat: .bgra8Unorm_srgb)
            var cmd = GPU.shared.q.makeCommandBuffer()!; e2.beginFrame()
            for op in e.history {
                switch op {
                case .stroke(_, let ss), .squeeze(_, let ss):
                    e2.begin(op.empty); for m in ss { e2.add(m) }
                    e2.process(cmd)
                    while e2.backlog > 0 { cmd.commit(); cmd = GPU.shared.q.makeCommandBuffer()!; e2.beginFrame(); e2.process(cmd) }
                    e2.end(cmd)
                case .clean(let b): e2.cleanBrush(cmd, b)
                default: break
                }
                if !fast { cmd.commit(); cmd.waitUntilCompleted(); cmd = GPU.shared.q.makeCommandBuffer()!; e2.beginFrame() }
            }
            cmd.commit(); cmd.waitUntilCompleted()
            let r = px(sc2.export(scale: 0.5)!)
            let bad = zip(ref, r).filter { abs(Int($0) - Int($1)) > 8 }.count
            print(fast ? "full-speed" : "op-per-frame", "replay: bytes off by >8:", bad, "of", ref.count)
        }
    case "v2":
        // an older (LZFSE) file loads, re-saves as the current version and reloads identically
        let e1 = Engine(); try! e1.load(from: URL(fileURLWithPath: out))
        let d = try! e1.encode(), e2 = Engine(); try! e2.decode(d)
        func px(_ e: Engine) -> Data { Scene(e, sampleCount: 4, colorFormat: .bgra8Unorm_srgb).export(scale: 0.5)!.dataProvider!.data! as Data }
        let a = px(e1)
        print("old file: history", e1.history.count, "painted tiles", e1.canvas.painted.filter { $0 }.count, "non-blank", a.contains { $0 < 200 })
        print("re-saved as version", (try! JSONDecoder().decode(Engine.Header.self, from: d.subdata(in: 16..<(16 + Int(d.subdata(in: 8..<16).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }))))).version,
              "and reloads identically:", a == px(e2), Engine.packHistory(e1.history) == Engine.packHistory(e2.history))
    case "savesize":
        // a canvas covered in two layers of paint (the second over a dried first), then the save format measured
        var rng = Rng(s: 7)
        let cols = [0, 7, 4, 10, 5, 2]
        for (k, c) in cols.enumerated() { s.squeeze(c, [V3(P(25, 160 - Float(k) * 22), 1), V3(P(50, 160 - Float(k) * 22), 1)]) }
        func layer(_ brush: Int, _ step: Float, _ diag: Float) {
            var y: Float = -145, k = 0
            while y < 145 {
                let py = 160 - Float(k % cols.count) * 22
                s.stroke(brush, 1, [V3(P(30, py), 0.8), V3(P(48, py + 2), 0.8), V3(P(32, py - 1), 0.8)], speed: 50)
                s.stroke(brush, 0, [V3(-195, y, 0.6), V3(0, y + diag + rng.gauss() * 3, 0.7), V3(195, y + 2 * diag, 0.5)], speed: 150)
                y += step; k += 1
            }
        }
        layer(5, 9, 0)
        s.run { cmd in e.dry(cmd, hours: 48, palette: false) }
        layer(2, 14, 20)
        writePNG(sc.export(scale: 0.25)!, out)
        let t0 = CACurrentMediaTime()
        let url = URL(fileURLWithPath: out.replacingOccurrences(of: ".png", with: ".pirl"))
        try! e.save(to: url)
        print(String(format: "save %.2fs, %d MB, %d painted tiles, history %d ops", CACurrentMediaTime() - t0, (try! Data(contentsOf: url)).count >> 20, e.canvas.painted.filter { $0 }.count, e.history.count))
        e.saveReport()
        s.stroke(2, 0, [V3(-20, 0, 0.6), V3(20, 5, 0.6)], speed: 100)
        var t1 = CACurrentMediaTime()
        let d = try! e.encode(); let te = CACurrentMediaTime() - t1
        try! d.write(to: url, options: .atomic)
        print(String(format: "re-save after one stroke %.3fs (encode %.3fs, write %.3fs)", CACurrentMediaTime() - t1, te, CACurrentMediaTime() - t1 - te))
        t1 = CACurrentMediaTime()
        let e2 = Engine()
        try! e2.load(from: url)
        print(String(format: "load %.2fs, history %d ops, same history bytes: %@", CACurrentMediaTime() - t1, e2.history.count,
                     Engine.packHistory(e2.history) == Engine.packHistory(e.history) ? "yes" : "no"))
        t1 = CACurrentMediaTime()
        try! e2.save(to: url)
        print(String(format: "save right after load %.3fs", CACurrentMediaTime() - t1))
    case "paint":
        var rng = Rng(s: 42)
        // hand-like wobbly stroke between two points
        func hand(_ a: V2, _ b: V2, _ p: Float, _ n: Int = 6) -> [V3] {
            (0...n).map { k in
                let f = Float(k) / Float(n)
                let q = a + (b - a) * f + V2(rng.gauss(), rng.gauss()) * 0.6
                return V3(q.x, q.y, p * (0.85 + 0.3 * sin(f * 3.1)))
            }
        }
        s.squeeze(0, [V3(P(25, 160), 1), V3(P(60, 160), 1)])
        s.squeeze(7, [V3(P(25, 135), 1), V3(P(42, 135), 1)])
        s.squeeze(4, [V3(P(25, 110), 1), V3(P(42, 110), 1)])
        s.squeeze(10, [V3(P(25, 85), 1), V3(P(42, 85), 1)])
        s.squeeze(5, [V3(P(25, 60), 1), V3(P(38, 60), 1)])
        s.squeeze(2, [V3(P(25, 40), 1), V3(P(38, 40), 1)])
        // sky mix: white + a little ultramarine, scrubbed together on the palette
        s.stroke(5, 1, [V3(P(40, 160), 0.9), V3(P(52, 160), 0.9), V3(P(40, 161), 0.9)], speed: 40)
        s.stroke(5, 1, [V3(P(32, 136), 0.6), V3(P(36, 135), 0.6)], speed: 30)
        for k in 0..<4 { s.stroke(5, 1, [V3(P(110, 150 + Float(k)), 0.9), V3(P(150, 140), 0.9), V3(P(110, 135 - Float(k)), 0.9)], speed: 60) }
        // sky: horizontal strokes, lighter toward the horizon
        for row in 0..<9 {
            let y = 80 - Float(row) * 7
            if row % 3 == 2 { s.stroke(5, 1, [V3(P(120, 145), 0.8), V3(P(145, 140), 0.8), V3(P(120, 136), 0.8)], speed: 50) }
            if row == 5 { s.stroke(5, 1, [V3(P(40, 160), 0.7), V3(P(55, 158), 0.7)], speed: 40) }
            s.stroke(5, 0, hand(V2(-112 + rng.next() * 10, y), V2(112, y + rng.gauss() * 2), 0.45), speed: 140)
        }
        // ground: ochre + sienna
        s.stroke(4, 1, [V3(P(32, 112), 0.8), V3(P(38, 108), 0.8), V3(P(30, 86), 0.8), V3(P(38, 84), 0.8)], speed: 40)
        for k in 0..<4 { s.stroke(4, 1, [V3(P(110, 100 + Float(k)), 0.9), V3(P(140, 90), 0.9), V3(P(110, 85), 0.9)], speed: 60) }
        for row in 0..<7 {
            let y = 14 - Float(row) * 13
            if row % 2 == 1 { s.stroke(4, 1, [V3(P(32, 110), 0.8), V3(P(36, 104), 0.8), V3(P(32, 88), 0.6)], speed: 40) }
            s.stroke(4, 0, hand(V2(-115, y), V2(115, y - 3 + rng.gauss() * 3), 0.55), speed: 120)
        }
        // a few red and yellow dabs with the round
        s.stroke(1, 1, [V3(P(30, 60), 0.8), V3(P(34, 62), 0.8)], speed: 30)
        for k in 0..<6 { let c = V2(-60 + Float(k) * 18, -25 + rng.gauss() * 5); s.stroke(1, 0, [V3(c, 0.6), V3(c + V2(3, 1.5), 0.7), V3(c + V2(5, 0), 0.4)], speed: 40) }
        s.stroke(1, 1, [V3(P(30, 40), 0.8), V3(P(34, 42), 0.8)], speed: 30)
        for k in 0..<6 { let c = V2(-50 + Float(k) * 18, -45 + rng.gauss() * 5); s.stroke(1, 0, [V3(c, 0.6), V3(c + V2(3, 1.5), 0.7), V3(c + V2(5, 0), 0.4)], speed: 40) }
        writePNG(sc.export(scale: 0.6)!, out)
        render(V2(-40, 10), 120, 1400, 1000, out.replacingOccurrences(of: ".png", with: "_zoom.png"))
        render(V2(P(70, 110).x, P(70, 110).y), 330, 1400, 1000, out.replacingOccurrences(of: ".png", with: "_pal.png"))
    case "hover":
        s.squeeze(7, [V3(P(30, 120), 1), V3(P(50, 118), 1)])
        s.stroke(4, 1, [V3(P(36, 112), 0.8), V3(P(40, 125), 0.9)], speed: 40)
        s.stroke(4, 0, [V3(-60, 20, 0.6), V3(0, 25, 0.6)], speed: 100)
        sc.setHeld(.brush(4))
        let b = e.brushes[4]
        b.baseZ = TABLE_Z
        b.pose(target: V2(60, -170), pressure: 0, tilt: V2(0.3, -0.3), roll: 0.4)
        var bb = IRect.none
        _ = b.simulate(surface: nil, moved: 0, out: nil, bbox: &bb)
        render(V2(30, -110), 260, 1400, 1000, out)
    case "canvas":
        s.run { cmd in e.newCanvas(cmd, 500, 400, texture: 2) }
        sc.setHeld(.brush(5))
        let b = e.brushes[5]
        b.baseZ = 0
        b.pose(target: V2(190, -120), pressure: 0, tilt: V2(0.3, -0.3), roll: 0)
        var bb = IRect.none
        _ = b.simulate(surface: nil, moved: 0, out: nil, bbox: &bb)
        sc.fitDesk(1.6)
        render(sc.cam.center, sc.cam.dist, 1600, 1000, out)
        render(V2(215, -160), 160, 1400, 1000, out.replacingOccurrences(of: ".png", with: "_shadow.png"))
    case "detail":
        s.run { cmd in e.newCanvas(cmd, 800, 600, texture: 0) }
        s.squeeze(14, [V3(P(30, 120), 1), V3(P(46, 118), 1)])
        s.squeeze(15, [V3(P(30, 90), 1), V3(P(46, 92), 1)])
        s.stroke(7, 1, [V3(P(36, 119), 0.5), V3(P(40, 121), 0.5)], speed: 20)
        let t0 = CACurrentMediaTime()
        for k in 0..<5 { s.stroke(7, 0, [V3(100, 100 + Float(k) * 3, 0.15 + Float(k) * 0.12), V3(140, 102 + Float(k) * 3, 0.15 + Float(k) * 0.12)], speed: 60) }
        s.stroke(8, 1, [V3(P(36, 91), 0.5), V3(P(42, 91), 0.6), V3(P(36, 92), 0.5)], speed: 20)
        s.stroke(8, 0, [V3(100, 80, 0.3), V3(120, 85, 0.35), V3(140, 76, 0.3), V3(160, 82, 0.2)], speed: 80)
        print("detail strokes on 80x60 canvas:", CACurrentMediaTime() - t0, "s")
        render(V2(130, 92), 70, 1400, 1000, out)
        sc.fitDesk(1.6)
        render(sc.cam.center, sc.cam.dist, 1600, 1000, out.replacingOccurrences(of: ".png", with: "_desk.png"))
    case "pencil":
        // sketch lines at rising pressure, a scribble, then thin and thick paint over them
        for k in 0..<5 { s.stroke(9, 0, [V3(-150, 100 - Float(k) * 8, 0.15 + Float(k) * 0.2), V3(-60, 104 - Float(k) * 8, 0.15 + Float(k) * 0.2)], speed: 120) }
        s.stroke(9, 0, (0...40).map { k in let a = Float(k) * 0.3; return V3(-100 + cos(a) * (14 + Float(k) * 0.5), 20 + sin(a) * (14 + Float(k) * 0.5), 0.5) }, speed: 150)
        s.squeeze(7, [V3(P(30, 120), 1), V3(P(50, 118), 1)])
        s.stroke(4, 1, [V3(P(36, 112), 0.8), V3(P(40, 125), 0.9), V3(P(46, 112), 0.9)], speed: 40)
        s.stroke(4, 0, [V3(-160, 70, 0.2), V3(-40, 72, 0.2)], speed: 80)
        // move the palette over the canvas, squeeze and pick up there
        s.run { _ in e.movePalette(V2(-60, -120)) }
        print("palette at", e.palette.origin, "z", e.palette.z)
        let q = e.palette.origin
        s.squeeze(0, [V3(q + V2(80, 150), 1), V3(q + V2(105, 150), 1)])
        s.stroke(5, 1, [V3(q + V2(85, 152), 0.8), V3(q + V2(100, 149), 0.8)], speed: 40)
        s.stroke(5, 0, [V3(-160, 40, 0.6), V3(-40, 45, 0.6)], speed: 200)
        render(V2(-95, 55), 260, 1400, 1000, out)
        render(V2(-120, 90), 40, 1400, 1000, out.replacingOccurrences(of: ".png", with: "_close.png"))
        sc.fitDesk(1.6)
        render(sc.cam.center, sc.cam.dist, 1600, 1000, out.replacingOccurrences(of: ".png", with: "_desk.png"))
        s.run { cmd in e.undo(cmd); e.undo(cmd); e.undo(cmd); e.undo(cmd) }
        print("after undoing the move: palette at", e.palette.origin, "z", e.palette.z)
        s.run { cmd in e.redo(cmd) }
        print("after redo: palette at", e.palette.origin, "history", e.history.count)
        print("tube 7 left", e.tubes[7].left)
    case "circles":
        // small quick circles (mouse-like 120 Hz samples), round and flat
        s.squeeze(7, [V3(P(30, 120), 1), V3(P(50, 118), 1)])
        for (k, bi) in [1, 4].enumerated() {
            s.stroke(bi, 1, [V3(P(36, 112), 0.8), V3(P(40, 125), 0.9), V3(P(46, 112), 0.9)], speed: 40)
            let c = V2(-60 + Float(k) * 50, 0)
            s.stroke(bi, 0, (0...60).map { i in let a = Float(i) * 0.45; return V3(c + V2(cos(a), sin(a)) * 9, 0.6) }, speed: 400)
        }
        render(V2(-35, 0), 110, 1400, 1000, out)
        // pencil sketch, paint over it, dry: the sketch must stay under the dried paint
        s.stroke(9, 0, [V3(-80, -40, 0.9), V3(10, -40, 0.9)], speed: 120)
        s.stroke(1, 0, [V3(-70, -45, 0.6), V3(0, -35, 0.6)], speed: 80)
        s.run { cmd in e.dry(cmd, hours: 500, palette: false) }
        s.stroke(9, 0, [V3(-80, -30, 0.9), V3(10, -50, 0.9)], speed: 120)
        render(V2(-35, -40), 110, 1400, 1000, out.replacingOccurrences(of: ".png", with: "_dry.png"))
    case "scene":
        s.squeeze(0, [V3(P(30, 150), 1), V3(P(55, 150), 1)])
        let hp = e.readHeights(e.palette, Int(42 * PPM), Int(140 * PPM), 1, 200)
        print("bead x-section", stride(from: 0, to: 200, by: 5).map { String(format: "%.0f", hp[$0]) }.joined(separator: " "))
        render(V2(90, -45), 980, 1800, 1200, out)
    case "brush":
        s.squeeze(7, [V3(P(30, 120), 1), V3(P(50, 118), 1)])
        s.squeeze(0, [V3(P(30, 150), 1), V3(P(55, 150), 1)])
        s.stroke(2, 1, [V3(P(36, 112), 0.8), V3(P(40, 125), 0.9), V3(P(46, 112), 0.9)], speed: 40)
        s.stroke(2, 0, [V3(-60, 20, 0.5), V3(0, 25, 0.5)], speed: 100)
        // hold the brush pressed mid-stroke for the picture
        let b = e.brushes[2]
        sc.setHeld(.brush(2))
        e.begin(.stroke(brush: 2, samples: []))
        s.run { cmd in
            for k in 0..<40 { e.add(Sample(x: 10 + Float(k) * 0.8, y: 0, p: 0.6, tx: 0.3, ty: -0.3, roll: 0, t: s.t + Float(k) / 120, s: 0)) }
            e.process(cmd)
        }
        _ = b
        render(V2(30, 5), 140, 1400, 1000, out)
        sc.cam.center = V2(P(45, 105).x, P(45, 105).y)
        render(V2(P(45, 130).x, P(45, 130).y), 110, 1400, 1000, out.replacingOccurrences(of: ".png", with: "_pal.png"))
    default:
        print("unknown test")
    }
}

// ---------------------------------------------------------------- interactive view
final class PaintView: MTKView {
    var eng: Engine!
    var sc: Scene!
    var sema = DispatchSemaphore(value: 3)
    var t0 = CACurrentMediaTime()
    var mouseDown = false, ending = false, panning = false, spaceDown = false
    var palDrag: (start: V2, origin: V2, z: Float)?       // dragging the palette by its thumb hole
    var lastPt = NSPoint.zero
    var pressure: Float = 0.5, tilt = V2(0.32, -0.32), roll: Float = 0
    var rollMode = 0 { didSet { status() } }     // flats: 0 follow the stroke, 1 follow edgewise, 2 fixed angle ([ ])
    var lastMove: V2?
    var mousePressure: Float = 0.5
    var tabletSeen = false
    var curSample: Sample?
    var lastAdd: Float = 0
    var surfIdx: Int8 = -1
    var downTime: Float = 0
    var dryHours: Float = 24
    var dryPalette = false
    var previewDirty = false
    var dryPreview = false { didSet { sc.heat = dryPreview; previewDirty = true } }
    struct Replay { var ops: [Op]; var i = 0, k = 0; var started = false; var t0: Float; var speed = 1; var snapshot: Data?; var edited = false }
    static let replaySpeeds: [Float] = [1, 2, 4, 8, 16, 32, .infinity]
    var replay: Replay?
    var cropMode = false { didSet { status() } }
    var cropStart: V2?
    var cropSaved = V4.zero
    var onStatus: ((String) -> Void)?
    var frameMS: Double = 0
    weak var hud: HUD?

    override var acceptsFirstResponder: Bool { true }
    var now: Float { Float(CACurrentMediaTime() - t0) }

    func setup() {
        eng = Engine()
        sc = Scene(eng, sampleCount: 4, colorFormat: .bgra8Unorm_srgb)
        device = GPU.shared.dev
        colorPixelFormat = .bgra8Unorm_srgb; depthStencilPixelFormat = .depth32Float; sampleCount = 4
        clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1)
        preferredFramesPerSecond = 120
        delegate = self
        NSEvent.isMouseCoalescingEnabled = false
        let ta = NSTrackingArea(rect: .zero, options: [.mouseMoved, .activeAlways, .inVisibleRect], owner: self, userInfo: nil)
        addTrackingArea(ta)
        status()
    }

    // ------------------------------------------------ coordinates
    var aspect: Float { Float(bounds.width / max(bounds.height, 1)) }
    func ray(_ p: NSPoint) -> (V3, V3) { sc.cam.ray(V2(Float(p.x / bounds.width) * 2 - 1, Float(p.y / bounds.height) * 2 - 1), aspect) }
    func locate(_ p: NSPoint, margin: Float = 0) -> (V2, Int8, Float) {
        let r = ray(p)
        let pq = hitPlane(r, eng.palette.z)
        if eng.palette.z > eng.canvas.z && eng.palette.contains(pq) { return (pq, 1, eng.palette.z) }     // held above the canvas
        let c = hitPlane(r, eng.canvas.z)
        if eng.canvas.contains(c) { return (c, 0, eng.canvas.z) }
        if eng.palette.contains(pq) { return (pq, 1, eng.palette.z) }
        // just off an edge: stay on that surface while the bristles still reach it
        if eng.palette.z > eng.canvas.z && eng.palette.near(pq, margin) { return (pq, 1, eng.palette.z) }
        if eng.canvas.near(c, margin) { return (c, 0, eng.canvas.z) }
        if eng.palette.near(pq, margin) { return (pq, 1, eng.palette.z) }
        return (hitPlane(r, TABLE_Z), -1, TABLE_Z)
    }
    var reach: Float {      // how far off an edge the held brush still paints
        guard case .brush(let i) = sc.held else { return 0 }
        let b = eng.brushes[i]
        return b.pencil ? 0.3 : 0.45 * max(b.width, b.thick) + 0.2 * b.length
    }
    func sample(_ p: NSPoint) -> Sample {
        let (w, s, z) = locate(p, margin: reach)
        surfIdx = s
        sc.cursor = V3(w.x, w.y, z)
        return Sample(x: w.x, y: w.y, p: pressure, tx: tilt.x, ty: tilt.y, roll: roll, t: now, s: s)
    }

    // ------------------------------------------------ input
    func readPen(_ ev: NSEvent, down: Bool) {
        if ev.subtype == .tabletPoint {
            tabletSeen = true
            pressure = ev.pressure
            let tl = V2(Float(ev.tilt.x), Float(-ev.tilt.y))
            tilt = simd_length(tl) > 0.05 ? tl * 0.9 : V2(0.32, -0.32)
            if ev.rotation != 0 { roll = -Float(ev.rotation) * .pi / 180 }
        } else {
            // mouse: ease pressure in over the first ~60 ms like a hand settling the brush
            let held = now - downTime
            pressure = down ? mousePressure * min(1, 0.35 + held / 0.06) : 0
        }
    }
    override func mouseDown(with ev: NSEvent) {
        let p = convert(ev.locationInWindow, from: nil)
        lastPt = p
        if spaceDown || replay != nil { panning = spaceDown; return }
        if cropMode {
            let t = eng.canvas.toTexel(hitPlane(ray(p), eng.canvas.z))
            cropStart = t; cropSaved = eng.canvas.mask
            return
        }
        let pal = eng.palette, pq = hitPlane(ray(p), pal.z)
        if simd_distance(pal.toTexel(pq), V2(pal.hole.x, pal.hole.y)) < pal.hole.z {
            palDrag = (pq, pal.origin, pal.z); NSCursor.closedHand.push(); return
        }
        if !pal.contains(pq), let h = sc.pick(ray(p)) { choose(h); return }
        if sc.held == .none { return }
        downTime = now
        readPen(ev, down: true)
        let s = sample(p)
        mouseDown = true
        switch sc.held {
        case .brush(let i): eng.begin(.stroke(brush: i, samples: []))
        case .tube(let i): eng.begin(.squeeze(tube: i, samples: []))
        case .none: return
        }
        eng.add(s); lastAdd = s.t
        lastMove = V2(s.x, s.y)
    }
    override func mouseDragged(with ev: NSEvent) {
        let p = convert(ev.locationInWindow, from: nil)
        if panning { pan(p); return }
        if let d = palDrag { eng.placePalette(d.origin + hitPlane(ray(p), d.z) - d.start); return }
        if cropMode, let a = cropStart {
            let b = eng.canvas.toTexel(hitPlane(ray(p), eng.canvas.z))
            eng.canvas.mask = cropRect(a, b)
            eng.canvas.dirty = IRect(x0: 0, y0: 0, x1: eng.canvas.w, y1: eng.canvas.h)
            return
        }
        guard mouseDown else { return }
        readPen(ev, down: true)
        var s = sample(p)
        if let l = lastMove, rollMode < 2, case .brush(let i) = sc.held, !eng.brushes[i].round {
            let d = V2(s.x, s.y) - l
            if simd_length(d) > 0.05 { roll = Engine.followRoll(roll, dir: d, edge: rollMode == 1); s.roll = roll }
        }
        lastMove = V2(s.x, s.y)
        eng.add(s); lastAdd = s.t; curSample = s
    }
    func cropRect(_ a: V2, _ b: V2) -> V4 {
        let c = eng.canvas
        let lo = simd_clamp(simd_min(a, b), .zero, V2(Float(c.w), Float(c.h))), hi = simd_clamp(simd_max(a, b), .zero, V2(Float(c.w), Float(c.h)))
        return V4(lo.x.rounded(), lo.y.rounded(), hi.x.rounded(), hi.y.rounded())
    }
    override func mouseUp(with ev: NSEvent) {
        if panning { panning = false; return }
        if let d = palDrag {
            let o = eng.palette.origin
            eng.placePalette(d.origin)
            if o != d.origin { eng.movePalette(o) }
            palDrag = nil; NSCursor.pop(); return
        }
        if cropMode, let a = cropStart {
            let b = eng.canvas.toTexel(hitPlane(ray(convert(ev.locationInWindow, from: nil)), eng.canvas.z))
            var r = cropRect(a, b)
            eng.canvas.mask = cropSaved
            if r.z - r.x < 50 || r.w - r.y < 50 { r = V4(0, 0, Float(eng.canvas.w), Float(eng.canvas.h)) }   // a click resets the crop
            if r != cropSaved { eng.crop(r) }
            cropStart = nil; cropMode = false
            return
        }
        if mouseDown {
            mouseDown = false; ending = true
            if !tabletSeen, let c = eng.cur, case .stroke(_, let ss) = c.op { for m in Engine.liftSamples(ss) { eng.add(m) } }
        }
    }
    override func mouseMoved(with ev: NSEvent) {
        let p = convert(ev.locationInWindow, from: nil)
        if ev.subtype == .tabletPoint { readPen(ev, down: false) }
        _ = sample(p)
        lastPt = p
    }
    override func tabletPoint(with ev: NSEvent) { readPen(ev, down: mouseDown) }
    override func rightMouseDown(with ev: NSEvent) { lastPt = convert(ev.locationInWindow, from: nil); panning = true }
    override func rightMouseDragged(with ev: NSEvent) { pan(convert(ev.locationInWindow, from: nil)) }
    override func rightMouseUp(with ev: NSEvent) { panning = false }
    func pan(_ p: NSPoint) {
        let a = hitPlane(ray(lastPt), 0), b = hitPlane(ray(p), 0)
        sc.cam.center -= b - a
        lastPt = p
    }
    override func scrollWheel(with ev: NSEvent) {
        let p = convert(ev.locationInWindow, from: nil)
        if ev.hasPreciseScrollingDeltas && ev.momentumPhase == [] && ev.phase != [] || ev.hasPreciseScrollingDeltas && ev.momentumPhase != [] {
            let k = sc.cam.dist / Float(bounds.height) * 0.45
            sc.cam.center += V2(-Float(ev.scrollingDeltaX), Float(ev.scrollingDeltaY)) * k
        } else { zoom(by: Float(1 - ev.scrollingDeltaY * 0.04), at: p) }
    }
    override func magnify(with ev: NSEvent) { zoom(by: Float(1 - ev.magnification), at: convert(ev.locationInWindow, from: nil)) }
    func zoom(by f: Float, at p: NSPoint) {
        let before = hitPlane(ray(p), 0)
        sc.cam.dist = min(max(sc.cam.dist * f, 40), 2200)
        let after = hitPlane(ray(p), 0)
        sc.cam.center += before - after
    }
    func fit() { sc.fitDesk(aspect) }
    func fitCanvas() { sc.fitCanvas(aspect) }
    var fitted = false

    override func keyDown(with ev: NSEvent) {
        guard let ch = ev.charactersIgnoringModifiers else { return }
        if ev.keyCode == 49 { spaceDown = true; return }
        let cmd = ev.modifierFlags.contains(.command)
        if replay != nil {
            switch ev.keyCode {
            case 53: stopReplay()
            case 123, 125: replaySpeed(-1)
            case 124, 126: replaySpeed(1)
            default:
                switch ch { case "-": replaySpeed(-1); case "=": replaySpeed(1); case "f": fit(); case "g": fitCanvas(); default: break }
            }
            return
        }
        switch ch {
        case "z" where cmd, "Z" where cmd: if ev.modifierFlags.contains(.shift) { doRedo() } else { doUndo() }
        case "1", "2", "3", "4", "5", "6", "7", "8", "9", "0": choose(.brush(BRUSH_ORDER[(Int(ch)! + 9) % 10]))
        case "[": roll += .pi / 12; rollMode = 2
        case "]": roll -= .pi / 12; rollMode = 2
        case "\\": rollMode = (rollMode + 1) % 3
        case "-": mousePressure = max(0.1, mousePressure - 0.1)
        case "=": mousePressure = min(1, mousePressure + 0.1)
        case "h": sc.hideMode = (sc.hideMode + 1) % 3
        case "c": let i = eng.active; pendingActions.append { [unowned self] c in eng.cleanBrush(c, i) }
        case "f": fit()
        case "g": fitCanvas()
        case "d": dryPreview.toggle()
        default:
            if ev.keyCode == 53 { if !dryPreview && !cropMode { choose(.none) }; dryPreview = false; cropMode = false }
            else { super.keyDown(with: ev) }
        }
        status()
    }
    override func keyUp(with ev: NSEvent) { if ev.keyCode == 49 { spaceDown = false } }

    func choose(_ h: Held) {
        sc.setHeld(h)
        if case .brush(let i) = h { eng.active = i }
        status()
    }
    func status() {
        var s: String
        switch sc.held {
        case .brush(let i):
            let b = eng.brushes[i]
            s = b.name + (tabletSeen ? "" : String(format: "   ·   mouse pressure %.1f (−/=)", mousePressure))
            if !b.round { s += "   ·   " + ["follows the stroke", "edgewise", "fixed angle"][rollMode] + " (\\)" }
        case .tube(let i): s = eng.tubes[i].name + " — drag on the palette to squeeze; click its slot or Esc to put it back"
        case .none: s = "Pick up a brush, the knife, the pencil or a tube"
        }
        if sc.hideMode > 0 { s += sc.hideMode == 1 ? "   ·   handle hidden (H)" : "   ·   brush hidden (H)" }
        if dryPreview { s = "Drying preview — blue: still wet, yellow: tacky, red: dry" }
        if cropMode { s = "Drag a rectangle on the canvas to crop (click to reset), Esc to cancel" }
        if let r = replay {
            let v = PaintView.replaySpeeds[r.speed]
            s = "Replaying " + (v.isInfinite ? "at full speed" : "×\(Int(v))") + "   ·   ←/→ speed, Esc to stop"
        }
        onStatus?(s)
    }

    // ------------------------------------------------ actions (queued into the next frame's command buffer)
    var pendingActions: [(MTLCommandBuffer) -> Void] = []
    func doUndo() { pendingActions.append { [unowned self] c in eng.undo(c) } }
    func doRedo() { pendingActions.append { [unowned self] c in eng.redo(c) } }
    func doDry() { let h = dryHours, p = dryPalette; pendingActions.append { [unowned self] c in eng.dry(c, hours: h, palette: p) }; dryPreview = false }
    func doScrape() { pendingActions.append { [unowned self] c in eng.clearSurface(c, palette: true) } }
    func doWipe() { pendingActions.append { [unowned self] c in eng.clearSurface(c, palette: false) } }
    func startReplay() {
        if replay != nil { stopReplay(); return }
        let ops = eng.history
        guard !ops.isEmpty, cropMode == false else { return }
        let snap = try? eng.encode()        // the finished painting, restored when the replay is stopped
        pendingActions.append { [unowned self] c in
            eng.resetAll(c); eng.undoStack.removeAll(); eng.redoStack.removeAll(); eng.history.removeAll(); eng.redoOps.removeAll()
            replay = Replay(ops: ops, t0: now, snapshot: snap, edited: window?.isDocumentEdited ?? false)
            status()
        }
    }
    func stopReplay() {
        guard let r = replay else { return }
        replay = nil
        eng.cur = nil; eng.undoRec = nil
        if let d = r.snapshot { do { try eng.decode(d) } catch { NSAlert(error: error).runModal() } }
        window?.isDocumentEdited = r.edited
        sc.setHeld(.brush(eng.active)); status()
    }
    func replaySpeed(_ d: Int) {
        guard var r = replay else { return }
        r.speed = min(max(r.speed + d, 0), PaintView.replaySpeeds.count - 1)
        replay = r; status()
    }

    func stepReplay(_ cmd: MTLCommandBuffer) {
        guard var r = replay else { return }
        let speed = PaintView.replaySpeeds[r.speed], fast = speed.isInfinite, t0 = CACurrentMediaTime()
        func next() { r.i += 1; r.k = 0; r.started = false; r.t0 = now }
        // at full speed, run as many ops as fit in ~30 ms per frame
        while r.i < r.ops.count {
            let op = r.ops[r.i]
            switch op {
            case .stroke(let b, let ss), .squeeze(let b, let ss):
                if !r.started {
                    eng.begin(op.empty)
                    if case .stroke = op { sc.setHeld(.brush(b)); eng.active = b } else { sc.setHeld(.tube(b)) }
                    r.started = true; r.t0 = now
                }
                let base = ss.first?.t ?? 0, clock = fast ? Float.infinity : (now - r.t0) * speed
                while r.k < ss.count && ss[r.k].t - base <= clock {
                    eng.add(ss[r.k]); sc.cursor = V3(ss[r.k].x, ss[r.k].y, ss[r.k].s == 1 ? eng.palette.z : 0); r.k += 1
                }
                if fast { eng.process(cmd) }
                if r.k >= ss.count && eng.backlog == 0 { eng.end(cmd); next() }
            case .dry(let h, let p): eng.dry(cmd, hours: h, palette: p); next()
            case .clean(let b): eng.cleanBrush(cmd, b); next()
            case .scrape: eng.clearSurface(cmd, palette: true); next()
            case .wipe: eng.clearSurface(cmd, palette: false); next()
            case .crop(let rc): eng.crop(V4(rc[0], rc[1], rc[2], rc[3])); next()
            case .movePalette(let x, let y): eng.movePalette(V2(x, y)); next()
            }
            if !fast || eng.backlog > 0 || CACurrentMediaTime() - t0 > 0.03 { break }
        }
        replay = r
        if r.i >= r.ops.count {
            replay = nil; window?.isDocumentEdited = r.edited
            if case .tube = sc.held { sc.setHeld(.brush(eng.active)) }
            status()
        }
    }
}

extension PaintView: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    func draw(in view: MTKView) {
        guard let rp = currentRenderPassDescriptor, let drawable = currentDrawable else { return }
        if !fitted && bounds.width > 0 { fit(); fitted = true }
        sema.wait()
        eng.beginFrame()
        let cmd = GPU.shared.q.makeCommandBuffer()!
        sc.time = now
        for a in pendingActions { a(cmd) }
        pendingActions.removeAll()
        stepReplay(cmd)
        osc?.step(cmd)
        if mouseDown, now - lastAdd > 0.012, var s = curSample ?? Optional(sample(lastPt)) {
            // keep the brush working while held still
            s.t = now; s.p = pressure
            if !tabletSeen { let held = now - downTime; s.p = mousePressure * min(1, 0.35 + held / 0.06) }
            eng.add(s); lastAdd = s.t
        }
        eng.process(cmd)
        if ending { eng.end(cmd); ending = false; curSample = nil }
        if eng.cur == nil, case .brush(let i) = sc.held {
            let b = eng.brushes[i]
            b.baseZ = sc.cursor.z
            b.pose(target: V2(sc.cursor.x, sc.cursor.y), pressure: 0, tilt: tilt, roll: roll)
            var bb = IRect.none
            _ = b.simulate(surface: nil, moved: 0, out: nil, bbox: &bb)
        }
        if dryPreview && previewDirty { eng.preview(cmd, hours: dryHours, palette: dryPalette); previewDirty = false }
        eng.shade(cmd)
        sc.encode(cmd, rp, aspect: aspect)
        cmd.present(drawable)
        let s = sema
        let t0 = CACurrentMediaTime()
        cmd.addCompletedHandler { [weak self] c in
            s.signal()
            if ProcessInfo.processInfo.environment["PAINT_TIMING"] != nil { self?.frameMS = (c.gpuEndTime - c.gpuStartTime) * 1000; _ = t0 }
        }
        cmd.commit()
    }
}

// ---------------------------------------------------------------- HUD
final class HUD: NSObject {
    let view: NSVisualEffectView
    let label = NSTextField(labelWithString: "")
    weak var pv: PaintView?
    let drySlider = NSSlider(value: 24, minValue: 0, maxValue: 1, target: nil, action: nil)
    let dryLabel = NSTextField(labelWithString: "")
    let dryRow = NSStackView()

    init(_ pv: PaintView) {
        self.pv = pv
        view = NSVisualEffectView()
        view.material = .hudWindow; view.blendingMode = .withinWindow; view.state = .active
        view.wantsLayer = true; view.layer?.cornerRadius = 9
        super.init()
        let stack = NSStackView(); stack.orientation = .vertical; stack.alignment = .leading; stack.spacing = 6
        stack.edgeInsets = NSEdgeInsets(top: 8, left: 10, bottom: 8, right: 14)
        func btn(_ sym: String, _ tip: String, _ sel: Selector) -> NSButton {
            let b = NSButton(image: NSImage(systemSymbolName: sym, accessibilityDescription: tip)!, target: self, action: sel)
            b.bezelStyle = .texturedRounded; b.isBordered = false; b.toolTip = tip
            b.contentTintColor = .labelColor
            return b
        }
        let row = NSStackView(views: [
            btn("arrow.uturn.backward", "Undo (⌘Z)", #selector(undo)), btn("arrow.uturn.forward", "Redo (⇧⌘Z)", #selector(redo)),
            btn("sun.max", "Dry… (D)", #selector(dry)), btn("drop", "Clean brush (C)", #selector(clean)),
            btn("eraser", "Scrape palette", #selector(scrape)), btn("paintbrush.pointed", "Show/hide handle (H)", #selector(handle)),
            btn("crop", "Crop canvas", #selector(crop)), btn("doc.badge.plus", "New canvas…", #selector(newCanvas(_:))),
            btn("play", "Replay painting (click again or Esc to stop)", #selector(replay)), btn("square.and.arrow.down", "Save (⌘S) · Save As (⇧⌘S)", #selector(save)),
            btn("folder", "Open (⌘O)", #selector(open)), btn("photo", "Export PNG (⌘E)", #selector(export)),
            btn("antenna.radiowaves.left.and.right", "OSC remote control", #selector(toggleOSC(_:))),
        ])
        row.spacing = 4
        drySlider.target = self; drySlider.action = #selector(dryChanged)
        drySlider.doubleValue = 0.5
        drySlider.widthAnchor.constraint(equalToConstant: 180).isActive = true
        let go = NSButton(title: "Dry", target: self, action: #selector(dryApply)); go.bezelStyle = .rounded; go.controlSize = .small
        let palToggle = NSButton(checkboxWithTitle: "palette too", target: self, action: #selector(dryPal))
        palToggle.controlSize = .small
        let dullToggle = NSButton(checkboxWithTitle: "dulling", target: self, action: #selector(dryDull))
        dullToggle.controlSize = .small; dullToggle.state = .on; dullToggle.toolTip = "Dried paint loses a little of its gloss"
        dryRow.setViews([drySlider, dryLabel, palToggle, dullToggle, go], in: .leading)
        dryRow.spacing = 8; dryRow.isHidden = true
        label.textColor = .secondaryLabelColor; label.font = .systemFont(ofSize: 11)
        stack.addArrangedSubview(row); stack.addArrangedSubview(dryRow); stack.addArrangedSubview(label)
        stack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stack)
        NSLayoutConstraint.activate([stack.leadingAnchor.constraint(equalTo: view.leadingAnchor), stack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -4),
                                     stack.topAnchor.constraint(equalTo: view.topAnchor), stack.bottomAnchor.constraint(equalTo: view.bottomAnchor)])
        dryChanged()
    }
    func hours(_ v: Double) -> Float { Float(pow(v, 2.2) * 240) }
    @objc func undo() { pv?.doUndo() }
    @objc func redo() { pv?.doRedo() }
    @objc func dry() { guard let pv = pv else { return }; pv.dryPreview.toggle(); dryRow.isHidden = !pv.dryPreview; pv.status() }
    @objc func dryChanged() {
        let h = hours(drySlider.doubleValue)
        dryLabel.stringValue = h < 1 ? "\(Int(h * 60)) min" : (h < 48 ? String(format: "%.0f h", h) : String(format: "%.1f days", h / 24))
        pv?.dryHours = h; pv?.previewDirty = true; pv?.status()
    }
    @objc func dryPal(_ s: NSButton) { pv?.dryPalette = s.state == .on; pv?.previewDirty = true }
    @objc func dryDull(_ s: NSButton) { pv?.eng.dullDry = s.state == .on }
    @objc func dryApply() { pv?.doDry(); dryRow.isHidden = true; pv?.status() }
    @objc func clean() { guard let pv = pv else { return }; let i = pv.eng.active; pv.pendingActions.append { c in pv.eng.cleanBrush(c, i) } }
    @objc func scrape() { pv?.doScrape() }
    @objc func handle() { guard let pv = pv else { return }; pv.sc.hideMode = (pv.sc.hideMode + 1) % 3; pv.status() }
    @objc func replay() { pv?.startReplay() }
    @objc func crop() { pv?.cropMode.toggle() }
    @objc func wipe() { pv?.doWipe() }
    var portrait = false, texture = 0
    @objc func newCanvas(_ sender: NSButton) {
        let m = NSMenu()
        for (w, h) in [(24, 18), (30, 24), (40, 30), (50, 40), (60, 50), (80, 60)] {
            let (a, b) = portrait ? (h, w) : (w, h)
            let it = NSMenuItem(title: "\(a) × \(b) cm", action: #selector(pickSize(_:)), keyEquivalent: "")
            it.target = self; it.representedObject = [a, b]; m.addItem(it)
        }
        m.addItem(.separator())
        let o = NSMenuItem(title: "Portrait", action: #selector(togglePortrait), keyEquivalent: ""); o.target = self; o.state = portrait ? .on : .off; m.addItem(o)
        for (i, name) in [(0, "Cotton canvas"), (2, "Fine linen"), (3, "Smooth gessoed panel")] {
            let it = NSMenuItem(title: name, action: #selector(pickTexture(_:)), keyEquivalent: ""); it.target = self; it.tag = i
            it.state = texture == i ? .on : .off; m.addItem(it)
        }
        m.addItem(.separator())
        let w = NSMenuItem(title: "Wipe this canvas", action: #selector(wipe), keyEquivalent: ""); w.target = self; m.addItem(w)
        m.popUp(positioning: nil, at: NSPoint(x: 0, y: sender.bounds.height + 4), in: sender)
    }
    @objc func togglePortrait() { portrait.toggle() }
    @objc func pickTexture(_ it: NSMenuItem) { texture = it.tag }
    @objc func pickSize(_ it: NSMenuItem) {
        guard let pv = pv, let wh = it.representedObject as? [Int] else { return }
        if !pv.eng.history.isEmpty {
            let a = NSAlert(); a.messageText = "Start a new painting?"; a.informativeText = "The current painting and its history will be discarded. Save it first if you want to keep it."
            a.addButton(withTitle: "New Painting"); a.addButton(withTitle: "Cancel")
            if a.runModal() != .alertFirstButtonReturn { return }
        }
        let tex = texture
        pv.pendingActions.append { c in pv.eng.newCanvas(c, Float(wh[0]) * 10, Float(wh[1]) * 10, texture: tex); pv.fit() }
        docURL = nil; pv.window?.isDocumentEdited = false
    }
    var docURL: URL? { didSet { pv?.window?.title = docURL?.deletingPathExtension().lastPathComponent ?? "Untitled" } }
    // .pirl is the default; .paintirl and the older .paint open too (same format)
    static let docTypes = ["pirl", "paintirl", "paint"].compactMap { UTType(filenameExtension: $0) }
    // ⌘S writes straight to the open file (only tiles changed since the last save are re-encoded); ⇧⌘S picks a new one
    @objc func save() { if let u = docURL { write(u) } else { saveAs() } }
    @objc func saveAs() {
        let p = NSSavePanel(); p.allowedContentTypes = HUD.docTypes
        p.nameFieldStringValue = (docURL?.deletingPathExtension().lastPathComponent ?? "Painting") + ".pirl"
        if p.runModal() == .OK, let url = p.url { write(url) }
    }
    func write(_ url: URL) {
        guard let pv = pv, pv.replay == nil else { return }
        do { try pv.eng.save(to: url); docURL = url; pv.window?.isDocumentEdited = false } catch { NSAlert(error: error).runModal(); return }
        HUD.setThumbnail(pv.sc, url)
    }
    // Finder shows the file with a small preview of the painting as its icon
    static let iconQueue = DispatchQueue(label: "icons", qos: .utility)
    static func setThumbnail(_ sc: Scene, _ url: URL, wait: Bool = false) {
        let m = sc.eng.canvas.mask, side = 64     // just enough to tell files apart at a glance
        guard let img = sc.export(scale: min(1, Float(side) / max(m.z - m.x, m.w - m.y))),
              let ctx = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
        let w = CGFloat(img.width), h = CGFloat(img.height), k = CGFloat(side) / max(w, h)
        ctx.draw(img, in: CGRect(x: (CGFloat(side) - w * k) / 2, y: (CGFloat(side) - h * k) / 2, width: w * k, height: h * k))
        guard let sq = ctx.makeImage() else { return }
        // one small bitmap keeps the icon file (and the time to write it) small; done off the main thread
        let icon = NSImage(size: NSSize(width: side, height: side)); icon.addRepresentation(NSBitmapImageRep(cgImage: sq))
        let job = { _ = NSWorkspace.shared.setIcon(icon, forFile: url.path) }
        if wait { iconQueue.sync(execute: job) } else { iconQueue.async(execute: job) }
    }
    @objc func open() {
        guard let pv = pv, pv.replay == nil else { return }
        let p = NSOpenPanel(); p.allowedContentTypes = HUD.docTypes
        if p.runModal() == .OK, let url = p.url { openFile(url) }
    }
    func openFile(_ url: URL) {
        guard let pv = pv, pv.replay == nil else { return }
        if pv.window?.isDocumentEdited == true {
            let a = NSAlert(); a.messageText = "Discard unsaved changes?"; a.informativeText = "Opening \(url.lastPathComponent) replaces the current painting."
            a.addButton(withTitle: "Discard"); a.addButton(withTitle: "Cancel")
            if a.runModal() != .alertFirstButtonReturn { return }
        }
        do { try pv.eng.load(from: url); docURL = url; pv.window?.isDocumentEdited = false; pv.fitted = false } catch { NSAlert(error: error).runModal() }
    }
    @objc func export() {
        guard let pv = pv else { return }
        let p = NSSavePanel(); p.allowedContentTypes = [.png]; p.nameFieldStringValue = "Painting.png"
        if p.runModal() == .OK, let url = p.url, let img = pv.sc.export() { writePNG(img, url.path) }
    }
}

// ---------------------------------------------------------------- app
final class AppDelegate: NSObject, NSApplicationDelegate {
    var win: NSWindow!
    var pv: PaintView!
    var hud: HUD!
    func applicationDidFinishLaunching(_ n: Notification) {
        win = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1500, height: 950), styleMask: [.titled, .closable, .resizable, .miniaturizable], backing: .buffered, defer: false)
        win.title = "PaintIRL"
        win.center()
        pv = PaintView(frame: win.contentView!.bounds, device: GPU.shared.dev)
        pv.autoresizingMask = [.width, .height]
        pv.setup()
        win.contentView!.addSubview(pv)
        hud = HUD(pv)
        hud.view.translatesAutoresizingMaskIntoConstraints = false
        win.contentView!.addSubview(hud.view)
        NSLayoutConstraint.activate([hud.view.leadingAnchor.constraint(equalTo: win.contentView!.leadingAnchor, constant: 12),
                                     hud.view.topAnchor.constraint(equalTo: win.contentView!.topAnchor, constant: 12)])
        pv.onStatus = { [weak self] s in self?.hud.label.stringValue = s }
        pv.hud = hud
        pv.status()
        OSC.launch(pv, hud)
        win.makeKeyAndOrderFront(nil)
        win.makeFirstResponder(pv)
        let menu = NSMenu(), item = NSMenuItem(); menu.addItem(item)
        let sub = NSMenu(); sub.addItem(withTitle: "Quit PaintIRL", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"); item.submenu = sub
        let fileItem = NSMenuItem(), file = NSMenu(title: "File"); menu.addItem(fileItem); fileItem.submenu = file
        for (t, sel, k, mods) in [("Open…", #selector(HUD.open), "o", NSEvent.ModifierFlags.command), ("Save", #selector(HUD.save), "s", .command),
                                  ("Save As…", #selector(HUD.saveAs), "s", [.command, .shift]), ("Export PNG…", #selector(HUD.export), "e", .command)] {
            let it = NSMenuItem(title: t, action: sel, keyEquivalent: k); it.keyEquivalentModifierMask = mods; it.target = hud; file.addItem(it)
        }
        NSApp.mainMenu = menu
        hud.docURL = nil
        pv.eng.onChange = { [weak self] in self?.win.isDocumentEdited = true }
        NSApp.activate(ignoringOtherApps: true)
        if let u = launchURL { hud.openFile(u) }
        if ProcessInfo.processInfo.environment["PAINT_SELFTEST"] != nil { selfTest() }
    }
    // double-clicked files (may arrive before the window exists)
    var launchURL: URL?
    func application(_ app: NSApplication, open urls: [URL]) {
        guard let u = urls.first else { return }
        if let hud = hud { hud.openFile(u); win.makeKeyAndOrderFront(nil) } else { launchURL = u }
    }

    // drives the real input handlers with synthesized events, then saves a frame and quits
    func selfTest() {
        var steps: [() -> Void] = []
        let pv = self.pv!, sc = pv.sc!, e = pv.eng!
        func ev(_ type: NSEvent.EventType, _ w: V3) -> NSEvent {
            let p = pv.convert(sc.project(w, pv.bounds.size), to: nil)
            return NSEvent.mouseEvent(with: type, location: p, modifierFlags: [], timestamp: 0, windowNumber: win.windowNumber, context: nil, eventNumber: 0, clickCount: 1, pressure: 1)!
        }
        func drag(_ pts: [V3], _ n: Int) {
            steps.append { pv.mouseDown(with: ev(.leftMouseDown, pts[0])) }
            for k in 1...n {
                let f = Float(k) / Float(n) * Float(pts.count - 1), i = min(Int(f), pts.count - 2), t = f - Float(i)
                let w = pts[i] + (pts[i + 1] - pts[i]) * t
                steps.append { pv.mouseDragged(with: ev(.leftMouseDragged, w)) }
            }
            steps.append { pv.mouseUp(with: ev(.leftMouseUp, pts.last!)) }
            for _ in 0..<10 { steps.append {} }
        }
        let pal = e.palette.origin, pz = e.palette.z
        func P(_ x: Float, _ y: Float) -> V3 { V3(pal.x + x, pal.y + y, pz) }
        let tube = sc.tubeRest(7).columns.3
        steps.append { pv.mouseDown(with: ev(.leftMouseDown, V3(tube.x, tube.y - 40, TABLE_Z + 10))); pv.mouseUp(with: ev(.leftMouseUp, V3(tube.x, tube.y - 40, TABLE_Z + 10))) }
        drag([P(30, 120), P(52, 118)], 50)
        steps.append { pv.choose(.brush(4)) }
        drag([P(36, 112), P(40, 125), P(46, 112), P(50, 124)], 60)
        drag([V3(-80, 40, 0), V3(-20, 50, 0), V3(40, 30, 0)], 50)
        drag([V3(-60, -20, 0), V3(0, 10, 0), V3(50, -30, 0)], 50)
        let t2 = sc.tubeRest(0).columns.3
        steps.append { pv.mouseDown(with: ev(.leftMouseDown, V3(t2.x, t2.y - 40, TABLE_Z + 10))); pv.mouseUp(with: ev(.leftMouseUp, V3(t2.x, t2.y - 40, TABLE_Z + 10))) }
        drag([P(60, 160), P(90, 160)], 40)
        steps.append { pv.choose(.brush(6)) }
        drag([P(70, 150), P(75, 175), P(50, 170), P(45, 120), P(70, 115)], 80)
        let ks = sc.brushRestPos(6)
        steps.append { pv.mouseDown(with: ev(.leftMouseDown, V3(ks.x, ks.y - 40, TABLE_Z + 4))); pv.mouseUp(with: ev(.leftMouseUp, V3(ks.x, ks.y - 40, TABLE_Z + 4))) }
        steps.append { print("held after clicking the knife slot:", sc.held) }
        steps.append { pv.choose(.brush(9)) }
        drag([V3(-150, 100, 0), V3(-60, 110, 0), V3(-40, 60, 0)], 40)
        drag([P(38, 222), P(-212, 232)], 30)        // carry the palette by its thumb hole over the canvas
        steps.append { print("palette moved to", e.palette.origin, "z", e.palette.z, "last op", e.history.last.map { "\($0)" } ?? "") }
        steps.append { pv.choose(.brush(4)) }
        drag([V3(-150, 155, 0), V3(-60, 155, 0)], 30)      // centred 5 mm off the top edge: the near bristles still paint
        steps.append { print("edge stroke painted top-row tiles:", (0..<e.canvas.tilesX).filter { e.canvas.painted[(e.canvas.tilesY - 1) * e.canvas.tilesX + $0] }.count) }
        var n0 = 0
        steps.append { n0 = e.history.count; pv.startReplay() }
        steps.append { pv.replaySpeed(9) }
        for _ in 0..<400 { steps.append {} }
        steps.append { print("replay at full speed finished:", pv.replay == nil, "history", e.history.count, "of", n0) }
        steps.append { pv.startReplay() }
        for _ in 0..<5 { steps.append {} }
        steps.append { pv.stopReplay(); print("replay stopped early: history", e.history.count, "of", n0) }
        steps.append { pv.mouseMoved(with: ev(.mouseMoved, V3(60, -10, 0))) }
        for _ in 0..<20 { steps.append {} }
        steps.append { [unowned self] in
            let off = Offscreen(Int(pv.drawableSize.width), Int(pv.drawableSize.height))
            let cmd = GPU.shared.q.makeCommandBuffer()!
            e.shade(cmd); sc.encode(cmd, off.pass, aspect: pv.aspect)
            cmd.commit(); cmd.waitUntilCompleted()
            writePNG(off.image(), "tool/out/selftest.png")
            print("selftest done: history \(e.history.count) ops, tube left \(e.tubes[7].left), frame \(pv.frameMS) ms")
            NSApp.terminate(nil)
        }
        var i = 0
        Timer.scheduledTimer(withTimeInterval: 1.0 / 60, repeats: true) { t in
            if i < steps.count { steps[i](); i += 1 } else { t.invalidate() }
        }
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }
}

let args = CommandLine.arguments
if let i = args.firstIndex(of: "--test"), args.count > i + 2 {
    runTest(args[i + 1], args[i + 2])
    exit(0)
}
let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
