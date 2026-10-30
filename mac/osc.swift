// OSC remote control: drive the pen, brushes and tubes from another program over UDP.
//
// Off until switched on with the antenna button in the HUD (remembered between launches); then listens on 127.0.0.1:9000.
// PAINT_OSC=9001 picks another port (and switches it on at launch), PAINT_OSC=0.0.0.0:9000 accepts other machines,
// PAINT_OSC=off keeps it off. Ints, floats and doubles are interchangeable; bundles are unpacked in order (timetags ignored).
//
// Commands run in order on the app's own clock, like g-code: a burst of messages becomes an evenly timed stroke.
// Coordinates are mm from the top-left corner of the current frame, x right, y down.
//
//   /brush n|name          pick up a brush: 1–10 in tray order (the 1–9,0 keys) or a name ("flat 8", "knife", "pencil")
//   /tube n|name           pick up a tube: 1–16 in tray order or part of its name ("ultramarine", "medium")
//   /drop                  put the held tool back
//   /down [p]              touch down (begins a stroke or a squeeze), optionally setting pressure
//   /up                    lift off (ends the stroke; it is recorded, undoable and replayed like any other)
//   /move x y [p [tx ty [roll]]]     move to x y, pressure/tilt/roll interpolated along the way
//   /moveby dx dy [p [tx ty [roll]]] the same, relative (the extra values are still absolute)
//   /pressure p            0..1
//   /tilt tx ty            direction the handle leans, as a vector in the frame; length = sin(angle from vertical), up to 0.85
//   /roll deg|follow|edge  flats & knife: fixed angle of the broad edge (deg, from +x toward +y), or turn with the stroke / edgewise
//   /feed down [up]        speed in mm/s while touching / hovering (default 100, 400); 0 = jump at once (stream your own timing)
//   /wait s                dwell (the brush keeps working while held down)
//   /frame canvas|palette  coordinate frame (default canvas); the palette frame follows the palette when it is moved
//   /palette x y           move the palette: its top-left corner, in canvas mm
//   /clean  /scrape  /wipe  /undo  /redo  /dry hours [palette 0|1]
//   /export path.png  /save path.pirl  /open path.pirl     absolute paths (~ ok); /open asks first if there are unsaved changes
//   /sync [args]           replies /sync args to the sender once everything before it has run (flow control)
//   /tufts x y len r …     add tufts for the next brush (any number of messages): position (mm around the pen point, y down),
//                          length (> 0: bristle length, or relief height for a pad; the longest touch first), radius
//   /newbrush name [stiffness]  make a brush from those tufts (then /brush name). Stiffness 0 bends and drags like the built-in
//                          brushes' hair; 1 (default) is a rigid pad (finger, stamp, sponge…): tufts stay straight, it's held
//                          upright whatever the tilt, and full pressure pushes it 0.6 mm past its relief (at least 2 mm), where
//                          soft bristles go 62% of their length. Turns with the stroke unless /roll sets an angle.
//      Up to 1500 tufts, 58 mm across. No 3D model: it shows as short tufts while held. Saved with the painting.
//   /probe x y w h         debug: replies /probe volume max load: paint in that canvas rectangle (unit·mm², units = 0.01 mm), the
//                          highest point (with the weave), and paint held by the brush (tool/blob.py uses it)
// Errors are printed and replied to the sender as /error message.
// Immediate (not queued):
//   /stop                  drop the queue and lift
//   /info                  replies /info/canvas w h, /info/palette x y w h, /info/pen x y p down queued, /info/brushes …, /info/tubes …
import AppKit
import Metal
import simd

var osc: OSC?

final class OSC {
    enum Arg { case n(Float), s(String) }
    struct Msg { var addr: String; var args: [Arg]; var from = sockaddr_in() }
    struct Pose {
        var pos: V2, p: Float, tilt: V2, roll: Float
        func mix(_ b: Pose, _ f: Float) -> Pose { Pose(pos: pos + (b.pos - pos) * f, p: p + (b.p - p) * f, tilt: tilt + (b.tilt - tilt) * f, roll: roll + (b.roll - roll) * f) }
    }
    unowned let pv: PaintView
    var eng: Engine { pv.eng }
    let fd: Int32
    var src: DispatchSourceRead?
    var queue: [Msg] = []
    var pose: Pose
    var move: (a: Pose, b: Pose, t0: Float, dur: Float)?
    var clock: Float = 0, lastT: Float = 0
    var down = false
    var rollMode = 0        // 0 follow the stroke, 1 edgewise, 2 fixed
    var frame = 0           // 0 canvas, 1 palette
    var feed: Float = 100, travel: Float = 400
    var tufts: [Float] = []         // x, y, length, radius for the next /newbrush

    static let env = ProcessInfo.processInfo.environment["PAINT_OSC"]
    static var address: (host: String, port: UInt16) {
        let parts = (env ?? "").split(separator: ":")
        return (parts.count > 1 ? String(parts[0]) : "127.0.0.1", UInt16(parts.last ?? "") ?? 9000)
    }
    // at launch: on if it was on last time (or PAINT_OSC asks for it)
    static func launch(_ pv: PaintView, _ hud: HUD) {
        if env != "off" && (env != nil || UserDefaults.standard.bool(forKey: "osc")) { osc = OSC(pv, host: address.host, port: address.port) }
        func find(_ v: NSView) -> NSButton? {
            if let b = v as? NSButton, b.action == #selector(HUD.toggleOSC(_:)) { return b }
            return v.subviews.lazy.compactMap(find).first
        }
        if let b = find(hud.view) { show(b) }
    }
    static func show(_ b: NSButton) {
        b.contentTintColor = osc == nil ? .labelColor : .controlAccentColor
        b.toolTip = osc == nil ? "OSC remote control: off" : "OSC remote control: listening on \(address.host):\(address.port)"
    }
    func stop() {
        if down { pv.pendingActions.append { [eng] c in eng.end(c) } }     // finish a stroke in progress
        src?.cancel()
        print("osc: stopped")
    }

    init?(_ pv: PaintView, host: String, port: UInt16) {
        self.pv = pv
        let c = pv.eng.canvas
        pose = Pose(pos: c.origin + V2(Float(c.w), Float(c.h)) / PPM / 2, p: 0.5, tilt: pv.tilt, roll: pv.roll)
        fd = socket(AF_INET, SOCK_DGRAM, 0)
        var one: Int32 = 1, buf: Int32 = 4 << 20       // a big receive buffer so long bursts of moves aren't dropped
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, 4)
        setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, 4)
        var a = sockaddr_in()
        a.sin_len = UInt8(MemoryLayout<sockaddr_in>.size); a.sin_family = sa_family_t(AF_INET)
        a.sin_port = port.bigEndian; a.sin_addr.s_addr = inet_addr(host)
        let ok = withUnsafePointer(to: &a) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }
        guard fd >= 0, ok == 0 else { print("osc: can't listen on \(host):\(port)"); close(fd); return nil }
        _ = fcntl(fd, F_SETFL, O_NONBLOCK)
        let s = DispatchSource.makeReadSource(fileDescriptor: fd, queue: .main)
        s.setEventHandler { [unowned self] in receive() }
        s.setCancelHandler { [fd] in close(fd) }
        s.resume(); src = s
        print("osc: listening on \(host):\(port)")
    }

    // ------------------------------------------------ network
    func receive() {
        var b = [UInt8](repeating: 0, count: 65536)
        while true {
            var from = sockaddr_in(), len = socklen_t(MemoryLayout<sockaddr_in>.size)
            let n = withUnsafeMutablePointer(to: &from) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { recvfrom(fd, &b, b.count, 0, $0, &len) } }
            if n <= 0 { return }
            var msgs: [Msg] = []
            OSC.parse(b[0..<n], &msgs)
            for var m in msgs {
                m.from = from
                switch m.addr {
                case "/stop": queue = [Msg(addr: "/up", args: [])]; move = nil
                case "/info": info(m.from)
                default: queue.append(m)
                }
            }
        }
    }
    static func parse(_ b: ArraySlice<UInt8>, _ out: inout [Msg]) {
        var o = b.startIndex
        func u32() -> UInt32? {
            guard o + 4 <= b.endIndex else { return nil }
            defer { o += 4 }
            return b[o..<(o + 4)].reduce(0) { $0 << 8 | UInt32($1) }
        }
        func u64() -> UInt64? { guard let hi = u32(), let lo = u32() else { return nil }; return UInt64(hi) << 32 | UInt64(lo) }
        func str() -> String? {
            guard let e = b[o...].firstIndex(of: 0) else { return nil }
            defer { o = b.startIndex + ((e - b.startIndex + 4) & ~3) }
            return String(decoding: b[o..<e], as: UTF8.self)
        }
        if b.starts(with: Array("#bundle".utf8) + [0]) {
            o += 16
            while let n = u32().map(Int.init), n > 0, o + n <= b.endIndex { parse(b[o..<(o + n)], &out); o += n }
            return
        }
        guard let addr = str(), addr.hasPrefix("/") else { return }
        var args: [Arg] = []
        for t in (o < b.endIndex ? str() ?? "," : ",").dropFirst() {
            switch t {
            case "i": guard let v = u32() else { break }; args.append(.n(Float(Int32(bitPattern: v))))
            case "f": guard let v = u32() else { break }; args.append(.n(Float(bitPattern: v)))
            case "d": guard let v = u64() else { break }; args.append(.n(Float(Double(bitPattern: v))))
            case "h": guard let v = u64() else { break }; args.append(.n(Float(Int64(bitPattern: v))))
            case "s", "S": guard let v = str() else { break }; args.append(.s(v))
            case "T": args.append(.n(1))
            case "F": args.append(.n(0))
            case "N", "I": break
            case "c", "r", "m": _ = u32()
            case "t": _ = u64()
            case "b": guard let n = u32() else { break }; o += (Int(n) + 3) & ~3
            default: break
            }
        }
        out.append(Msg(addr: addr, args: args))
    }
    func send(_ addr: String, _ args: [Arg], to: sockaddr_in) {
        var d = [UInt8]()
        func pad(_ s: String) { d += Array(s.utf8) + [0]; while d.count % 4 != 0 { d.append(0) } }
        pad(addr)
        pad("," + args.map { if case .s = $0 { return "s" } else { return "f" } }.joined())
        for a in args {
            switch a {
            case .n(let v): let u = v.bitPattern; d += [UInt8(u >> 24), UInt8(u >> 16 & 255), UInt8(u >> 8 & 255), UInt8(u & 255)]
            case .s(let s): pad(s)
            }
        }
        var to = to
        _ = withUnsafePointer(to: &to) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { sendto(fd, d, d.count, 0, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }
    }
    func info(_ to: sockaddr_in) {
        let c = eng.canvas, p = eng.palette, tl = fromWorld(p.origin + V2(0, Float(p.h) / PPM), 0), pen = fromWorld(pose.pos, frame)
        send("/info/canvas", [.n(Float(c.w) / PPM), .n(Float(c.h) / PPM)], to: to)
        send("/info/palette", [.n(tl.x), .n(tl.y), .n(Float(p.w) / PPM), .n(Float(p.h) / PPM)], to: to)
        send("/info/pen", [.n(pen.x), .n(pen.y), .n(pose.p), .n(down ? 1 : 0), .n(Float(queue.count))], to: to)
        send("/info/brushes", brushOrder.map { .s(eng.brushes[$0].name) }, to: to)
        send("/info/tubes", TUBE_ORDER.filter { $0 < eng.tubes.count }.map { .s(eng.tubes[$0].name) }, to: to)
    }

    // ------------------------------------------------ frames: mm from the top-left corner, y down
    func surf(_ f: Int) -> Surface { f == 1 ? eng.palette : eng.canvas }
    func toWorld(_ v: V2, _ f: Int) -> V2 { let s = surf(f); return V2(s.origin.x + v.x, s.origin.y + Float(s.h) / PPM - v.y) }
    func fromWorld(_ w: V2, _ f: Int) -> V2 { let s = surf(f); return V2(w.x - s.origin.x, s.origin.y + Float(s.h) / PPM - w.y) }
    func flip(_ v: V2) -> V2 { V2(v.x, -v.y) }
    // which surface a point is on, as PaintView.locate does for the mouse (seen from straight above)
    func surface(_ w: V2) -> (Int8, Float) {
        let p = eng.palette, c = eng.canvas, m = pv.reach
        if p.z > c.z && p.contains(w) { return (1, p.z) }
        if c.contains(w) { return (0, c.z) }
        if p.contains(w) { return (1, p.z) }
        if p.z > c.z && p.near(w, m) { return (1, p.z) }
        if c.near(w, m) { return (0, c.z) }
        if p.near(w, m) { return (1, p.z) }
        return (-1, TABLE_Z)
    }

    // ------------------------------------------------ per frame (called from PaintView.draw)
    func step(_ cmd: MTLCommandBuffer) {
        if pv.replay != nil || pv.mouseDown { return }
        let now = pv.now
        var gpu = false         // something was encoded this frame (export/save wait a frame so they include it)
        while true {
            if let m = move {
                if clock >= now || eng.backlog > 30 { break }
                clock = min(clock + 1 / 120, m.t0 + m.dur)
                let end = clock >= m.t0 + m.dur
                apply(m.a.mix(m.b, end ? 1 : (clock - m.t0) / m.dur), clock)
                if end { move = nil }
                continue
            }
            guard let c = queue.first, run(c, cmd, &gpu) else { break }
            queue.removeFirst()
        }
        if move == nil && queue.isEmpty {
            clock = max(clock, now)
            if down && now - lastT > 0.04 { apply(pose, now) }       // held still: keep the brush working
        }
    }
    func apply(_ q: Pose, _ t: Float) {
        var q = q
        if rollMode < 2 {
            q.roll = pose.roll
            let d = q.pos - pose.pos
            if case .brush(let i) = pv.sc.held, !eng.brushes[i].round, simd_length(d) > 0.05 { q.roll = Engine.followRoll(pose.roll, dir: d, edge: rollMode == 1) }
        }
        pose = q
        let (s, z) = surface(q.pos)
        pv.sc.cursor = V3(q.pos.x, q.pos.y, z); pv.tilt = q.tilt; pv.roll = q.roll
        if down { eng.add(Sample(x: q.pos.x, y: q.pos.y, p: q.p, tx: q.tilt.x, ty: q.tilt.y, roll: q.roll, t: t, s: s)); lastT = t }
    }
    // ends the stroke once the engine has caught up; false means try again next frame
    func lift(_ cmd: MTLCommandBuffer, _ gpu: inout Bool) -> Bool {
        if !down { return true }
        if eng.backlog > 0 { return false }
        eng.end(cmd); down = false; gpu = true
        return true
    }
    var brushOrder: [Int] { BRUSH_ORDER + Array(eng.builtinBrushes..<eng.brushes.count) }      // defined brushes are 11, 12, …
    func pick(_ a: [Arg], _ names: [String], _ order: [Int]) -> Int? {
        switch a.first {
        case .n(let v)?: let k = Int(v) - 1; return order.indices.contains(k) ? order[k] : nil
        case .s(let s)?: let s = s.lowercased(); return names.lastIndex { $0.lowercased() == s } ?? names.firstIndex { $0.lowercased().contains(s) }
        case nil: return nil
        }
    }

    // runs one command; false means it has to wait (for the brush to finish, or for the next frame)
    func run(_ m: Msg, _ cmd: MTLCommandBuffer, _ gpu: inout Bool) -> Bool {
        let n = m.args.compactMap { if case .n(let v) = $0 { return v } else { return nil } }
        let s = m.args.compactMap { if case .s(let v) = $0 { return v.lowercased() } else { return nil } }.first
        switch m.addr {
        case "/move", "/moveby":
            guard n.count >= 2 else { break }
            var b = pose
            b.pos = m.addr == "/move" ? toWorld(V2(n[0], n[1]), frame) : pose.pos + flip(V2(n[0], n[1]))
            if n.count >= 3 { b.p = min(max(n[2], 0), 1) }
            if n.count >= 5 { b.tilt = flip(V2(n[3], n[4])) }
            if n.count >= 6 { b.roll = -n[5] * .pi / 180; rollMode = 2 }
            let v = down ? feed : travel
            move = (pose, b, clock, v > 0 ? simd_distance(pose.pos, b.pos) / v : 0)
        case "/wait": move = (pose, pose, clock, max(n.first ?? 0, 0))
        case "/pressure": if let p = n.first { pose.p = min(max(p, 0), 1) }
        case "/tilt": if n.count >= 2 { pose.tilt = flip(V2(n[0], n[1])) }
        case "/roll":
            if let r = n.first { pose.roll = -r * .pi / 180; rollMode = 2 } else if s == "follow" { rollMode = 0 } else if s == "edge" { rollMode = 1 }
        case "/feed": if let f = n.first { feed = f; if n.count > 1 { travel = n[1] } }
        case "/frame": frame = s == "palette" || n.first == 1 ? 1 : 0
        case "/down":
            if let p = n.first { pose.p = min(max(p, 0), 1) }
            if down { break }
            switch pv.sc.held {
            case .brush(let i): eng.begin(.stroke(brush: i, samples: []))
            case .tube(let i): eng.begin(.squeeze(tube: i, samples: []))
            case .none: return true
            }
            down = true; apply(pose, clock)
        case "/up": return lift(cmd, &gpu)
        case "/brush", "/tube", "/drop":
            if !lift(cmd, &gpu) { return false }
            if m.addr == "/drop" { pv.choose(.none) }
            else if m.addr == "/brush", let i = pick(m.args, eng.brushes.map { $0.name }, brushOrder) {
                pv.choose(.brush(i))
            }
            else if m.addr == "/tube", let i = pick(m.args, eng.tubes.map { $0.name + " " + PIGMENTS[$0.pigment].code }, TUBE_ORDER) { pv.choose(.tube(i)) }
            else { fail(m, "no such tool") }
        case "/clean", "/scrape", "/wipe", "/undo", "/redo", "/dry", "/palette":
            if !lift(cmd, &gpu) { return false }
            switch m.addr {
            case "/clean": eng.cleanBrush(cmd, eng.active)
            case "/scrape": eng.clearSurface(cmd, palette: true)
            case "/wipe": eng.clearSurface(cmd, palette: false)
            case "/undo": eng.undo(cmd)
            case "/redo": eng.redo(cmd)
            case "/dry": eng.dry(cmd, hours: n.first ?? 24, palette: n.count > 1 && n[1] != 0)
            default:
                guard n.count >= 2 else { break }
                eng.movePalette(toWorld(V2(n[0], n[1]), 0) - V2(0, Float(eng.palette.h) / PPM))
            }
            gpu = true
        case "/export", "/save", "/open":
            if !lift(cmd, &gpu) || gpu { return false }
            guard case .s(let p)? = m.args.first else { break }
            let path = (p as NSString).expandingTildeInPath
            switch m.addr {
            case "/export": if let img = pv.sc.export() { writePNG(img, path) }
            case "/save": pv.hud?.write(URL(fileURLWithPath: path))
            default: pv.hud?.openFile(URL(fileURLWithPath: path))
            }
        case "/probe":      // debug: x y w h (mm, canvas) -> /probe paint volume (unit·mm²), max height (units), paint held by the brush
            if !lift(cmd, &gpu) || gpu { return false }
            guard n.count >= 4 else { break }
            let c = eng.canvas, x0 = max(Int(n[0] * PPM), 0), y0 = max(c.h - Int((n[1] + n[3]) * PPM), 0)     // texels run y up
            let w = min(Int(n[2] * PPM), c.w - x0), h = min(Int(n[3] * PPM), c.h - y0)
            let hs = eng.readHeights(c, x0, y0, w, h), b = eng.brushes[eng.active]
            let l = eng.brushLoad(b)
            send("/probe", [.n(hs.reduce(0, +) / (PPM * PPM)), .n(hs.max() ?? 0), .n((l.tufts + l.belly) / (PPM * PPM))], to: m.from)
        case "/sync": send("/sync", m.args, to: m.from)
        case "/tufts": tufts += n
        case "/newbrush": newBrush(m)
        default: fail(m, "unknown command")
        }
        return true
    }
    func fail(_ m: Msg, _ why: String) {
        print("osc:", m.addr, why)
        send("/error", [.s(m.addr + ": " + why)], to: m.from)
    }

    // ------------------------------------------------ brushes defined from outside, tuft by tuft
    func newBrush(_ m: Msg) {
        defer { tufts = [] }
        guard case .s(let name)? = m.args.first else { return fail(m, "needs a name") }
        let t = stride(from: 0, to: tufts.count - 3, by: 4).map { V4(tufts[$0], tufts[$0 + 1], tufts[$0 + 2], tufts[$0 + 3]) }
        guard !t.isEmpty, t.count <= 1500 else { return fail(m, "needs 1 to 1500 tufts (sent with /tufts), not \(t.count)") }
        let lo = t.reduce(V2(repeating: .infinity)) { simd_min($0, V2($1.x, $1.y) - $1.w) }, hi = t.reduce(-lo) { simd_max($0, V2($1.x, $1.y) + $1.w) }
        guard hi.x - lo.x <= 58, hi.y - lo.y <= 58 else { return fail(m, "brushes can be at most 58 mm across") }
        guard t.allSatisfy({ $0.w > 0 && $0.z > 0 }) else { return fail(m, "length and radius must be > 0") }
        let stiff: Float = m.args.dropFirst().compactMap { (a: Arg) -> Float? in if case .n(let v) = a { return min(max(v, 0), 1) } else { return nil } }.first ?? 1
        let def = BrushDef(name: name, tufts: t.flatMap { [$0.x, $0.y, $0.z, $0.w] }, stiff: stiff)
        let i = eng.addBrush(def)
        print("osc: brush \(brushOrder.firstIndex(of: i).map { $0 + 1 } ?? i) \"\(name)\": \(t.count) tufts")
    }
}

// the antenna button in the HUD
extension HUD {
    @objc func toggleOSC(_ b: NSButton) {
        guard let pv = pv else { return }
        if let o = osc { o.stop(); osc = nil } else {
            let a = OSC.address
            osc = OSC(pv, host: a.host, port: a.port)
            if osc == nil {
                let al = NSAlert(); al.messageText = "Couldn't listen for OSC on \(a.host):\(a.port)"
                al.informativeText = "Another program may be using that port. PAINT_OSC=<port> picks another one."; al.runModal()
            }
        }
        UserDefaults.standard.set(osc != nil, forKey: "osc")
        OSC.show(b)
    }
}
