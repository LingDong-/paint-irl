# PaintIRL

A thick-paint (oil / acrylic / gouache) painting simulator for macOS. Swift + Metal, no dependencies.

    ./build.sh && open build/PaintIRL.app

**Desk**: canvas (40×30 cm by default; New canvas… offers 24×18 up to 80×60 cm, portrait, cotton / linen / gessoed panel), palette with thumb hole, a tray of 15 small pigment tubes + glazing medium, a rigger, rounds 0/2/6/12, flats 4/8/14, a palette knife and an HB pencil.

- Click a tube to pick it up, drag on the palette to squeeze. Esc puts it back. Tubes flatten as they're used but never run dry.
- Click a brush or the knife to pick it up (or keys 1–9 and 0, left to right). The knife scrapes palette space clean and mixes piles. Drag through paint on the palette to load it; paint on the canvas.
- The pencil sketches on the ground: a light touch catches only the tops of the weave. Paint covers it like a real underdrawing (thin glazes let it show).
- Drag the palette by its thumb hole to carry it next to the area you're working on; over the canvas it's held just above it. Moves are recorded, undoable and replayed.
- Each brush keeps its own paint load. C cleans the active brush.
- Flats turn to follow the stroke; `\` cycles following → edgewise (turned 90°, for thin lines) → fixed angle, and `[` `]` turn a fixed angle. Tablet pressure/tilt/rotation are used when present; with a mouse, `-` `=` set pressure.
- Scroll / pinch to zoom, two-finger scroll, right-drag or space-drag to pan. F fits the desk, G the canvas. H cycles: whole brush / no handle / hidden.
- D (or the sun button) previews drying as a heatmap; pick hours, Dry. Paint that dried stops mixing with new strokes. Layers keep their order as they dry; untick "dulling" to keep dried paint as glossy as wet.
- ⌘Z / ⇧⌘Z undo/redo everything (strokes, squeezes, drying, crop, wipes). ⌘S saves to the open `.pirl` file (`.paintirl` and older `.paint` files open too); saved files open by double-click in Finder (after the app has been launched once) and show a tiny preview of the painting as their icon (state + full operation history; only what changed is re-encoded), ⇧⌘S saves as, ⌘O opens (older files too). ⌘E exports PNG. Play replays the painting from its history: ←/→ change speed (up to full speed), Esc or Play again stops and brings back the finished painting.
- Brushes keep painting when their centre runs a little off the edge of the canvas or palette.

**Simulation**: each texel holds amounts of 16 paint channels in a wet top film, a set under-layer and a dried layer; colour is computed with 8-band Kubelka–Munk. Brushes are a few hundred bristle tufts with friction, clumping and splay; paint moves both ways between tufts and canvas (film splitting, pickup, mixing), and tufts are fed from per-zone reservoirs.

**OSC** (experimental, `osc.swift`): other programs can pick up tools and drive the pen (position, pressure, tilt, roll) over UDP on 127.0.0.1:9000, once switched on with the antenna button in the toolbar. Custom brushes (fingers, stamps, sponges, odd bristle layouts) are sent tuft by tuft. The command list is at the top of `osc.swift`; `tool/osc.py demo` / `brushes` paint small studies, and `PaintOSC/` is a Processing (oscP5) example.

Headless test renders: `build/PaintIRL.app/Contents/MacOS/PaintIRL --test {swatch|strokes|brush|paint|dry|dull|save|savesize|replay|v2|scene|perf} out.png` (outputs in `tool/out/`).
