# PaintIRL (WebGPU)

Browser port of the thick-paint simulator in `../mac/` (Swift + Metal). Plain JS modules + WGSL, no dependencies. Chrome / Edge.

    node tool/serve.js        # then open http://localhost:8123

(Any static server works; ES modules and `fetch` need http, not `file://`.)

Same desk, tools, keys and simulation as the Mac version (see `../mac/README.md`): tubes, palette with thumb hole, rounds / flats / rigger / knife / pencil, Kubelka–Munk mixing, bristle physics, drying with heatmap preview, undo/redo of everything, crop, replay.

Input: pen tablets through Pointer Events (pressure, tilt, twist, coalesced full-rate samples); with a mouse `-` `=` set pressure. Trackpad scroll pans, pinch or mouse wheel zooms, right-drag or space-drag pans.

Files: ⌘S / ⇧⌘S save `.pirl` version 3, the same format as the Mac app (painted 128×128 tiles with its plane/predictor codec and raw deflate, brush loads, full history), bit-exact both ways; ⌘O opens; ⌘E exports PNG. Older LZFSE `.pirl` files (v1–2) need re-saving in the Mac app first. Tile encoding/decoding runs in a few workers (warmed up in the background while you're not painting) and encoded tiles are cached until painted over, so saving again only re-encodes what changed.

## How it differs from the Metal version

- Paint state (16 pigment channels × wet/set layers, aux, dried reflectance, height) lives in a pool of 64×64-texel tiles in a storage buffer, allocated only where something is painted, found through a page table. Ground, display caches (colour as sqrt-encoded rgba8) and the drying preview are dense textures.
- Brush reservoir deltas are reduced with subgroups when available, plain atomics otherwise.
- Per-substep bristle records and kernel parameters go into one arena buffer, uploaded once per frame, bound with dynamic offsets.
- Memory: about 600 MB at start (40×30 cm canvas + palette + shadow maps), plus ~0.4 MB per painted 6.4 mm tile. Canvas sizes go up to 50×40 cm.

## Files

- `shaders.wgsl`: all kernels and render shaders, in sections (`//// name deps…`) compiled per pipeline
- `engine.js`: GPU setup, pigments, surfaces, bristle brushes, ops, undo, save format
- `scene.js`: procedural meshes, labels, camera, picking, rendering
- `app.js`: input, HUD, frame loop, replay, files, scripted tests
- `tool/serve.js`: dev server + headless test runner: `node tool/serve.js run "test=strokes"` (also `scene`, `dry`, `paint`, `brush`, `perf`, `selftest`, and `pirl`: load `pirlv3test.pirl`, re-save, compare tile by tile; `&file=x.pirl` for another file); renders land in `tool/out/`. Runs use port 8124 so they don't clash with a running dev server.
