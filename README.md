# PaintIRL

Very realistic oil painting app/simulator.

| ![](screenshots/001.png) | ![](screenshots/002.png) | 
|---|---|
| ![](screenshots/003.png) | ![](screenshots/004.png) | 
| ![](screenshots/005.png) | ![](screenshots/006.png) | 

Currently works on macOS (Metal) and web (webgpu), see respective folders fo implementation. ([Downloads](https://github.com/LingDong-/paint-irl/releases))

Caution: mostly vibe-coded, so don't look too hard inside. But seems to work well. Here're some paintings I'm making with it:

The [peppers standard test image](https://sipi.usc.edu/database/database.php?volume=misc&image=13):

![](screenshots/010.gif)

![](screenshots/011.png)

Now with OSC support! You can control the app from another software, much like how you send commands to a plotter, via OSC (Open Sound Control). See [mac/osc.swift](mac/osc.swift) comments.

![](screenshots/101.png)