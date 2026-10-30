#!/bin/sh
# Builds PaintIRL.app (arm64) with swiftc + the Metal compiler. No dependencies.
set -e
cd "$(dirname "$0")"
APP=build/PaintIRL.app
mkdir -p $APP/Contents/MacOS $APP/Contents/Resources
xcrun -sdk macosx metal -O3 -c shaders.metal -o build/shaders.air
xcrun -sdk macosx metallib build/shaders.air -o $APP/Contents/Resources/shaders.metallib
swiftc -O -target arm64-apple-macos14 -o $APP/Contents/MacOS/PaintIRL engine.swift scene.swift main.swift osc.swift \
  -framework Metal -framework MetalKit -framework AppKit
cat > $APP/Contents/Info.plist <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>PaintIRL</string>
<key>CFBundleIdentifier</key><string>local.paintirl</string>
<key>CFBundleName</key><string>PaintIRL</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>NSHighResolutionCapable</key><true/>
<key>LSMinimumSystemVersion</key><string>14.0</string>
<key>CFBundleDocumentTypes</key><array><dict>
  <key>CFBundleTypeName</key><string>PaintIRL Painting</string><key>CFBundleTypeRole</key><string>Editor</string>
  <key>LSHandlerRank</key><string>Owner</string><key>LSItemContentTypes</key><array><string>local.paintirl.painting</string></array>
</dict></array>
<key>UTExportedTypeDeclarations</key><array><dict>
  <key>UTTypeIdentifier</key><string>local.paintirl.painting</string><key>UTTypeDescription</key><string>PaintIRL Painting</string>
  <key>UTTypeConformsTo</key><array><string>public.data</string></array>
  <key>UTTypeTagSpecification</key><dict><key>public.filename-extension</key><array><string>pirl</string><string>paintirl</string><string>paint</string></array></dict>
</dict></array>
</dict></plist>
PL
echo built $APP
