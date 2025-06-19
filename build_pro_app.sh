#!/bin/bash

# Build AI Terminal Pro as a proper macOS app

APP_NAME="AI Terminal Pro"
APP_DIR="$APP_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

# Clean previous build
rm -rf "$APP_DIR"

# Create app structure
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Compile Swift code
echo "Compiling AI Terminal Pro..."
swiftc -o "$MACOS_DIR/$APP_NAME" AITerminalPro.swift \
    -framework SwiftUI \
    -framework AppKit \
    -target arm64-apple-macos12.0 \
    -parse-as-library

if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

# Create Info.plist
cat > "$CONTENTS_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>AI Terminal Pro</string>
    <key>CFBundleDisplayName</key>
    <string>AI Terminal Pro</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleExecutable</key>
    <string>AI Terminal Pro</string>
    <key>CFBundleIdentifier</key>
    <string>com.axisthorn.terminal</string>
    <key>CFBundleVersion</key>
    <string>2.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>2.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.developer-tools</string>
    <key>NSSupportsAutomaticTermination</key>
    <false/>
    <key>NSSupportsSuddenTermination</key>
    <false/>
</dict>
</plist>
EOF

# Copy icon if it exists
if [ -f "AppIcon.icns" ]; then
    cp "AppIcon.icns" "$RESOURCES_DIR/"
fi

# Make executable
chmod +x "$MACOS_DIR/$APP_NAME"

echo "AI Terminal Pro.app created successfully!"
echo "You can now double-click the app to run it."