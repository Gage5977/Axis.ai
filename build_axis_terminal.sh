#!/bin/bash

echo "Building AXIS Terminal..."

APP_NAME="AXIS Terminal"
APP_DIR="$APP_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

# Clean previous build
rm -rf "$APP_DIR"

# Create app structure
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Compile Swift code
echo "Compiling Swift code..."
swiftc -o "$MACOS_DIR/$APP_NAME" AxisTerminal.swift \
    -framework SwiftUI \
    -framework Combine \
    -target arm64-apple-macos12.0 \
    -parse-as-library \
    -O

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed"
    exit 1
fi

# Create Info.plist
cat > "$CONTENTS_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>AXIS Terminal</string>
    <key>CFBundleDisplayName</key>
    <string>AXIS Terminal</string>
    <key>CFBundleExecutable</key>
    <string>AXIS Terminal</string>
    <key>CFBundleIdentifier</key>
    <string>com.axisthorn.terminal</string>
    <key>CFBundleVersion</key>
    <string>3.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>3.0.0</string>
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

# Copy icon if exists
if [ -f "AxisTerminal.icns" ]; then
    cp "AxisTerminal.icns" "$RESOURCES_DIR/AppIcon.icns"
elif [ -f "AppIcon.icns" ]; then
    cp "AppIcon.icns" "$RESOURCES_DIR/"
fi

# Make executable
chmod +x "$MACOS_DIR/$APP_NAME"

echo "✅ AXIS Terminal.app created successfully!"
echo ""
echo "The app will:"
echo "• Auto-detect your local AI services on ports 3000, 5001, 8000, 8080"
echo "• Connect to whatever service is available"
echo "• Work with any JSON API format"
echo ""
echo "You can now double-click 'AXIS Terminal.app' to run it."