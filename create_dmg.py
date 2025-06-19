#!/usr/bin/env python3
"""
Create a macOS .dmg installer for AI Terminal
"""

import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path

def create_app_bundle():
    """Create the .app bundle"""
    print("🔨 Creating AI Terminal.app...")
    
    app_name = "AI Terminal"
    bundle_name = f"{app_name}.app"
    base_dir = Path(__file__).parent.absolute()
    
    # Create app structure
    app_path = base_dir / bundle_name
    contents = app_path / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    
    # Clean existing
    if app_path.exists():
        shutil.rmtree(app_path)
    
    # Create directories
    for dir in [contents, macos, resources]:
        dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all necessary files to Resources
    print("📁 Copying application files...")
    
    # Core files
    files_to_copy = [
        "clean_terminal.py",
        "terminal_web.html",
        "local_ai_server.py",
        "start_terminal.sh",
        "setup_ai_system.sh",
        "run_ai.py"
    ]
    
    for file in files_to_copy:
        src = base_dir / file
        if src.exists():
            shutil.copy2(src, resources / file)
    
    # Copy directories
    dirs_to_copy = [
        "capabilities",
        "scripts",
        "tools",
        "shared",
        "mistral",
        "web-interfaces"
    ]
    
    for dir_name in dirs_to_copy:
        src_dir = base_dir / dir_name
        if src_dir.exists():
            shutil.copytree(src_dir, resources / dir_name, dirs_exist_ok=True)
    
    # Create Info.plist
    info_plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>AI Terminal</string>
    <key>CFBundleDisplayName</key>
    <string>AI Terminal</string>
    <key>CFBundleIdentifier</key>
    <string>com.axisthorn.aiterminal</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleExecutable</key>
    <string>AI Terminal</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.developer-tools</string>
    <key>NSAppleScriptEnabled</key>
    <true/>
</dict>
</plist>"""
    
    with open(contents / "Info.plist", "w") as f:
        f.write(info_plist)
    
    # Create launcher script
    launcher_script = """#!/bin/bash
# AI Terminal Launcher

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESOURCES="$( cd "$DIR/../Resources" && pwd )"

# Check if Terminal is running
if ! pgrep -x "Terminal" > /dev/null; then
    open -a Terminal
    sleep 1
fi

# Create a temporary script to run
TEMP_SCRIPT="/tmp/ai_terminal_launcher_$$.sh"
cat > "$TEMP_SCRIPT" << 'EOF'
#!/bin/bash
clear
echo "🚀 AI Terminal Starting..."
echo ""

# Change to resources directory
cd "$RESOURCES"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Please install from https://ollama.ai"
    echo ""
    read -p "Press Enter to continue anyway..."
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found."
    echo "Please install Python 3 from https://python.org"
    exit 1
fi

# Install Flask if needed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Installing required packages..."
    pip3 install flask flask-cors
fi

# Show options
echo "AI Terminal - Choose Interface:"
echo ""
echo "1) Terminal UI (Recommended)"
echo "2) Web Terminal"
echo "3) Advanced Terminal"
echo "4) Exit"
echo ""
read -p "Choose (1-4): " choice

case $choice in
    1)
        python3 clean_terminal.py
        ;;
    2)
        echo "Starting web terminal..."
        python3 local_ai_server.py &
        SERVER_PID=$!
        sleep 2
        open terminal_web.html
        echo ""
        echo "Web terminal opened in browser."
        echo "Press Ctrl+C to stop server..."
        wait $SERVER_PID
        ;;
    3)
        python3 advanced_terminal_ui.py
        ;;
    4)
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
EOF

chmod +x "$TEMP_SCRIPT"

# Tell Terminal to run our script
osascript -e "tell application \\"Terminal\\"
    activate
    do script \\"/bin/bash '$TEMP_SCRIPT'; rm -f '$TEMP_SCRIPT'; exit\\"
end tell"
"""
    
    launcher_path = macos / "AI Terminal"
    with open(launcher_path, "w") as f:
        f.write(launcher_script)
    
    os.chmod(launcher_path, 0o755)
    
    # Create simple icon
    create_icon(resources / "AppIcon.icns")
    
    return app_path

def create_icon(icon_path):
    """Create app icon"""
    print("🎨 Creating app icon...")
    
    # Create a simple iconset
    temp_dir = tempfile.mkdtemp()
    iconset = Path(temp_dir) / "AppIcon.iconset"
    iconset.mkdir()
    
    # Create PNG files using ImageMagick or sips
    sizes = [16, 32, 64, 128, 256, 512]
    
    for size in sizes:
        # Create simple colored square with text
        png_path = iconset / f"icon_{size}x{size}.png"
        
        # Try using sips to create a simple icon
        create_cmd = f"""
convert -size {size}x{size} xc:black \
    -fill '#00ff00' -draw "rectangle 2,2 {size-2},{size-2}" \
    -fill black -pointsize {size//4} -gravity center \
    -annotate +0+0 'AI' "{png_path}" 2>/dev/null || \
sips -s format png -z {size} {size} /System/Library/CoreServices/CoreTypes.bundle/Contents/Resources/GenericApplicationIcon.icns \
    --out "{png_path}" 2>/dev/null || \
touch "{png_path}"
"""
        os.system(create_cmd)
        
        # Also create @2x versions
        if size <= 256:
            os.system(f"cp '{png_path}' '{iconset}/icon_{size}x{size}@2x.png' 2>/dev/null || true")
    
    # Convert to icns
    os.system(f"iconutil -c icns '{iconset}' -o '{icon_path}' 2>/dev/null || touch '{icon_path}'")
    
    # Cleanup
    shutil.rmtree(temp_dir)

def create_dmg(app_path):
    """Create DMG installer"""
    print("💿 Creating DMG installer...")
    
    base_dir = Path(__file__).parent.absolute()
    dmg_name = "AI-Terminal-1.0.0.dmg"
    
    # Create temporary directory for DMG contents
    temp_dir = tempfile.mkdtemp()
    dmg_dir = Path(temp_dir) / "dmg"
    dmg_dir.mkdir()
    
    # Copy app to DMG directory
    shutil.copytree(app_path, dmg_dir / app_path.name)
    
    # Create Applications symlink
    os.symlink("/Applications", dmg_dir / "Applications")
    
    # Create README
    readme = """AI Terminal
===========

Installation:
1. Drag 'AI Terminal.app' to the Applications folder
2. Double-click to run

First Run:
- Right-click and select 'Open' to bypass security warning
- The app will check for required dependencies

Requirements:
- macOS 10.15 or later
- Python 3 (usually pre-installed)
- Ollama (optional, for AI features)

Features:
- Terminal UI with clean text flow
- Web-based terminal interface
- Instant calculations and code analysis
- Integration with Ollama models

Support:
https://github.com/Gage5977/Axis.ai
"""
    
    with open(dmg_dir / "README.txt", "w") as f:
        f.write(readme)
    
    # Create DMG
    dmg_path = base_dir / dmg_name
    if dmg_path.exists():
        dmg_path.unlink()
    
    # Build DMG using hdiutil
    dmg_cmd = f"""
hdiutil create -volname "AI Terminal" \
    -srcfolder "{dmg_dir}" \
    -ov -format UDZO \
    "{dmg_path}"
"""
    
    result = subprocess.run(dmg_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Created {dmg_name}")
        print(f"📍 Location: {dmg_path}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        # Open in Finder
        os.system(f"open -R '{dmg_path}'")
    else:
        print(f"❌ Error creating DMG: {result.stderr}")
        
    return dmg_path

def main():
    """Main function"""
    print("🚀 AI Terminal DMG Creator")
    print("=" * 40)
    
    try:
        # Create app bundle
        app_path = create_app_bundle()
        print(f"✅ Created {app_path.name}")
        
        # Create DMG
        dmg_path = create_dmg(app_path)
        
        print("\n✨ Success!")
        print("\nThe DMG installer is ready.")
        print("Users can:")
        print("1. Double-click the DMG")
        print("2. Drag AI Terminal to Applications")
        print("3. Run from Applications or Launchpad")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()