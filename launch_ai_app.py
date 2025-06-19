#!/usr/bin/env python3
"""
macOS App Launcher for AI Terminal
Creates a native app experience
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def create_macos_app():
    """Create a macOS .app bundle"""
    
    app_name = "AI Terminal"
    bundle_name = f"{app_name}.app"
    
    # Get the directory of this script
    base_dir = Path(__file__).parent.absolute()
    
    # Create app structure
    app_path = base_dir / bundle_name
    contents = app_path / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    
    # Create directories
    for dir in [contents, macos, resources]:
        dir.mkdir(parents=True, exist_ok=True)
    
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
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.12</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>"""
    
    with open(contents / "Info.plist", "w") as f:
        f.write(info_plist)
    
    # Create launcher script
    launcher_script = f"""#!/bin/bash
cd "{base_dir}"
osascript -e 'tell application "Terminal"
    activate
    do script "cd \\"{base_dir}\\" && python3 advanced_terminal_ui.py"
end tell'
"""
    
    launcher_path = macos / "launcher"
    with open(launcher_path, "w") as f:
        f.write(launcher_script)
    
    # Make launcher executable
    os.chmod(launcher_path, 0o755)
    
    # Create icon (using emoji as placeholder)
    create_icon(resources / "AppIcon.icns")
    
    print(f"✅ Created {bundle_name}")
    print(f"📍 Location: {app_path}")
    print("\nTo install:")
    print(f"1. Drag {bundle_name} to Applications folder")
    print("2. Right-click and select 'Open' the first time (security)")
    print("\nOr run directly:")
    print(f"open '{app_path}'")
    
    return app_path

def create_icon(icon_path):
    """Create a simple icon for the app"""
    # Create temporary icon using system tools
    try:
        # Create a simple icon using sips and iconutil
        temp_dir = tempfile.mkdtemp()
        iconset = Path(temp_dir) / "AppIcon.iconset"
        iconset.mkdir()
        
        # Generate icon sizes using emoji
        sizes = [16, 32, 64, 128, 256, 512]
        
        for size in sizes:
            # Create a simple colored square as placeholder
            cmd = f"""
cat > /tmp/icon_script.py << 'EOF'
from PIL import Image, ImageDraw, ImageFont
import os

size = {size}
img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Gradient background
for i in range(size):
    color = int(255 * (1 - i/size))
    draw.rectangle([i, i, size-i, size-i], outline=(0, color, 255, 255))

# AI text
if size >= 64:
    text = "AI"
    # Try to use a system font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size//3)
    except:
        font = ImageFont.load_default()
    
    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center text
    x = (size - text_width) // 2
    y = (size - text_height) // 2
    
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

img.save('/tmp/icon_{size}.png')
EOF

python3 /tmp/icon_script.py 2>/dev/null || echo '🤖' > /tmp/icon_{size}.txt
"""
            os.system(cmd)
            
            # If PIL not available, create simple icons
            if not os.path.exists(f'/tmp/icon_{size}.png'):
                # Create a simple blue square
                os.system(f'convert -size {size}x{size} xc:"#0080FF" /tmp/icon_{size}.png 2>/dev/null || touch /tmp/icon_{size}.png')
            
            if os.path.exists(f'/tmp/icon_{size}.png'):
                os.system(f'cp /tmp/icon_{size}.png "{iconset}/icon_{size}x{size}.png"')
                if size <= 512:
                    os.system(f'cp /tmp/icon_{size}.png "{iconset}/icon_{size}x{size}@2x.png"')
        
        # Convert to icns
        os.system(f'iconutil -c icns "{iconset}" -o "{icon_path}" 2>/dev/null')
        
        # Cleanup
        os.system(f'rm -rf {temp_dir} /tmp/icon_*.png /tmp/icon_script.py')
        
    except Exception as e:
        print(f"Note: Could not create custom icon: {e}")
        # Create a basic icon file
        open(icon_path, 'wb').close()

def create_dock_integration():
    """Create scripts for dock integration"""
    
    dock_script = """#!/bin/bash
# Add AI Terminal to Dock

APP_PATH="$1"
if [ -z "$APP_PATH" ]; then
    APP_PATH="/Applications/AI Terminal.app"
fi

# Add to dock
defaults write com.apple.dock persistent-apps -array-add "<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>$APP_PATH</string><key>_CFURLStringType</key><integer>0</integer></dict></dict></dict>"

# Restart dock
killall Dock

echo "✅ Added AI Terminal to Dock"
"""
    
    with open("add_to_dock.sh", "w") as f:
        f.write(dock_script)
    os.chmod("add_to_dock.sh", 0o755)

def create_terminal_profile():
    """Create a custom Terminal profile for the app"""
    
    profile = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>name</key>
    <string>AI Terminal</string>
    <key>type</key>
    <string>Window Settings</string>
    <key>BackgroundColor</key>
    <data>YnBsaXN0MDDUAQIDBAUGBwpYJHZlcnNpb25ZJGFyY2hpdmVyVCR0b3BYJG9iamVjdHMSAAGGoF8QD05TS2V5ZWRBcmNoaXZlctEICVRyb290gAGjCwwTVSRudWxs0w0ODxARElYkY2xhc3NcTlNDb2xvclNwYWNlVU5TUkdCgAIQAU8QJDAuMDU4ODIzNTMgMC4wNTg4MjM1MyAwLjExNzY0NzA2IDEA0hQVFhdaJGNsYXNzbmFtZVgkY2xhc3Nlc1dOU0NvbG9yohYYWE5TT2JqZWN0CBEaJCkyN0lMUVNYXmdud36FjpCSlKatr7S9xsfJAAAAAAAAAQEAAAAAAAAAGQAAAAAAAAAAAAAAAAAAANE=</data>
    <key>CursorColor</key>
    <data>YnBsaXN0MDDUAQIDBAUGBwpYJHZlcnNpb25ZJGFyY2hpdmVyVCR0b3BYJG9iamVjdHMSAAGGoF8QD05TS2V5ZWRBcmNoaXZlctEICVRyb290gAGjCwwTVSRudWxs0w0ODxARElYkY2xhc3NcTlNDb2xvclNwYWNlVU5TUkdCgAIQAU8QGzAgMC41MDE5NjA4MSAwLjUwMTk2MDgxIDEA0hQVFhdaJGNsYXNzbmFtZVgkY2xhc3Nlc1dOU0NvbG9yohYYWE5TT2JqZWN0CBEaJCkyN0lMUVNYXmdud36FjpCSlKCnqauwucvM0AAAAAAAAAEBAAAAAAAAGQAAAAAAAAAAAAAAAAAA1A==</data>
    <key>Font</key>
    <data>YnBsaXN0MDDUAQIDBAUGBwpYJHZlcnNpb25ZJGFyY2hpdmVyVCR0b3BYJG9iamVjdHMSAAGGoF8QD05TS2V5ZWRBcmNoaXZlctEICVRyb290gAGkCwwVFlUkbnVsbNQNDg8QERITFFYkY2xhc3NdTlNGb250RmFtaWx5Vk5TU2l6ZVZOU05hbWWAAoADI0AuAAAAAAAAgAFfEA9NZW5sby1SZWd1bGFy0hcYGRpWJGNsYXNzWiRjbGFzc25hbWWiGhtWTlNGb250WE5TT2JqZWN0CBEaJCkyN0lMUVNYXWVud36FjpCSlKustLm8xsfJAAAAAAAAAQEAAAAAAAAAHAAAAAAAAAAAAAAAAAAAANI=</data>
    <key>FontAntialias</key>
    <true/>
    <key>FontWidthSpacing</key>
    <real>1.0</real>
    <key>ProfileCurrentVersion</key>
    <real>2.0</real>
    <key>SelectionColor</key>
    <data>YnBsaXN0MDDUAQIDBAUGBwpYJHZlcnNpb25ZJGFyY2hpdmVyVCR0b3BYJG9iamVjdHMSAAGGoF8QD05TS2V5ZWRBcmNoaXZlctEICVRyb290gAGjCwwTVSRudWxs0w0ODxARElYkY2xhc3NcTlNDb2xvclNwYWNlVU5TUkdCgAIQAU8QJDAuMTU2ODYyNzUgMC4xNTY4NjI3NSAwLjUwMTk2MDc4IDAuNQDSFBUWF1okY2xhc3NuYW1lWCRjbGFzc2VzV05TQ29sb3KiFhhYTlNPYmplY3QIERokKTI3SUxRU1hee250foCBg4WIlJmbnqCnucvM0AAAAAAAAAEBAAAAAAAAGQAAAAAAAAAAAAAAAAAA1A==</data>
    <key>ShowWindowSettingsNameInTitle</key>
    <false/>
    <key>TextBoldColor</key>
    <data>YnBsaXN0MDDUAQIDBAUGBwpYJHZlcnNpb25ZJGFyY2hpdmVyVCR0b3BYJG9iamVjdHMSAAGGoF8QD05TS2V5ZWRBcmNoaXZlctEICVRyb290gAGjCwwTVSRudWxs0w0ODxARElYkY2xhc3NcTlNDb2xvclNwYWNlVU5TUkdCgAIQAU8QGzAgMC41MDE5NjA4MSAwLjUwMTk2MDgxIDEA0hQVFhdaJGNsYXNzbmFtZVgkY2xhc3Nlc1dOU0NvbG9yohYYWE5TT2JqZWN0CBEaJCkyN0lMUVNYXmdud36FjpCSlKCnqauwuc7P0QAAAAAAAAEBAAAAAAAAGQAAAAAAAAAAAAAAAAAA1Q==</data>
    <key>TextColor</key>
    <data>YnBsaXN0MDDUAQIDBAUGBwpYJHZlcnNpb25ZJGFyY2hpdmVyVCR0b3BYJG9iamVjdHMSAAGGoF8QD05TS2V5ZWRBcmNoaXZlctEICVRyb290gAGjCwwTVSRudWxs0w0ODxARElYkY2xhc3NcTlNDb2xvclNwYWNlVU5TUkdCgAIQAU8QHDAuOTAxOTYwODEgMC45MDE5NjA4MSAwLjkwMTk2MDgxANIUFRYXWiRjbGFzc25hbWVYJGNsYXNzZXNXTlNDb2xvcqIWGFhOU09iamVjdAgRGiQpMjdJTFFTWF5nbnd+hY2PkZOgpqittr/AwgAAAAAAAAEBAAAAAAAAABkAAAAAAAAAAAAAAAAAAADG</data>
    <key>columnCount</key>
    <integer>120</integer>
    <key>rowCount</key>
    <integer>40</integer>
</dict>
</plist>"""
    
    with open("AI_Terminal.terminal", "w") as f:
        f.write(profile)
    
    print("📄 Created AI_Terminal.terminal profile")
    print("   Double-click to install the Terminal theme")

def main():
    """Main entry point"""
    print("🚀 AI Terminal App Builder")
    print("=" * 40)
    
    # Create the app
    app_path = create_macos_app()
    
    # Create additional integration files
    create_dock_integration()
    create_terminal_profile()
    
    print("\n✨ Setup complete!")
    print("\nQuick start:")
    print(f"1. Run the app: open '{app_path}'")
    print("2. Or launch directly: python3 advanced_terminal_ui.py")
    print("\nFor the best experience:")
    print("- Install the Terminal profile (double-click AI_Terminal.terminal)")
    print("- Add to Dock: ./add_to_dock.sh")

if __name__ == "__main__":
    main()