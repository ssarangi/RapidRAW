#!/usr/bin/env bash
set -e

# RapidRAW Installer & File Association Script for Linux
# Installs binary, desktop entry, icons, and associates RAW & image file types.

PREFIX="${HOME}/.local"
BIN_DIR="${PREFIX}/bin"
APP_DIR="${PREFIX}/share/applications"
ICON_DIR="${PREFIX}/share/icons/hicolor"
MIME_DIR="${PREFIX}/share/mime"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY_SOURCE="${REPO_DIR}/src-tauri/target/release/RapidRAW"

if [ ! -f "${BINARY_SOURCE}" ]; then
  echo "Error: Release binary not found at ${BINARY_SOURCE}"
  echo "Please build the release binary first using: npm run tauri build"
  exit 1
fi

echo "Installing RapidRAW..."

# 1. Install Binary
mkdir -p "${BIN_DIR}"
cp "${BINARY_SOURCE}" "${BIN_DIR}/RapidRAW"
chmod +x "${BIN_DIR}/RapidRAW"
echo "  [✓] Installed binary to ${BIN_DIR}/RapidRAW"

# 2. Install Icons
SIZES=("32x32" "64x64" "128x128")
for SIZE in "${SIZES[@]}"; do
  ICON_SRC="${REPO_DIR}/src-tauri/icons/${SIZE}.png"
  ICON_DEST_DIR="${ICON_DIR}/${SIZE}/apps"
  if [ -f "${ICON_SRC}" ]; then
    mkdir -p "${ICON_DEST_DIR}"
    cp "${ICON_SRC}" "${ICON_DEST_DIR}/io.github.CyberTimon.RapidRAW.png"
  fi
done

# High-res icon fallback/scalable
mkdir -p "${ICON_DIR}/512x512/apps"
if [ -f "${REPO_DIR}/src-tauri/icons/icon.png" ]; then
  cp "${REPO_DIR}/src-tauri/icons/icon.png" "${ICON_DIR}/512x512/apps/io.github.CyberTimon.RapidRAW.png"
fi
echo "  [✓] Installed icons to ${ICON_DIR}"

# 3. List of supported MIME types
MIME_TYPES=(
  "image/x-adobe-dng"
  "image/x-canon-crw"
  "image/x-canon-cr2"
  "image/x-canon-cr3"
  "image/x-nikon-nef"
  "image/x-nikon-nrw"
  "image/x-olympus-orf"
  "image/x-fuji-raf"
  "image/x-sony-arw"
  "image/x-sony-srf"
  "image/x-sony-sr2"
  "image/x-panasonic-raw"
  "image/x-panasonic-rw2"
  "image/x-pentax-pef"
  "image/x-sigma-x3f"
  "image/x-samsung-srw"
  "image/x-minolta-mrw"
  "image/x-kodak-kdc"
  "image/x-kodak-k25"
  "image/x-kodak-dcr"
  "image/jpeg"
  "image/png"
  "image/webp"
  "image/tiff"
  "image/gif"
  "image/bmp"
  "image/avif"
  "image/jxl"
)

MIME_TYPE_STR=$(IFS=';'; echo "${MIME_TYPES[*]};")

# 4. Create Desktop File
mkdir -p "${APP_DIR}"
DESKTOP_FILE="${APP_DIR}/io.github.CyberTimon.RapidRAW.desktop"

cat <<EOF > "${DESKTOP_FILE}"
[Desktop Entry]
Name=RapidRAW
GenericName=RAW Image Editor
Comment=Non-destructive and GPU-accelerated RAW image editor
Exec="${BIN_DIR}/RapidRAW" %F
Icon=io.github.CyberTimon.RapidRAW
Terminal=false
Type=Application
Categories=Graphics;Photography;GTK;
MimeType=${MIME_TYPE_STR}
StartupWMClass=RapidRAW
EOF

chmod +x "${DESKTOP_FILE}"
echo "  [✓] Created desktop entry at ${DESKTOP_FILE}"

# 5. Set File Associations
echo "Setting file associations..."
for MIME in "${MIME_TYPES[@]}"; do
  xdg-mime default io.github.CyberTimon.RapidRAW.desktop "${MIME}" 2>/dev/null || true
done

# 6. Update desktop & icon databases
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "${APP_DIR}" 2>/dev/null || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f -t "${ICON_DIR}" 2>/dev/null || true
fi

echo "Done! RapidRAW release build is installed and registered as the default handler for image/RAW formats."
