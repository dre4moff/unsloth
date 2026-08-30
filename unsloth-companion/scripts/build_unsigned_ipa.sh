#!/bin/zsh

set -euo pipefail

SCRIPT_DIR=${0:A:h}
COMPANION_ROOT=${SCRIPT_DIR:h}
PROJECT_DIR="$COMPANION_ROOT/Unsloth Companion"
PROJECT_FILE="$PROJECT_DIR/Unsloth Companion.xcodeproj"
RELEASE_DIR=${1:-"${COMPANION_ROOT:h}/release"}
IPA_NAME="Unsloth-Companion_0.0.1_unsigned.ipa"
DEVELOPER_DIR_PATH=${DEVELOPER_DIR:-/Applications/Xcode-beta.app/Contents/Developer}
DERIVED_DIR=$(mktemp -d /tmp/unsloth-companion-derived.XXXXXX)
PACKAGE_DIR=$(mktemp -d /tmp/unsloth-companion-ipa.XXXXXX)

cleanup() {
  for path in "$DERIVED_DIR" "$PACKAGE_DIR"; do
    if [[ -d "$path" && "$path" == /tmp/unsloth-companion-* ]]; then
      /usr/bin/trash "$path"
    fi
  done
}
trap cleanup EXIT

DEVELOPER_DIR="$DEVELOPER_DIR_PATH" xcodebuild -quiet \
  -project "$PROJECT_FILE" \
  -scheme "Unsloth Companion" \
  -configuration Release \
  -sdk iphoneos \
  -derivedDataPath "$DERIVED_DIR" \
  CODE_SIGNING_ALLOWED=NO \
  ENABLE_CODE_COVERAGE=NO \
  CLANG_COVERAGE_MAPPING=NO \
  build

APP="$DERIVED_DIR/Build/Products/Release-iphoneos/Unsloth Companion.app"
EXECUTABLE="$APP/Unsloth Companion"
FRAMEWORK="$APP/Frameworks/llama.framework"

[[ -d "$APP" && -f "$EXECUTABLE" && -f "$FRAMEWORK/llama" ]]
[[ ! -e "$APP/embedded.mobileprovision" && ! -e "$APP/_CodeSignature" ]]
[[ ! -e "$FRAMEWORK/_CodeSignature" ]]

# La normale azione build di Xcode conserva la symbol table locale anche in
# Release. La rimuoviamo prima del packaging, come avviene per un archivio di
# distribuzione, per non incorporare percorsi della macchina di compilazione.
DEVELOPER_DIR="$DEVELOPER_DIR_PATH" xcrun strip -S -x "$EXECUTABLE"

[[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$APP/Info.plist")" == "3" ]]
[[ "$(/usr/libexec/PlistBuddy -c 'Print :MinimumOSVersion' "$APP/Info.plist")" == "18.6" ]]
[[ "$(lipo -archs "$EXECUTABLE")" == "arm64" ]]
[[ "$(lipo -archs "$FRAMEWORK/llama")" == "arm64" ]]

if /usr/bin/grep -a -F -q "$PROJECT_DIR" "$EXECUTABLE"; then
  print -u2 "Il binario Release contiene il percorso locale del progetto."
  exit 1
fi

mkdir -p "$PACKAGE_DIR/Payload" "$RELEASE_DIR"
ditto "$APP" "$PACKAGE_DIR/Payload/Unsloth Companion.app"
(cd "$PACKAGE_DIR" && /usr/bin/zip -qry "$PACKAGE_DIR/$IPA_NAME" Payload)
unzip -tq "$PACKAGE_DIR/$IPA_NAME"
mv -f "$PACKAGE_DIR/$IPA_NAME" "$RELEASE_DIR/$IPA_NAME"
shasum -a 256 "$RELEASE_DIR/$IPA_NAME"
