#!/usr/bin/env bash
set -e

echo "🔍 Checking C++ toolchain for Python builds on macOS (Intel)..."

# Step 1: Ensure Xcode Command Line Tools exist
if ! xcode-select -p &>/dev/null; then
  echo "⚠️ Command Line Tools not found. Installing..."
  xcode-select --install
  exit 1
fi

# Step 2: Locate SDK path
SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
echo "✅ Found macOS SDK at: $SDKROOT"

# Step 3: Verify <stdexcept> exists
STD_HEADER="$SDKROOT/usr/include/c++/v1/stdexcept"
if [[ -f "$STD_HEADER" ]]; then
  echo "✅ Found stdexcept header at: $STD_HEADER"
else
  echo "❌ Could not find stdexcept in $SDKROOT/usr/include/c++/v1"
  echo "👉 Falling back to Homebrew LLVM..."
  brew install llvm
  export PATH="/usr/local/opt/llvm/bin:$PATH"
  export LDFLAGS="-L/usr/local/opt/llvm/lib"
  export CPPFLAGS="-I/usr/local/opt/llvm/include -I/usr/local/opt/llvm/include/c++/v1"
  echo "✅ LLVM installed and environment set."
fi

# Step 4: Export vars for current session
export SDKROOT
export CPATH="$SDKROOT/usr/include/c++/v1:$CPATH"

# Step 5: Show compiler info
echo "🛠 Using compiler:"
clang++ --version || echo "⚠️ clang++ not found in PATH"

echo "🚀 Environment ready. Run 'uv sync' now."
