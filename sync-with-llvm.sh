#!/usr/bin/env bash
set -e

echo "🚀 Setting up Homebrew LLVM environment for uv sync..."

# Homebrew LLVM path
LLVM_PREFIX=$(brew --prefix llvm)

export CC="$LLVM_PREFIX/bin/clang"
export CXX="$LLVM_PREFIX/bin/clang++"
export LDFLAGS="-L$LLVM_PREFIX/lib"
export CPPFLAGS="-I$LLVM_PREFIX/include -I$LLVM_PREFIX/include/c++/v1"
export CPATH="$LLVM_PREFIX/include/c++/v1:$CPATH"

# Verify compiler sees stdexcept
echo "🔍 Verifying C++ headers..."
echo | $CXX -E -x c++ - -v | grep stdexcept -A2

echo "✅ Environment ready. Running uv sync..."
uv sync
