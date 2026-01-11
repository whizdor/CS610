#!/usr/bin/env bash

# Usage: ./build_threads.sh <source.cpp>
SRC="${1:-main.cpp}"

# Choose compiler via $CXX if you like (e.g., CXX=clang++)
CXX="${CXX:-g++}"
CXXFLAGS="-O3 -march=native -std=c++20 -Wall -Wextra -pthread"

THREADS=(1 2 4 8 16 32 64)
BUILD_DIR="build_threads"

mkdir -p "$BUILD_DIR"

echo "Compiling $SRC for NUM_THREADS in: ${THREADS[*]}"
for T in "${THREADS[@]}"; do
  OUT="$BUILD_DIR/bench_${T}"
  echo "  -> NUM_THREADS=$T"
  "$CXX" $CXXFLAGS -DNUM_THREADS="$T" "$SRC" -o "$OUT"
  if [[ "${RUN:-1}" -eq 1 ]]; then
    echo "----- Running $OUT (NUM_THREADS=$T) -----"
    "$OUT"
    echo
  fi
done

echo "Done. Binaries are in ./$BUILD_DIR/"
