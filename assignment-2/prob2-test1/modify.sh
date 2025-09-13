#!/usr/bin/env bash
# usage: ./replicate.sh <file> <N>   (N=2 -> a b c a b c)

set -euo pipefail
file="$1"; N="${2:-1}"

[[ -f "$file" ]] || { echo "No such file: $file" >&2; exit 1; }
[[ "$N" =~ ^[0-9]+$ && "$N" -ge 1 ]] || { echo "N must be >= 1" >&2; exit 1; }

orig="$(mktemp)"; trap 'rm -f "$orig"' EXIT
cp -- "$file" "$orig"

for ((i=1; i<N; i++)); do
  # If the current file doesn't end with a newline, add one.
  if [ -s "$file" ] && [ "$(tail -c1 -- "$file" || true)" != $'\n' ]; then
    printf '\n' >> "$file"
  fi
  cat -- "$orig" >> "$file"
done
