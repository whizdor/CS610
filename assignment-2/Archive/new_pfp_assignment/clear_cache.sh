python3 - <<'PY'
# Touch ~512 MB in 64-byte steps to blow past LLC
size = 512*1024*1024
b = bytearray(size)
for i in range(0, size, 64):
    b[i] = (b[i] + 1) & 0xFF
PY