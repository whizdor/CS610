#!/bin/bash
if [ $# -ne 1 ]; then
  echo "Usage: $0 <thread_num>"
  exit 1
fi

THREAD_NUM=$1

# Extract workstation number from hostname (e.g. csews25 -> 25)
HOSTNAME=$(hostname)
CSEWS_NUM=$(echo "$HOSTNAME" | sed -E 's/[^0-9]*([0-9]+)/\1/')
NAME="pthread"
BIN="bin_threads/${NAME}_${THREAD_NUM}.out"
LOG="outputs/${NAME}_${THREAD_NUM}_ws${CSEWS_NUM}.log"

if [ ! -f "$BIN" ]; then
  echo "Error: Binary $BIN not found!"
  exit 1
fi

nohup "$BIN" > "$LOG" 2>&1 &
echo "Started $BIN on $HOSTNAME with output redirected to $LOG"
