#!/bin/bash

LOG_DIR="logs"
KEEP_DIR="example"

if [ ! -d "$LOG_DIR" ]; then
  echo "Log directory $LOG_DIR does not exist."
  exit 1
fi

echo "Deleting all contents of $LOG_DIR except $KEEP_DIR..."

find "$LOG_DIR" -mindepth 1 \
  -path "$LOG_DIR/$KEEP_DIR" -prune -o \
  -exec rm -rf {} +

echo "Done."
