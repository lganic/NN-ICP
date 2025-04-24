#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <slurm-script.sh>"
  exit 1
fi

SCRIPT="$1"

# Submit the job and get the Job ID
JOB_SUBMIT_OUTPUT=$(sbatch "$SCRIPT")
echo "$JOB_SUBMIT_OUTPUT"
JOB_ID=$(echo "$JOB_SUBMIT_OUTPUT" | grep -oP '\d+$')

if [ -z "$JOB_ID" ]; then
  echo "Failed to extract job ID."
  exit 1
fi

# Extract log paths from the SBATCH lines
OUT_LOG=$(grep -oP '^#SBATCH\s+--output=\K.*' "$SCRIPT" | sed "s/%[Jj]/$JOB_ID/g")
ERR_LOG=$(grep -oP '^#SBATCH\s+--error=\K.*' "$SCRIPT" | sed "s/%[Jj]/$JOB_ID/g")

# Wait for log files to be created (up to 30s)
wait_for_file() {
  local file="$1"
  for i in {1..3000}; do
    [ -f "$file" ] && return 0
    sleep 1
  done
  echo "Warning: $file not found after 3000s"
}

echo "Waiting for logs to appear..."
wait_for_file "$OUT_LOG"
wait_for_file "$ERR_LOG"

echo "Tailing logs:"
echo "  $OUT_LOG"
echo "  $ERR_LOG"

# Start tailing logs in background
tail -f "$OUT_LOG" "$ERR_LOG" &
TAIL_PID=$!

# Wait until the job finishes
while squeue -j "$JOB_ID" 2>/dev/null | grep -q "$JOB_ID"; do
  sleep 2
done

# Kill the tail process
kill "$TAIL_PID"
echo "Job $JOB_ID completed."
