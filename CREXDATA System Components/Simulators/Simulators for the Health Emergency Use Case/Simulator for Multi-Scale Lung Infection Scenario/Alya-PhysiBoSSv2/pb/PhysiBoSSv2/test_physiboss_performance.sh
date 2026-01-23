#!/bin/bash
# test_physiboss_performance.sh
# Script to run consecutive tests of the PhysiBoSS COVID simulation and check for memory errors with valgrind.

# Set the executable and config file (edit as needed)
EXECUTABLE=./COVID19
CONFIG=./config/local.xml

# Number of consecutive runs
total_runs=5

# Output directory for logs
LOGDIR=performance_test_logs
mkdir -p "$LOGDIR"

# Run consecutive tests
for i in $(seq 1 $total_runs); do
    echo "[INFO] Run $i/$total_runs: Starting simulation..."
    logfile="$LOGDIR/run_${i}.log"
    "$EXECUTABLE" "$CONFIG" > "$logfile" 2>&1
    status=$?
    if [ $status -ne 0 ]; then
        echo "[ERROR] Simulation run $i failed. Check $logfile for details."
    else
        echo "[INFO] Simulation run $i completed successfully."
    fi

done

echo "[INFO] Running valgrind memory check on a single simulation run..."
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --error-exitcode=1 "$EXECUTABLE" "$CONFIG" > "$LOGDIR/valgrind.log" 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] Valgrind detected memory errors. See $LOGDIR/valgrind.log for details."
else
    echo "[INFO] Valgrind run completed with no errors."
fi

echo "[INFO] All tests completed."
