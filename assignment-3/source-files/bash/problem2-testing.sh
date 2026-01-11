#!/bin/bash

# Simple benchmark script for problem2
# Usage: ./benchmark.sh [iterations]
# Default: 1000 iterations

ITERATIONS=${1:-100}
WORKSTATION=$(hostname)
OUTPUT_FILE="results/problem2_benchmark_$(date +%Y%m%d_%H%M%S).txt"

echo "Running benchmark for problem2..."
echo "Workstation: $WORKSTATION"
echo "Iterations: $ITERATIONS"

# Initialize totals
serial_total=0
omp_total=0
sse_total=0
avx2_total=0

# Calculate progress step (show progress every 10% or every 100, whichever is larger)
progress_step=$((ITERATIONS / 10))
if [ $progress_step -gt 100 ]; then
    progress_step=10
fi

# Run benchmark
for ((i=1; i<=ITERATIONS; i++)); do
    if (( i % progress_step == 0 )) || (( i == ITERATIONS )); then
        echo "Progress: $i/$ITERATIONS"
    fi
    
    # Run and capture times
    output=$(./problem2-dir/220573-prob2.out)
    
    serial_time=$(echo "$output" | grep "Serial" | awk '{print $5}')
    omp_time=$(echo "$output" | grep "OMP" | awk '{print $5}')
    sse_time=$(echo "$output" | grep "SSE" | awk '{print $5}')
    avx2_time=$(echo "$output" | grep "AVX2" | awk '{print $5}')
    
    # Add to totals
    serial_total=$((serial_total + serial_time))
    omp_total=$((omp_total + omp_time))
    sse_total=$((sse_total + sse_time))
    avx2_total=$((avx2_total + avx2_time))
done

# Calculate averages
serial_avg=$((serial_total / ITERATIONS))
omp_avg=$((omp_total / ITERATIONS))
sse_avg=$((sse_total / ITERATIONS))
avx2_avg=$((avx2_total / ITERATIONS))

# Display results
echo ""
echo "Average Times (microseconds):"
echo "Serial: $serial_avg"
echo "OMP:    $omp_avg"
echo "SSE:    $sse_avg"
echo "AVX2:   $avx2_avg"
echo ""
echo "Speedup vs Serial:"
echo "OMP:  $(echo "scale=2; $serial_avg / $omp_avg" | bc)x"
echo "SSE:  $(echo "scale=2; $serial_avg / $sse_avg" | bc)x"
echo "AVX2: $(echo "scale=2; $serial_avg / $avx2_avg" | bc)x"


# Save results to file
{
    echo "Problem2 Benchmark Results"
    echo "========================="
    echo "Date: $(date)"
    echo "Workstation: $WORKSTATION"
    echo "Iterations: $ITERATIONS"
    echo ""
    echo "Average Execution Times (microseconds):"
    echo "Serial: $serial_avg"
    echo "OMP:    $omp_avg"
    echo "SSE:    $sse_avg"
    echo "AVX2:   $avx2_avg"
    echo ""
    echo "Speedup vs Serial:"
    echo "OMP:  ${omp_speedup}x"
    echo "SSE:  ${sse_speedup}x"
    echo "AVX2: ${avx2_speedup}x"
} > "$OUTPUT_FILE"

echo ""
echo "Results saved to: $OUTPUT_FILE"
