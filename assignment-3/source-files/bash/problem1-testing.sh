#!/bin/bash

# Benchmark script for problem1 - 3D Stencil Performance Comparison
# Usage: ./problem1-testing.sh [iterations]
# Default: 10 iterations (since each run is much longer than problem2)

ITERATIONS=${1:-20}
WORKSTATION=$(hostname)
OUTPUT_FILE="results/problem1_benchmark_$(date +%Y%m%d_%H%M%S).txt"

echo "Running benchmark for problem1 (3D Stencil)..."
echo "Workstation: $WORKSTATION"
echo "Iterations: $ITERATIONS"
echo "Note: Each iteration includes 100 timesteps on 258x258x258 grid"

# Create results directory if it doesn't exist
mkdir -p results

# Initialize totals for all kernels
baseline_total=0
omp_collapse_total=0
omp_static_total=0
unrolled_total=0
tiled_total=0
unrolled_omp_total=0
omp_simd_total=0
best_total=0

# Calculate progress step (show progress every iteration for small counts)
progress_step=1
if [ $ITERATIONS -gt 10 ]; then
    progress_step=$((ITERATIONS / 10))
fi

echo ""
echo "Starting benchmark runs..."

# Run benchmark
for ((i=1; i<=ITERATIONS; i++)); do
    if (( i % progress_step == 0 )) || (( i == ITERATIONS )); then
        echo "Progress: $i/$ITERATIONS"
    fi
    
    # Run and capture times
    output=$(./problem1-dir/220573-prob1.out 2>&1)
    
    # Extract execution times (in milliseconds)
    baseline_time=$(echo "$output" | grep "Baseline (scalar) kernel time:" | awk '{print $5}')
    omp_collapse_time=$(echo "$output" | grep "OpenMP Collapse time:" | awk '{print $4}')
    omp_static_time=$(echo "$output" | grep "OpenMP Static time:" | awk '{print $4}')
    unrolled_time=$(echo "$output" | grep "Unrolled only time:" | awk '{print $4}')
    tiled_time=$(echo "$output" | grep "Tiled only time:" | awk '{print $4}')
    unrolled_omp_time=$(echo "$output" | grep "Unrolled + OpenMP time:" | awk '{print $5}')
    omp_simd_time=$(echo "$output" | grep "OpenMP + SIMD time:" | awk '{print $5}')
    best_time=$(echo "$output" | grep "Best (Tiled + Unrolled + OpenMP) time:" | awk '{print $8}')
    
    # Add to totals (handle empty values)
    if [[ -n "$baseline_time" ]]; then
        baseline_total=$((baseline_total + baseline_time))
    fi
    if [[ -n "$omp_collapse_time" ]]; then
        omp_collapse_total=$((omp_collapse_total + omp_collapse_time))
    fi
    if [[ -n "$omp_static_time" ]]; then
        omp_static_total=$((omp_static_total + omp_static_time))
    fi
    if [[ -n "$unrolled_time" ]]; then
        unrolled_total=$((unrolled_total + unrolled_time))
    fi
    if [[ -n "$tiled_time" ]]; then
        tiled_total=$((tiled_total + tiled_time))
    fi
    if [[ -n "$unrolled_omp_time" ]]; then
        unrolled_omp_total=$((unrolled_omp_total + unrolled_omp_time))
    fi
    if [[ -n "$omp_simd_time" ]]; then
        omp_simd_total=$((omp_simd_total + omp_simd_time))
    fi
    if [[ -n "$best_time" ]]; then
        best_total=$((best_total + best_time))
    fi
done

# Calculate averages
if [ $ITERATIONS -gt 0 ]; then
    baseline_avg=$((baseline_total / ITERATIONS))
    omp_collapse_avg=$((omp_collapse_total / ITERATIONS))
    omp_static_avg=$((omp_static_total / ITERATIONS))
    unrolled_avg=$((unrolled_total / ITERATIONS))
    tiled_avg=$((tiled_total / ITERATIONS))
    unrolled_omp_avg=$((unrolled_omp_total / ITERATIONS))
    omp_simd_avg=$((omp_simd_total / ITERATIONS))
    best_avg=$((best_total / ITERATIONS))
else
    echo "Error: No valid iterations completed"
    exit 1
fi

# Calculate speedups (using bc for floating point)
if command -v bc >/dev/null 2>&1 && [ $baseline_avg -gt 0 ]; then
    omp_collapse_speedup=$(echo "scale=2; $baseline_avg / $omp_collapse_avg" | bc 2>/dev/null || echo "N/A")
    omp_static_speedup=$(echo "scale=2; $baseline_avg / $omp_static_avg" | bc 2>/dev/null || echo "N/A")
    unrolled_speedup=$(echo "scale=2; $baseline_avg / $unrolled_avg" | bc 2>/dev/null || echo "N/A")
    tiled_speedup=$(echo "scale=2; $baseline_avg / $tiled_avg" | bc 2>/dev/null || echo "N/A")
    unrolled_omp_speedup=$(echo "scale=2; $baseline_avg / $unrolled_omp_avg" | bc 2>/dev/null || echo "N/A")
    omp_simd_speedup=$(echo "scale=2; $baseline_avg / $omp_simd_avg" | bc 2>/dev/null || echo "N/A")
    best_speedup=$(echo "scale=2; $baseline_avg / $best_avg" | bc 2>/dev/null || echo "N/A")
else
    omp_collapse_speedup="N/A"
    omp_static_speedup="N/A"
    unrolled_speedup="N/A"
    tiled_speedup="N/A"
    unrolled_omp_speedup="N/A"
    omp_simd_speedup="N/A"
    best_speedup="N/A"
fi

# Display results
echo ""
echo "=== BENCHMARK RESULTS ==="
echo "Average Execution Times (milliseconds):"
echo "Baseline (scalar):           $baseline_avg ms"
echo "OpenMP Collapse:             $omp_collapse_avg ms"
echo "OpenMP Static:               $omp_static_avg ms"
echo "Unrolled only:               $unrolled_avg ms"
echo "Tiled only:                  $tiled_avg ms"
echo "Unrolled + OpenMP:           $unrolled_omp_avg ms"
echo "OpenMP + SIMD:               $omp_simd_avg ms"
echo "Best (Tiled+Unrolled+OpenMP): $best_avg ms"
echo ""
echo "Speedup vs Baseline:"
echo "OpenMP Collapse:             ${omp_collapse_speedup}x"
echo "OpenMP Static:               ${omp_static_speedup}x"
echo "Unrolled only:               ${unrolled_speedup}x"
echo "Tiled only:                  ${tiled_speedup}x"
echo "Unrolled + OpenMP:           ${unrolled_omp_speedup}x"
echo "OpenMP + SIMD:               ${omp_simd_speedup}x"
echo "Best (Tiled+Unrolled+OpenMP): ${best_speedup}x"

# Latex table output (optional)
echo "Baseline &\\texttt{$baseline_avg} & \\texttt{${baseline_speedup}}$\\times$\\\\"
echo "OpenMP with \\verb|collapse(2)| &\\texttt{$omp_collapse_avg} & \\texttt{${omp_collapse_speedup}}$\\times$\\\\"
echo "OpenMP with \\verb|schedule(static)| &\\texttt{$omp_static_avg} & \\texttt{${omp_static_speedup}}$\\times$\\\\"
echo "Unrolled-only (no OpenMP)  &\\texttt{$unrolled_avg} & \\texttt{${unrolled_speedup}}$\\times$\\\\"
echo "Tiled-only (no OpenMP)  &\\texttt{$tiled_avg} & \\texttt{${tiled_speedup}}$\\times$\\\\"
echo "Unrolled + OpenMP  &\\texttt{$unrolled_omp_avg} & \\texttt{${unrolled_omp_speedup}}$\\times$\\\\"
echo "OpenMP + SIMD  &\\texttt{$omp_simd_avg} & \\texttt{${omp_simd_speedup}}$\\times$\\\\"
echo "\\textbf{\\textit{\\small{Tiled + Unrolled + OpenMP + SIMD + Prefetch} }} &\\texttt{$best_avg} & \\texttt{${best_speedup}}\\\\"

# Save results to file
{
    echo "Problem1 (3D Stencil) Benchmark Results"
    echo "======================================="
    echo "Date: $(date)"
    echo "Workstation: $WORKSTATION"
    echo "Iterations: $ITERATIONS"
    echo "Grid Size: 258x258x258"
    echo "Timesteps per iteration: 100"
    echo "OpenMP Threads: $(nproc 2>/dev/null || echo 'Unknown')"
    echo ""
    echo "Average Execution Times (milliseconds):"
    echo "Baseline (scalar):           $baseline_avg"
    echo "OpenMP Collapse:             $omp_collapse_avg"
    echo "OpenMP Static:               $omp_static_avg"
    echo "Unrolled only:               $unrolled_avg"
    echo "Tiled only:                  $tiled_avg"
    echo "Unrolled + OpenMP:           $unrolled_omp_avg"
    echo "OpenMP + SIMD:               $omp_simd_avg"
    echo "Best (Tiled+Unrolled+OpenMP): $best_avg"
    echo ""
    echo "Speedup vs Baseline:"
    echo "OpenMP Collapse:             ${omp_collapse_speedup}x"
    echo "OpenMP Static:               ${omp_static_speedup}x"
    echo "Unrolled only:               ${unrolled_speedup}x"
    echo "Tiled only:                  ${tiled_speedup}x"
    echo "Unrolled + OpenMP:           ${unrolled_omp_speedup}x"
    echo "OpenMP + SIMD:               ${omp_simd_speedup}x"
    echo "Best (Tiled+Unrolled+OpenMP): ${best_speedup}x"
    echo ""
    echo "Performance Analysis:"
    echo "- Grid size: 258x258x258 (~17.2M points)"
    echo "- Memory footprint: ~275 MB per grid"
    echo "- Total FLOPs per timestep: ~120M operations"
    echo "- Memory bandwidth critical for performance"
    echo ""
    echo "Optimization Techniques Applied:"
    echo "1. OpenMP Parallelization (collapse/static scheduling)"
    echo "2. Loop Tiling (16x16x32 blocks for cache locality)"
    echo "3. Loop Unrolling (2x unrolling in innermost loop)"
    echo "4. SIMD Vectorization (compiler hints and intrinsics)"
    echo "5. Memory Prefetching (software prefetch hints)"
} > "$OUTPUT_FILE"

echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "Benchmark completed successfully!"

# Optional: Display best performing configuration
if [ "$best_avg" -lt "$baseline_avg" ] 2>/dev/null; then
    echo "Best performing configuration achieved ${best_speedup}x speedup!"
else
    echo "Note: Check compilation flags and system configuration if speedups are not achieved."
fi
