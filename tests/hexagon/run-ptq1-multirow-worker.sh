#!/usr/bin/env bash
# Functional multirow PTQ1 worker gates (production op_matmul).
# Reference runner pattern only: tests/hexagon/run-ptq1-dma-bench.sh
# Usage:
#   run-ptq1-multirow-worker.sh /path/to/candidate
#   PTQ1_WORKER_K=256 PTQ1_WORKER_M=3 PTQ1_WORKER_N=81 run-ptq1-multirow-worker.sh /path/to/candidate
set -euo pipefail

cand=${1:?pass the functional-rowchunk patched Prism source root}
htp=$cand/ggml/src/ggml-hexagon/htp
sdk_inc=$(cd "$(dirname "$0")/../../sdk/hexagon" && pwd)
here=$(cd "$(dirname "$0")" && pwd)

sdk=${HEXAGON_SDK_ROOT:-/opt/hexagon/6.6.0.0}
tools=${HEXAGON_TOOLS_ROOT:-$sdk/tools/HEXAGON_Tools/19.0.07}/Tools/bin
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

includes=(-I"$sdk_inc" -I"$htp" -I"$htp/.." -I"$htp/../.." -I"$sdk/incs" -I"$sdk/incs/stddef"
    -I"$sdk/rtos/qurt/computev75/include/qurt")
flags=(-mcpu=v75 -mv75 -mhvx=v75 -mhmx -O2)

run_one() {
    local k=$1 m=$2 n=$3 threads=$4 bias=$5 reject=$6
    "$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" -fpic -ffunction-sections -fdata-sections -c \
        -DPTQ1_WORKER_K="$k" -DPTQ1_WORKER_M="$m" -DPTQ1_WORKER_N="$n" \
        -DPTQ1_WORKER_THREADS="$threads" -DPTQ1_WORKER_BIAS="$bias" \
        -DPTQ1_WORKER_VTCM_REJECT="$reject" \
        "$here/ptq1_multirow_worker_sim.c" -o "$build_dir/worker.o"
    "$tools/hexagon-clang" "${flags[@]}" -c "$htp/dma-queue.c" -o "$build_dir/queue.o" \
        "${includes[@]}" -fpic
    "$tools/hexagon-clang" "${flags[@]}" -Wl,--gc-sections -o "$build_dir/worker.elf" \
        "$build_dir/worker.o" "$build_dir/queue.o" -lm
    local output
    if ! output=$("$tools/hexagon-sim" --march v75na_1 -r "$build_dir/worker.elf" 2>&1); then
        printf '%s\n' "$output" >&2
        exit 1
    fi
    if [[ $reject == 1 ]]; then
        grep -Fx 'PTQ1 multirow VTCM reject passed' <<<"$output"
    else
        grep -F "PTQ1 multirow op_matmul K=$k M=$m N=$n threads=$threads bias=$bias checksum " <<<"$output"
    fi
    grep -F 'Total: Insns=' <<<"$output" || true
}

if [[ -n ${PTQ1_WORKER_K:-} || -n ${PTQ1_WORKER_M:-} || -n ${PTQ1_WORKER_N:-} ]]; then
    run_one "${PTQ1_WORKER_K:-256}" "${PTQ1_WORKER_M:-3}" "${PTQ1_WORKER_N:-81}" \
        "${PTQ1_WORKER_THREADS:-4}" "${PTQ1_WORKER_BIAS:-1}" "${PTQ1_WORKER_VTCM_REJECT:-0}"
    exit 0
fi

# Default correctness matrix (not board tokens/s; simulator only).
run_one 256 1 81 4 1 0
run_one 256 3 81 4 1 0
run_one 256 25 81 4 1 0
run_one 256 512 81 4 1 0
run_one 256 3 81 4 0 0
run_one 256 3 97 4 1 0
run_one 256 3 273 4 1 0
run_one 256 3 81 4 1 1
run_one 17408 1 81 4 1 0
run_one 17408 3 97 4 1 0

echo 'PTQ1 multirow op_matmul matrix passed'
