#!/usr/bin/env bash
# Run the PTQ1_0 DMA comparison in a v75 Hexagon SDK container.
# Usage: run-ptq1-dma-bench.sh /path/to/prism-llama.cpp
set -euo pipefail

prism_htp=${1:?pass the Prism llama.cpp checkout path}
prism_htp=$prism_htp/ggml/src/ggml-hexagon/htp
if [[ ${2:-} == --ffn-worker && -z ${PTQ1_WORKER_N:-} ]]; then
    want_reject=${PTQ1_FFN_REJECT_CHECK:-0}
    for n in 256 1 33 81 97 129 273; do
        PTQ1_FFN_REJECT_CHECK=0 PTQ1_WORKER_N=$n PTQ1_WORKER_REPEATS=1 "$0" "${1}" --ffn-worker
    done
    PTQ1_FFN_REJECT_CHECK=0 PTQ1_WORKER_N=256 PTQ1_WORKER_REPEATS=4 "$0" "${1}" --ffn-worker
    if [[ $want_reject == 1 && ${PTQ1_FFN_FUSED:-0} == 1 ]]; then
        PTQ1_WORKER_N=256 PTQ1_WORKER_REPEATS=1 PTQ1_FFN_REJECT_CHECK=1 \
            "$0" "${1}" --ffn-worker
    fi
    exit 0
fi
if [[ ${2:-} == --worker && -z ${PTQ1_WORKER_K:-} ]]; then
    PTQ1_WORKER_K=256 "$0" "${1}" --worker
    PTQ1_WORKER_K=5120 "$0" "${1}" --worker
    if [[ -z ${PTQ1_WORKER_M:-} && -z ${PTQ1_WORKER_N:-} ]]; then
        PTQ1_WORKER_K=5120 PTQ1_WORKER_M=3 PTQ1_WORKER_N=273 "$0" "${1}" --worker
        PTQ1_WORKER_K=5120 PTQ1_WORKER_M=4 PTQ1_WORKER_N=273 "$0" "${1}" --worker
        PTQ1_WORKER_K=5120 PTQ1_WORKER_M=8 PTQ1_WORKER_N=273 "$0" "${1}" --worker
        PTQ1_WORKER_K=5120 PTQ1_WORKER_M=9 PTQ1_WORKER_N=273 "$0" "${1}" --worker
        PTQ1_WORKER_K=5120 PTQ1_WORKER_M=16 PTQ1_WORKER_N=256 "$0" "${1}" --worker
        PTQ1_WORKER_K=6144 PTQ1_WORKER_M=8 PTQ1_WORKER_N=256 "$0" "${1}" --worker
        PTQ1_WORKER_K=6144 PTQ1_WORKER_M=9 PTQ1_WORKER_N=273 "$0" "${1}" --worker
        PTQ1_WORKER_K=6144 PTQ1_WORKER_M=16 PTQ1_WORKER_N=256 "$0" "${1}" --worker
        PTQ1_WORKER_K=5120 PTQ1_WORKER_M=1 PTQ1_WORKER_N=273 "$0" "${1}" --worker
        PTQ1_WORKER_K=17408 PTQ1_WORKER_M=1 PTQ1_WORKER_N=145 "$0" "${1}" --worker
        PTQ1_WORKER_K=17408 PTQ1_WORKER_M=3 PTQ1_WORKER_N=97 "$0" "${1}" --worker
        PTQ1_WORKER_K=17408 PTQ1_WORKER_M=3 PTQ1_WORKER_N=256 "$0" "${1}" --worker
        PTQ1_WORKER_K=17408 PTQ1_WORKER_M=4 PTQ1_WORKER_N=256 "$0" "${1}" --worker
        PTQ1_WORKER_K=17408 PTQ1_WORKER_M=5 PTQ1_WORKER_N=256 "$0" "${1}" --worker
        PTQ1_WORKER_K=17408 PTQ1_WORKER_M=5 PTQ1_WORKER_N=273 "$0" "${1}" --worker
        PTQ1_WORKER_K=17408 PTQ1_WORKER_M=8 PTQ1_WORKER_N=256 "$0" "${1}" --worker
    fi
    exit 0
fi
sdk=${HEXAGON_SDK_ROOT:-/opt/hexagon/6.6.0.0}
tools=${HEXAGON_TOOLS_ROOT:-$sdk/tools/HEXAGON_Tools/19.0.07}/Tools/bin
here=${PTQ1_BENCH_SOURCE_DIR:-$(cd "$(dirname "$0")" && pwd)}
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

includes=(-I"$here/../../sdk/hexagon" -I"$prism_htp" -I"$sdk/incs" -I"$sdk/incs/stddef"
    -I"$sdk/rtos/qurt/computev75/include/qurt")
flags=(-mcpu=v75 -mv75 -mhvx=v75 -mhmx -O2)
layout_flags=()
if [[ ${2:-} == --ffn-worker && ${PTQ1_FFN_FUSED:-0} == 1 ]]; then
    layout_flags=(-DPTQ1_FFN_LAYOUT_CHECK)
fi
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" "${layout_flags[@]}" "$here/ptq1_output_layout.c" -o "$build_dir/layout.elf"
"$tools/hexagon-sim" --march v75na_1 -r "$build_dir/layout.elf" | grep -Fx 'PTQ1 output layout passed'
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" -fpic -c "$prism_htp/dma-queue.c" -o "$build_dir/queue.o"

if [[ ${2:-} == --worker ]]; then
    worker_k=${PTQ1_WORKER_K:-256}
    worker_m=${PTQ1_WORKER_M:-3}
    worker_n=${PTQ1_WORKER_N:-81}
    worker_repeats=${PTQ1_WORKER_REPEATS:-1}
    includes+=(-I"$prism_htp/.." -I"$prism_htp/../..")
    "$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" -fpic -ffunction-sections -fdata-sections \
        -DPTQ1_WORKER_K="$worker_k" -DPTQ1_WORKER_M="$worker_m" -DPTQ1_WORKER_N="$worker_n" \
        -DPTQ1_WORKER_REPEATS="$worker_repeats" -DPTQ1_HALF_EXHAUSTIVE="${PTQ1_HALF_EXHAUSTIVE:-0}" \
        -c "$here/ptq1_worker_sim.c" -o "$build_dir/worker.o"
    "$tools/hexagon-clang" "${flags[@]}" -Wl,--gc-sections \
        "$build_dir/worker.o" "$build_dir/queue.o" -lm -o "$build_dir/worker.elf"
    if ! output=$("$tools/hexagon-sim" --march v75na_1 -r "$build_dir/worker.elf" 2>&1); then
        printf '%s\n' "$output" >&2
        exit 1
    fi
    grep -F "PTQ1 worker K=$worker_k M=$worker_m repeats=$worker_repeats checksum " <<< "$output"
    if [[ ${PTQ1_HALF_EXHAUSTIVE:-0} == 1 ]]; then grep -Fx 'PTQ1 half exhaustive passed' <<< "$output"; fi
    grep -F 'Total: Insns=' <<< "$output"
    exit 0
fi

if [[ ${2:-} == --ffn-worker ]]; then
    worker_k=${PTQ1_WORKER_K:-5120}
    worker_m=${PTQ1_WORKER_M:-1}
    worker_n=${PTQ1_WORKER_N:-256}
    worker_repeats=${PTQ1_WORKER_REPEATS:-1}
    fused=${PTQ1_FFN_FUSED:-0}
    count_quant=${PTQ1_FFN_COUNT_QUANT:-1}
    reject_check=${PTQ1_FFN_REJECT_CHECK:-0}
    includes+=(-I"$prism_htp/.." -I"$prism_htp/../..")
    if [[ $count_quant == 1 ]]; then
        # Copy unchanged source so its quoted header lookup reaches the test wrapper.
        cp "$prism_htp/matmul-ops.c" "$build_dir/matmul-ops.c"
        includes=("-I$build_dir" "-I$here/ptq1_ffn_quant_wrap" "${includes[@]}")
    fi
    "$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" -fpic -ffunction-sections -fdata-sections \
        -DPTQ1_WORKER_K="$worker_k" -DPTQ1_WORKER_M="$worker_m" -DPTQ1_WORKER_N="$worker_n" \
        -DPTQ1_WORKER_REPEATS="$worker_repeats" -DPTQ1_FFN_FUSED="$fused" \
        -DPTQ1_WORKER_THREADS="${PTQ1_WORKER_THREADS:-1}" \
        -DPTQ1_FFN_COUNT_QUANT="$count_quant" -DPTQ1_FFN_REJECT_CHECK="$reject_check" \
        -c "$here/ptq1_ffn_worker_sim.c" -o "$build_dir/ffn_worker.o"
    "$tools/hexagon-clang" "${flags[@]}" -Wl,--gc-sections \
        "$build_dir/ffn_worker.o" "$build_dir/queue.o" -lm -o "$build_dir/ffn_worker.elf"
    if ! output=$("$tools/hexagon-sim" --march v75na_1 -r "$build_dir/ffn_worker.elf" 2>&1); then
        printf '%s\n' "$output" >&2
        exit 1
    fi
    if [[ $reject_check == 1 ]]; then
        grep -Fx 'PTQ1 FFN reject check passed' <<< "$output"
    else
        grep -F "PTQ1 FFN worker K=$worker_k M=$worker_m N=$worker_n repeats=$worker_repeats checksum " <<< "$output"
    fi
    grep -F 'Total: Insns=' <<< "$output"
    grep -F 'Pcycles=' <<< "$output"
    exit 0
fi

for variant in single pair; do
    define=()
    if [[ $variant == pair ]]; then define=(-DPTQ1_DMA_PAIR); fi
    "$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" -fpic "${define[@]}" \
        -c "$here/ptq1_dma_sim.c" -o "$build_dir/bench.o"
    "$tools/hexagon-clang" "${flags[@]}" "$build_dir/bench.o" "$build_dir/queue.o" -o "$build_dir/bench.elf"
    if ! output=$("$tools/hexagon-sim" --march v75na_1 -r "$build_dir/bench.elf" 2>&1); then
        printf '%s\n' "$output" >&2
        exit 1
    fi
    checksum=$(sed -n 's/.*PTQ1 DMA checksum \([^[:space:]]*\).*/\1/p' <<< "$output")
    cycles=$(sed -n 's/.*Total: Insns=[0-9]* Pcycles=\([0-9]*\).*/\1/p' <<< "$output")
    [[ -n $checksum && -n $cycles ]] || { printf '%s\n' "$output"; exit 1; }
    printf '%s: %s Pcycles; checksum %s\n' "$variant" "$cycles" "$checksum"
    if [[ $variant == single ]]; then
        single_cycles=$cycles
        single_checksum=$checksum
    else
        [[ $checksum == "$single_checksum" ]] || { echo 'DMA checksum mismatch' >&2; exit 1; }
        awk -v before="$single_cycles" -v after="$cycles" \
            'BEGIN { printf "paired change: %.2f%% Pcycles\n", 100 * (after / before - 1) }'
    fi
    verify_tiles=(32)
    if [[ $variant == pair ]]; then verify_tiles+=(1 3); fi
    for tiles in "${verify_tiles[@]}"; do
        "$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" -fpic "${define[@]}" \
            -DPTQ1_DMA_VERIFY -DPTQ1_DMA_TILES="$tiles" -c "$here/ptq1_dma_sim.c" -o "$build_dir/verify.o"
        "$tools/hexagon-clang" "${flags[@]}" "$build_dir/verify.o" "$build_dir/queue.o" -o "$build_dir/verify.elf"
        if ! output=$("$tools/hexagon-sim" --march v75na_1 -r "$build_dir/verify.elf" 2>&1); then
            printf '%s\n' "$output" >&2
            exit 1
        fi
        printf '%s: %s-tile output verification passed\n' "$variant" "$tiles"
    done
done
