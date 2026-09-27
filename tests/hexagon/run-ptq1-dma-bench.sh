#!/usr/bin/env bash
# Run the PTQ1_0 DMA comparison in a v75 Hexagon SDK container.
# Usage: run-ptq1-dma-bench.sh /path/to/prism-llama.cpp
set -euo pipefail

prism_htp=${1:?pass the Prism llama.cpp checkout path}
prism_htp=$prism_htp/ggml/src/ggml-hexagon/htp
sdk=${HEXAGON_SDK_ROOT:-/opt/hexagon/6.6.0.0}
tools=${HEXAGON_TOOLS_ROOT:-$sdk/tools/HEXAGON_Tools/19.0.07}/Tools/bin
here=${PTQ1_BENCH_SOURCE_DIR:-$(cd "$(dirname "$0")" && pwd)}
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

includes=(-I"$here/../../sdk/hexagon" -I"$prism_htp" -I"$sdk/incs" -I"$sdk/incs/stddef"
    -I"$sdk/rtos/qurt/computev75/include/qurt")
flags=(-mcpu=v75 -mv75 -mhvx=v75 -mhmx -O2)
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" "$here/ptq1_output_layout.c" -o "$build_dir/layout.elf"
"$tools/hexagon-sim" --march v75na_1 -r "$build_dir/layout.elf" | grep -Fx 'PTQ1 output layout passed'
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" -fpic -c "$prism_htp/dma-queue.c" -o "$build_dir/queue.o"

if [[ ${2:-} == --worker ]]; then
    worker_k=${PTQ1_WORKER_K:-256}
    worker_m=${PTQ1_WORKER_M:-3}
    worker_repeats=${PTQ1_WORKER_REPEATS:-1}
    includes+=(-I"$prism_htp/.." -I"$prism_htp/../..")
    "$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" -fpic -ffunction-sections -fdata-sections \
        -DPTQ1_WORKER_K="$worker_k" -DPTQ1_WORKER_M="$worker_m" -DPTQ1_WORKER_REPEATS="$worker_repeats" \
        -c "$here/ptq1_worker_sim.c" -o "$build_dir/worker.o"
    "$tools/hexagon-clang" "${flags[@]}" -Wl,--gc-sections \
        "$build_dir/worker.o" "$build_dir/queue.o" -lm -o "$build_dir/worker.elf"
    if ! output=$("$tools/hexagon-sim" --march v75na_1 -r "$build_dir/worker.elf" 2>&1); then
        printf '%s\n' "$output" >&2
        exit 1
    fi
    grep -F "PTQ1 worker K=$worker_k M=$worker_m repeats=$worker_repeats checksum " <<< "$output"
    grep -F 'Total: Insns=' <<< "$output"
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
