#!/usr/bin/env bash
# Run the production HMX FlashAttention operator in the v75 simulator.
set -euo pipefail

prism_htp=${1:?pass the Prism llama.cpp checkout path}/ggml/src/ggml-hexagon/htp
sdk=${HEXAGON_SDK_ROOT:-/opt/hexagon/6.6.0.0}
tools=${HEXAGON_TOOLS_ROOT:-$sdk/tools/HEXAGON_Tools/19.0.07}/Tools/bin
here=${HMX_FA_BENCH_SOURCE_DIR:-$(cd "$(dirname "$0")" && pwd)}
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

includes=(-I"$prism_htp" -I"$prism_htp/.." -I"$prism_htp/../.."
    -I"$prism_htp/../../include" -I"$sdk/incs" -I"$sdk/incs/stddef"
    -I"$sdk/rtos/qurt/computev75/include/qurt")
flags=(-mcpu=v75 -mv75 -mhvx=v75 -mhmx -O2 -fpic -ffunction-sections -fdata-sections)
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" \
    -DHMX_FA_KV_LEN="${HMX_FA_KV_LEN:-64}" -DHMX_FA_Q_LEN="${HMX_FA_Q_LEN:-1}" \
    -DHMX_FA_REPEATS="${HMX_FA_REPEATS:-1}" \
    -c "$here/hmx_fa_worker_sim.c" -o "$build_dir/worker.o"
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" \
    -c "$prism_htp/dma-queue.c" -o "$build_dir/dma.o"
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" \
    -c "$prism_htp/hmx-queue.c" -o "$build_dir/hmx.o"
"$tools/hexagon-clang" -mcpu=v75 -mv75 -mhvx=v75 -mhmx -Wl,--gc-sections \
    "$build_dir/worker.o" "$build_dir/dma.o" "$build_dir/hmx.o" -lm -o "$build_dir/worker.elf"
if ! output=$("$tools/hexagon-sim" --march v75na_1 -r "$build_dir/worker.elf" 2>&1); then
    printf '%s\n' "$output" >&2
    exit 1
fi
grep -F "HMX FA worker L=${HMX_FA_KV_LEN:-64} Q=${HMX_FA_Q_LEN:-1} repeats=${HMX_FA_REPEATS:-1}" <<< "$output"
grep -F 'Total: Insns=' <<< "$output"
