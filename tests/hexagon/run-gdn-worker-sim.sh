#!/usr/bin/env bash
# Run the production GDN worker with real DMA on the v75 simulator.
# Usage: run-gdn-worker-sim.sh /path/to/prism
set -euo pipefail

prism_htp=${1:?pass the Prism llama.cpp checkout path}/ggml/src/ggml-hexagon/htp
sdk=${HEXAGON_SDK_ROOT:-/opt/hexagon/6.6.0.0}
hex_tools=${HEXAGON_TOOLS_ROOT:-$sdk/tools/HEXAGON_Tools/19.0.07}/Tools/bin
here=${GDN_BENCH_SOURCE_DIR:-$(cd "$(dirname "$0")" && pwd)}
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

includes=(-I"$prism_htp" -I"$prism_htp/.." -I"$prism_htp/../.."
    -I"$sdk/incs" -I"$sdk/incs/stddef" -I"$sdk/rtos/qurt/computev75/include/qurt")
flags=(-mcpu=v75 -mv75 -mhvx=v75 -mhmx -O2 -fpic)
"$hex_tools/hexagon-clang" "${includes[@]}" "${flags[@]}" -c "$prism_htp/dma-queue.c" -o "$build_dir/queue.o"
"$hex_tools/hexagon-clang" "${includes[@]}" "${flags[@]}" -ffunction-sections -fdata-sections \
    -DGDN_S="${GDN_S:-128}" -DGDN_H="${GDN_H:-32}" -DGDN_T="${GDN_T:-1}" \
    -DGDN_VECTOR_GATE="${GDN_VECTOR_GATE:-0}" -DGDN_REPEATS="${GDN_REPEATS:-1}" \
    -c "$here/gdn_worker_sim.c" -o "$build_dir/worker.o"
"$hex_tools/hexagon-clang" "${flags[@]}" -Wl,--gc-sections \
    "$build_dir/worker.o" "$build_dir/queue.o" -lm -o "$build_dir/worker.elf"
if ! output=$("$hex_tools/hexagon-sim" --march v75na_1 -r "$build_dir/worker.elf" 2>&1); then
    printf '%s\n' "$output" >&2
    exit 1
fi
grep -F 'GDN worker S=' <<< "$output"
grep -F 'Total: Insns=' <<< "$output"
