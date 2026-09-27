#!/usr/bin/env bash
set -euo pipefail

prism=${1:?pass the Prism llama.cpp checkout path}
mode=${2:-correctness}
sdk=${HEXAGON_SDK_ROOT:-/opt/hexagon/6.6.0.0}
tools=${HEXAGON_TOOLS_ROOT:-$sdk/tools/HEXAGON_Tools/19.0.07}/Tools/bin
here=$(cd "$(dirname "$0")" && pwd)
htp=$prism/ggml/src/ggml-hexagon/htp
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT

includes=(-I"$here/../../sdk/hexagon" -I"$htp" -I"$sdk/incs" -I"$sdk/incs/stddef"
    -I"$sdk/rtos/qurt/computev75/include/qurt")
flags=(-mcpu=v75 -mv75 -mhvx=v75 -mhmx -O2 -fpic -ffunction-sections -fdata-sections)
defines=()
if [[ $mode == --bench ]]; then
    width=${P13_BENCH_WIDTH:-5120}
    repeats=${P13_BENCH_REPEATS:-100}
    rows=${P13_BENCH_ROWS:-3}
    pad=${P13_BENCH_PAD:-0}
    defines=(-DP13_BENCH_WIDTH="$width" -DP13_BENCH_REPEATS="$repeats" -DP13_ROWS="$rows" -DP13_BENCH_PAD="$pad")
fi
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" "${defines[@]}" -I"$htp/.." -I"$htp/../.." \
    -c "$here/set_rows_f16_f32_sim.c" -o "$build/set-rows.o"
"$tools/hexagon-clang" -mcpu=v75 -mv75 -mhvx=v75 -mhmx -O2 -Wl,--gc-sections \
    "$build/set-rows.o" -lm -o "$build/set-rows.elf"
"$tools/hexagon-sim" --march v75na_1 -r "$build/set-rows.elf"
