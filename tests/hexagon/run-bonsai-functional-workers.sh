#!/usr/bin/env bash
# Actual DSP workers; serial adapters do not test QuRT concurrency.
set -euo pipefail
prism=${1:?pass the patched Prism source root}
here=$(cd "$(dirname "$0")" && pwd)
sdk=${HEXAGON_SDK_ROOT:-/opt/hexagon/6.6.0.0}
tools=${HEXAGON_TOOLS_ROOT:-$sdk/tools/HEXAGON_Tools/19.0.07}/Tools/bin
htp=$prism/ggml/src/ggml-hexagon/htp
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
flags=(-mcpu=v75 -mv75 -mhvx=v75 -mhmx -O2)
inc=(-I"$here/../../sdk/hexagon" -I"$htp" -I"$prism/ggml/src"
     -I"$sdk/incs" -I"$sdk/incs/stddef" -I"$sdk/rtos/qurt/computev75/include/qurt")
compile() {
    "$tools/hexagon-clang" "${flags[@]}" "${inc[@]}" \
        -ffunction-sections -fdata-sections -c "$1" -o "$scratch/$(basename "$1").o"
}
compile "$htp/dma-queue.c"
compile "$htp/get-rows-ops.c"
compile "$here/v75_get_rows_actual_fixture.c"
compile "$here/v75_get_rows_adapters.c"
# Do not pass -fpic here: the PIC libc lacks standalone _start_main.
"$tools/hexagon-clang" "${flags[@]}" -Wl,--gc-sections \
    "$scratch/v75_get_rows_actual_fixture.c.o" "$scratch/v75_get_rows_adapters.c.o" \
    "$scratch/get-rows-ops.c.o" "$scratch/dma-queue.c.o" -lm -o "$scratch/getrows.elf"
"$tools/hexagon-sim" --march v75na_1 -r "$scratch/getrows.elf" 2>&1 | tee "$scratch/getrows.log"
grep -q '^v75 GET_ROWS actual-worker fixture passed ' "$scratch/getrows.log"
compile "$htp/matmul-ops.c"
compile "$here/bf16_f32_matmul_v75_worker.c"
"$tools/hexagon-clang" "${flags[@]}" -Wl,--gc-sections \
    "$scratch/bf16_f32_matmul_v75_worker.c.o" "$scratch/matmul-ops.c.o" \
    "$scratch/dma-queue.c.o" -lm -o "$scratch/bf16.elf"
"$tools/hexagon-sim" --march v75na_1 -r "$scratch/bf16.elf" 2>&1 | tee "$scratch/bf16.log"
grep -q '^bf16_f32_matmul_v75_worker: OK (' "$scratch/bf16.log"
