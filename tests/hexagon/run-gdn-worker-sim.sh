#!/usr/bin/env bash
# Run the production GDN worker with real DMA on the v75 simulator.
# Usage: GDN_EXPECT_SCALAR4=0|1 run-gdn-worker-sim.sh /path/to/prism
set -euo pipefail

prism_htp=${1:?pass the Prism llama.cpp checkout path}/ggml/src/ggml-hexagon/htp
case ${GDN_EXPECT_SCALAR4:?set 0 for baseline or 1 for candidate} in
    0) expected_calls=1 ;;
    1) expected_calls=0 ;;
    *) printf 'GDN_EXPECT_SCALAR4 must be 0 or 1\n' >&2; exit 2 ;;
esac
actual_calls=$(grep -Fc 'gdn_mul_scalar_dot8_f32(row0' "$prism_htp/gated-delta-net-ops.c" || true)
if [[ "$actual_calls" != "$expected_calls" ]]; then
    printf 'GDN source patch state mismatch: expected %s scalar8 calls, found %s\n' "$expected_calls" "$actual_calls" >&2
    exit 2
fi
sdk=${HEXAGON_SDK_ROOT:-/opt/hexagon/6.6.0.0}
hex_tools=${HEXAGON_TOOLS_ROOT:-$sdk/tools/HEXAGON_Tools/19.0.07}/Tools/bin
here=${GDN_BENCH_SOURCE_DIR:-$(cd "$(dirname "$0")" && pwd)}
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

includes=(-I"$prism_htp" -I"$prism_htp/.." -I"$prism_htp/../.."
    -I"$sdk/incs" -I"$sdk/incs/stddef" -I"$sdk/rtos/qurt/computev75/include/qurt")
flags=(-mcpu=v75 -mv75 -mhvx=v75 -mhmx -O2)
"$hex_tools/hexagon-clang" "${includes[@]}" "${flags[@]}" -fpic -c "$prism_htp/dma-queue.c" -o "$build_dir/queue.o"
"$hex_tools/hexagon-clang" "${includes[@]}" "${flags[@]}" -fpic -ffunction-sections -fdata-sections \
    -DGDN_S="${GDN_S:-128}" -DGDN_H="${GDN_H:-32}" -DGDN_T="${GDN_T:-1}" \
    -DGDN_VECTOR_GATE="${GDN_VECTOR_GATE:-0}" -DGDN_REPEATS="${GDN_REPEATS:-1}" \
    -DGDN_FAULT_OUTPUT_ZERO="${GDN_FAULT_OUTPUT_ZERO:-0}" \
    -DGDN_FAULT_STATE_NO_DELTA="${GDN_FAULT_STATE_NO_DELTA:-0}" \
    -c "$here/gdn_worker_sim.c" -o "$build_dir/worker.o"
"$hex_tools/hexagon-clang" "${flags[@]}" -Wl,--gc-sections \
    "$build_dir/worker.o" "$build_dir/queue.o" -lm -o "$build_dir/worker.elf"
if ! output=$("$hex_tools/hexagon-sim" --march v75na_1 -r "$build_dir/worker.elf" 2>&1); then
    printf '%s\n' "$output" >&2
    exit 1
fi
grep -F 'GDN reference max output' <<< "$output"
grep -F 'GDN worker S=' <<< "$output"
grep -F 'Total: Insns=' <<< "$output"
