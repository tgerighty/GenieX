#!/usr/bin/env bash
# Run the production HMX FlashAttention operator in the v75 simulator.
set -euo pipefail

prism_htp=${1:?pass the Prism llama.cpp checkout path}/ggml/src/ggml-hexagon/htp
sdk=${HEXAGON_SDK_ROOT:-/opt/hexagon/6.6.0.0}
tools=${HEXAGON_TOOLS_ROOT:-$sdk/tools/HEXAGON_Tools/19.0.07}/Tools/bin
tool_root=${HEXAGON_TOOLS_ROOT:-$sdk/tools/HEXAGON_Tools/19.0.07}/Tools
qurt=$sdk/rtos/qurt/computev75
target=$tool_root/target/hexagon/lib/v75/G0
here=${HMX_FA_BENCH_SOURCE_DIR:-$(cd "$(dirname "$0")" && pwd)}
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

includes=(-I"$prism_htp" -I"$prism_htp/.." -I"$prism_htp/../.."
    -I"$prism_htp/../../include" -I"$sdk/incs" -I"$sdk/incs/stddef"
    -I"$sdk/rtos/qurt/computev75/include/qurt")
if [[ -n ${HMX_FA_OVERRIDE_DIR:-} ]]; then
    [[ -f $HMX_FA_OVERRIDE_DIR/flash-attn-ops.c ]] || exit 1
    includes=(-I"$HMX_FA_OVERRIDE_DIR" "${includes[@]}")
fi
flags=(-mcpu=v75 -mv75 -mhvx=v75 -mhmx -O2 -g -fpic -ffunction-sections -fdata-sections)
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" \
    -DHMX_FA_KV_LEN="${HMX_FA_KV_LEN:-64}" -DHMX_FA_Q_LEN="${HMX_FA_Q_LEN:-1}" \
    -DHMX_FA_REPEATS="${HMX_FA_REPEATS:-1}" -DHMX_FA_USE_MASK="${HMX_FA_USE_MASK:-0}" \
    -DHMX_FA_PIPELINE="${HMX_FA_PIPELINE:-0}" -DHMX_FA_MASK_PER_HEAD="${HMX_FA_MASK_PER_HEAD:-0}" \
    -c "$here/hmx_fa_worker_sim.c" -o "$build_dir/worker.o"
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" \
    -c "$prism_htp/dma-queue.c" -o "$build_dir/dma.o"
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" \
    -c "$prism_htp/hmx-queue.c" -o "$build_dir/hmx.o"
"$tools/hexagon-clang" -mcpu=v75 -mv75 -mhvx=v75 -mhmx -g -nodefaultlibs -nostdlib \
    -Wl,--section-start,.interp=0x23000000 -Wl,--dynamic-linker= -Wl,--force-dynamic \
    -Wl,-E -Wl,-z,muldefs -Wl,--whole-archive \
    "$target/init.o" "$build_dir/worker.o" "$build_dir/dma.o" "$build_dir/hmx.o" \
    "$qurt/lib/crt1.o" "$qurt/lib/debugmon.o" "$qurt/lib/libqurt.a" \
    "$target/libc.a" "$target/libqcc.a" "$target/libhexagon.a" \
    "$qurt/lib/libqurtcfs.a" "$qurt/lib/libtimer_island.a" \
    "$qurt/lib/libtimer_main.a" "$qurt/lib/libposix.a" "$target/fini.o" \
    -o "$build_dir/worker.elf"
printf '%s\n' "$qurt/debugger/lnx64/qurt_model.so" > "$build_dir/osam.cfg"
printf '%s\n' \
    "$tool_root/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=3 --freq=19200000 --cnttid=1" \
    "$tool_root/lib/iss/l2vic.so 32 0xFC910000" > "$build_dir/q6ss.cfg"
if ! output=$("$tools/hexagon-sim" --march v75na_1 --simulated_returnval \
    --usefs "$build_dir" --cosim_file "$build_dir/q6ss.cfg" \
    --l2tcm_base 0xd800 --subsystem_base 0xFC90 \
    --rtos "$build_dir/osam.cfg" "$qurt/sdksim_bin/runelf.pbn" -- "$build_dir/worker.elf" -- 2>&1); then
    printf '%s\n' "$output" >&2
    pc=$(sed -n 's/.* PC=\([0-9A-Fa-f]*\) VADDR=.*/\1/p' <<< "$output" | head -n 1)
    if [[ -n $pc ]]; then "$tools/hexagon-addr2line" -f -C -e "$build_dir/worker.elf" "0x$pc" >&2; fi
    elr=$(sed -n 's/^ELR[[:space:]]*0x\([0-9A-Fa-f]*\).*/\1/p' <<< "$output" | head -n 1)
    if [[ -n $elr ]]; then
        "$tools/hexagon-addr2line" -f -C -e "$build_dir/worker.elf" "0x$elr" >&2
        "$tools/hexagon-llvm-objdump" -d --start-address="$((16#$elr - 32))" \
            --stop-address="$((16#$elr + 32))" "$build_dir/worker.elf" >&2
    fi
    "$tools/hexagon-nm" --numeric-sort "$build_dir/worker.elf" | grep -E ' [bB] (vtcm|query|keys|values|output)$' >&2 || true
    exit 1
fi
grep -F "HMX FA worker L=${HMX_FA_KV_LEN:-64} Q=${HMX_FA_Q_LEN:-1} mask=${HMX_FA_USE_MASK:-0} per_head=${HMX_FA_MASK_PER_HEAD:-0} pipeline=${HMX_FA_PIPELINE:-0} repeats=${HMX_FA_REPEATS:-1}" <<< "$output"
grep -F 'HMX FA scalar max_error=' <<< "$output"
grep -F 'HMX FA sample repeat=' <<< "$output"
grep -F 'Total: Insns=' <<< "$output"
