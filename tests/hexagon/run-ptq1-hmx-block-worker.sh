#!/usr/bin/env bash
# Build one product PTQ1 HMX worker fixture with the real v75 SDK.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
product_root=$(cd "$here/../.." && pwd)
prism_htp=${1:?pass the Prism llama.cpp checkout path}/ggml/src/ggml-hexagon/htp
ptq1_inc=${2:-$product_root/sdk/hexagon}
[[ -f $ptq1_inc/ptq1_hvx.h && -f $ptq1_inc/ptq1_tile.h &&
   -f $ptq1_inc/geniex_ptq1_hmx_block.h ]] || {
    printf 'PTQ1 header directory must contain the accepted and product headers\n' >&2
    exit 2
}
case_id=${PTQ1_BREADTH_CASE:-0}
shape_k=${PTQ1_WORKER_K:-5120}
shape_m=${PTQ1_WORKER_M:-30}
shape_n=${PTQ1_WORKER_N:-256}
threads=${PTQ1_WORKER_THREADS:-4}
aux_reject=${PTQ1_HMX_AUX_REJECT:-0}
queue_error_only=${PTQ1_QUEUE_ERROR_ONLY:-0}
[[ $shape_k =~ ^(5120|6144|17408)$ && $shape_m =~ ^([1-9]|[12][0-9]|3[0-3])$ && $shape_n =~ ^[1-9][0-9]*$ &&
   $threads =~ ^[1-4]$ && $aux_reject =~ ^[01]$ && $queue_error_only =~ ^[01]$ ]] || {
    printf 'K must be 5120, 6144 or 17408; M must be 1..33, N positive, threads 1..4, auxiliary rejection and queue-error mode 0 or 1\n' >&2
    exit 2
}
case $case_id in
    0|1|2|3|4) ;;
    *) printf 'PTQ1_BREADTH_CASE must be 0..4\n' >&2; exit 2 ;;
esac
if [[ $queue_error_only == 1 && ! ( ( $shape_k == 5120 || $shape_k == 6144 ) && $case_id == 0 && $shape_m == 30 && $shape_n == 256 && $threads == 4 && $aux_reject == 0 ) ]]; then
    printf 'queue-error mode requires case0 K5120 or K6144/M30/N256/T4 and full HMX scratch\n' >&2
    exit 2
fi
expected_hash=${PTQ1_BREADTH_EXPECT_HASH:-}
if [[ $shape_k == 5120 && $case_id == 0 && $shape_m == 30 && $shape_n == 256 && $threads == 4 ]]; then
    [[ -z $expected_hash || $expected_hash == c8c29a3e05b6b36f ]] || exit 2
    expected_hash=c8c29a3e05b6b36f
elif [[ $shape_k == 5120 && $case_id == 0 && $shape_m == 30 && $shape_n == 33 && $threads == 1 ]]; then
    [[ -z $expected_hash || $expected_hash == 389fd22bb16290e1 ]] || exit 2
    expected_hash=389fd22bb16290e1
elif [[ $shape_k == 5120 && $case_id == 0 && $shape_m == 30 && $shape_n == 64 && $threads == 1 ]]; then
    [[ -z $expected_hash || $expected_hash == 2744204fe1718231 ]] || exit 2
    expected_hash=2744204fe1718231
elif [[ -n $expected_hash ]]; then
    [[ $expected_hash =~ ^[0-9a-f]{16}$ ]] || {
        printf 'PTQ1_BREADTH_EXPECT_HASH must be 16 lowercase hex digits\n' >&2
        exit 2
    }
fi
if [[ $shape_k != 5120 && ( $expected_hash == c8c29a3e05b6b36f ||
      $expected_hash == 389fd22bb16290e1 || $expected_hash == 2744204fe1718231 ) ]]; then
    printf 'Another K cannot use a pinned K5120 accepted-control hash\n' >&2
    exit 2
fi
sdk=${HEXAGON_SDK_ROOT:-/opt/hexagon/6.6.0.0}
tool_root=${HEXAGON_TOOLS_ROOT:-$sdk/tools/HEXAGON_Tools/19.0.07}/Tools
tools=$tool_root/bin
qurt=$sdk/rtos/qurt/computev75
target=$tool_root/target/hexagon/lib/v75/G0
[[ -f $prism_htp/matmul-ops.c && -f $prism_htp/dma-queue.c &&
   -f $prism_htp/work-queue.c && -f $prism_htp/hmx-queue.c ]] || exit 2
[[ -x $tools/hexagon-clang && -x $tools/hexagon-sim ]] || exit 2
check_sha() { printf '%s  %s\n' "$1" "$2" | sha256sum -c -; }
# The matmul hash is the strict-applied product patch postimage, not a prototype source.
check_sha 750bd876bf2c0bfd63cf7fa4ab72364cd9289a3fe666861989557281dd046786 "$product_root/sdk/patches/prism-ptq1-hexagon.patch"
check_sha c7722839fb5253a5b5ef63eb87011b19371c01815b67867f7aa5913c0a6c7bf5 "$ptq1_inc/ptq1_hvx.h"
check_sha a3d7f83e2391054289301d1bdec927894eebc038440b07499798cb7494845dbb "$ptq1_inc/ptq1_tile.h"
check_sha 733de6fc74696ce4c087c5002f42dfbd7c7b2fa2f3b1b016cb8dc8f19c1d44ee "$ptq1_inc/geniex_ptq1_hmx_block.h"
check_sha d9a2bf4117487a353deced8a67a7df01c6134b766ceec156de7a11bd5f9c73ac "$prism_htp/matmul-ops.c"
check_sha 95a80e54faf17e3b9fb32c70134fe0065e4cebb32f2e43068ef9a7dcb5cbfa3f "$here/ptq1_hmx_block_worker_sim.c"
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/geniex-ptq1-hmx-block-worker.XXXXXX")
exec > >(tee "$build_dir/run.log") 2>&1
printf 'build_dir=%s\nsource=%s\ncase=%s\nshape_k=%s\nshape_m=%s\nshape_n=%s\n' "$build_dir" "$prism_htp" "$case_id" "$shape_k" "$shape_m" "$shape_n"
printf 'expected_baseline_hash=%s\n' "${expected_hash:-discovery-only}"

includes=(-I"$tool_root/target/hexagon/include" -I"$here" -I"$ptq1_inc" -I"$prism_htp" -I"$prism_htp/.." -I"$prism_htp/../.."
    -I"$prism_htp/../../include" -I"$sdk/incs" -I"$sdk/incs/stddef"
    -I"$sdk/rtos/qurt/computev75/include/qurt")
flags=(-mcpu=v75 -mv75 -mhvx=v75 -mhmx -O2 -g -fvectorize -flto
    -Wall -Werror -fno-zero-initialized-in-bss -G0 -fdata-sections
    -ffunction-sections -fpic)
flags+=("-DPTQ1_BREADTH_CASE=$case_id")
flags+=("-DPTQ1_WORKER_K=$shape_k" "-DPTQ1_WORKER_M=$shape_m" "-DPTQ1_WORKER_N=$shape_n")
flags+=("-DPTQ1_QUEUE_ERROR_ONLY=$queue_error_only")
flags+=(-DPTQ1_EXPECT_DIRECT_PACK=1)
flags+=("-DPTQ1_WORKER_THREADS=$threads" "-DPTQ1_HMX_AUX_REJECT=$aux_reject")
if [[ -n $expected_hash ]]; then
    flags+=("-DPTQ1_BREADTH_EXPECT_HASH=0x${expected_hash}ULL")
fi
if [[ $case_id == 2 || $case_id == 4 ]]; then
    flags+=(-DPTQ1_WORKER_RANDOM_ACT=1)
fi
if [[ $case_id == 3 ]]; then
    flags+=(-DPTQ1_WORKER_BIAS=0)
fi
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" \
    -c "$here/ptq1_hmx_block_worker_sim.c" -o "$build_dir/fixture.o"
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" \
    -c "$prism_htp/hmx-queue.c" -o "$build_dir/hmx-queue.o"
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" \
    -c "$prism_htp/dma-queue.c" -o "$build_dir/dma-queue.o"
"$tools/hexagon-clang" "${includes[@]}" "${flags[@]}" \
    -c "$prism_htp/work-queue.c" -o "$build_dir/work-queue.o"
"$tools/hexagon-clang" -mcpu=v75 -mv75 -mhvx=v75 -mhmx -flto -g \
    -nodefaultlibs -nostdlib -Wl,--gc-sections \
    -Wl,--section-start,.interp=0x23000000 -Wl,--dynamic-linker= \
    -Wl,--force-dynamic -Wl,-z,muldefs -Wl,--whole-archive \
    "$target/init.o" "$build_dir/fixture.o" "$build_dir/hmx-queue.o" \
    "$build_dir/dma-queue.o" "$build_dir/work-queue.o" \
    "$qurt/lib/crt1.o" "$qurt/lib/debugmon.o" "$qurt/lib/libqurt.a" \
    "$target/libc.a" "$target/libqcc.a" "$target/libhexagon.a" \
    "$qurt/lib/libqurtcfs.a" "$qurt/lib/libtimer_island.a" \
    "$qurt/lib/libtimer_main.a" "$qurt/lib/libposix.a" "$target/fini.o" \
    -o "$build_dir/fixture.elf"
if [[ ${3:-} == --build-only ]]; then
    "$tools/hexagon-llvm-objdump" -d "$build_dir/fixture.elf" > "$build_dir/fixture.disasm"
    sha256sum "$build_dir/fixture.elf" "$build_dir/fixture.disasm"
    printf 'HMX_BREADTH_BUILD_ONLY_NOT_RUN case=%s %s\n' "$case_id" "$build_dir"
    exit 0
fi
printf '%s\n' "$qurt/debugger/lnx64/qurt_model.so" > "$build_dir/osam.cfg"
printf '%s\n' \
    "$tool_root/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=3 --freq=19200000 --cnttid=1" \
    "$tool_root/lib/iss/l2vic.so 32 0xFC910000" > "$build_dir/q6ss.cfg"
if timeout 600s "$tools/hexagon-sim" --march v75na_1 --simulated_returnval \
    --usefs "$build_dir" --cosim_file "$build_dir/q6ss.cfg" \
    --l2tcm_base 0xd800 --subsystem_base 0xFC90 \
    --rtos "$build_dir/osam.cfg" "$qurt/sdksim_bin/runelf.pbn" \
    -- "$build_dir/fixture.elf" -- > "$build_dir/sim.log" 2>&1; then
    cat "$build_dir/sim.log"
else
    status=$?
    cat "$build_dir/sim.log"
    exit "$status"
fi
if [[ -n $expected_hash ]]; then
    grep -Fxq "PASS accepted baseline hash $expected_hash before candidate case=$case_id pinned" "$build_dir/sim.log"
else
    grep -Eq "^PASS accepted baseline hash [0-9a-f]{16} before candidate case=$case_id discovery$" "$build_dir/sim.log"
fi
grep -Fxq 'PTQ1_HMX_SLOT_NEGATIVE_CONTROL_PASS' "$build_dir/sim.log"
grep -Fxq 'PTQ1_HMX_SLOT_BYTES_PASS cases=429 scratch_bytes=110592 source_unchanged=1' "$build_dir/sim.log"
[[ $(grep -Ec '^PTQ1_HMX_SLOT_BYTES rows=([0-9]|[12][0-9]|3[012]) part=(0|[1-9][0-9]*) slot=[0-7] bytes=2048 hash=[0-9a-f]{16}$' "$build_dir/sim.log") == 429 ]]
q8_stride=$((shape_k + (((shape_k / 16 + 127) / 128) * 128)))
q8_bytes=$(( (shape_m <= 32 ? shape_m : 1) * q8_stride ))
if [[ $shape_k == 6144 ]]; then
    grep -Fxq 'PTQ1_HMX_NO_QUEUE_FALLBACK_PASS output_padding_last_row_q8_exact=1' "$build_dir/sim.log"
    if (( aux_reject || shape_m < 4 || shape_m > 32 )); then
        q8_bytes=$q8_stride
    else
        grep -Fxq "PTQ1_HMX_Q8_BYTES_EXACT_PASS bytes=$q8_bytes" "$build_dir/sim.log"
    fi
fi
grep -Eq "^PTQ1_BREADTH_Q8 case=$case_id bytes=$q8_bytes accepted=[0-9a-f]{16} candidate=[0-9a-f]{16}$" "$build_dir/sim.log"
grep -Fxq 'PTQ1_HMX_ADMISSION_BOUNDARIES_PASS' "$build_dir/sim.log"
if (( aux_reject || shape_m < 4 || shape_m > 32 )); then
    grep -Fxq 'PTQ1_HMX_ROUTE jobs=0 expected_jobs=0 active_workers=0 unexpected=0' "$build_dir/sim.log"
    grep -Fxq "PTQ1_HMX_AUX_FALLBACK forced=$aux_reject untouched=1" "$build_dir/sim.log"
else
    partition_rows=$(( ((shape_n + threads - 1) / threads + 31) / 32 * 32 ))
    active_workers=$(( (shape_n + partition_rows - 1) / partition_rows ))
    jobs=$(( (shape_n + 31) / 32 * (shape_k / 128) ))
    grep -Fxq "PTQ1_HMX_ROUTE jobs=$jobs expected_jobs=$jobs active_workers=$active_workers unexpected=0" "$build_dir/sim.log"
    grep -Fxq 'PTQ1_HMX_AUX_FALLBACK forced=0 untouched=0' "$build_dir/sim.log"
fi
if [[ $case_id == 1 ]]; then
    tile_blocks=$(( (shape_n + 31) / 32 * (shape_k / 128) ))
    grep -Fxq "PTQ1_BREADTH_SEED_DIFF tiles=$tile_blocks/$tile_blocks" "$build_dir/sim.log"
fi
grep -Fq 'PASS changed output bit rejected and restored' "$build_dir/sim.log"
grep -Fxq "PASS candidate warmup $((shape_m * shape_n)) active output words, padding, and VTCM guards" "$build_dir/sim.log"
case_bias=1
[[ $case_id == 3 ]] && case_bias=0
marker=DISCOVERY_MATCH
[[ -n $expected_hash ]] && marker=PASS
grep -Eq "^PTQ1_BREADTH_$marker case=$case_id K=$shape_k M=$shape_m N=$shape_n threads=$threads bias=$case_bias q8stride=$q8_stride checksum .* hash=[0-9a-f]{16}$" \
    "$build_dir/sim.log"
if [[ -n $expected_hash ]]; then
    grep -Eq "^PTQ1_BREADTH_PASS case=$case_id .* hash=$expected_hash$" "$build_dir/sim.log"
fi
if [[ $queue_error_only == 1 ]]; then
    grep -Fxq 'PTQ1_T4_TIMING_SKIPPED_SHAPE_GATE_ONLY' "$build_dir/sim.log"
    for mode in 1 2; do
        grep -Fxq "PTQ1_QUEUE_ERROR mode=$mode status=INTERNAL_ERR consumed=1 drained=1" "$build_dir/sim.log"
        grep -Fxq "PTQ1_QUEUE_RECOVERY mode=$mode status=OK output_q8_guards_exact=1" "$build_dir/sim.log"
    done
elif [[ $shape_k != 6144 && -n $expected_hash && $case_id == 0 && $shape_m == 30 && $shape_n == 256 && $threads == 4 && $aux_reject == 0 ]]; then
    timing_lines=$(grep -Ec '^PTQ1_T4_WORKER_PCYCLES pair=[123] turn=[12] arm=(base|candidate) cycles=[1-9][0-9]*$' "$build_dir/sim.log")
    [[ $timing_lines == 6 ]]
    for spec in '1 1 base' '1 2 candidate' '2 1 candidate' '2 2 base' '3 1 base' '3 2 candidate'; do
        read -r pair turn arm <<< "$spec"
        grep -Eq "^PTQ1_T4_WORKER_PCYCLES pair=$pair turn=$turn arm=$arm cycles=[1-9][0-9]*$" "$build_dir/sim.log"
    done
    grep -Fxq 'PTQ1_T4_PAIRED_EXACT_PASS_INSTRUMENTED_NOT_BOARD_SPEED' "$build_dir/sim.log"
else
    grep -Fxq 'PTQ1_T4_TIMING_SKIPPED_SHAPE_GATE_ONLY' "$build_dir/sim.log"
fi
