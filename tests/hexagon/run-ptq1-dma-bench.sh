#!/usr/bin/env bash
# Run the PTQ1_0 DMA comparison in a v75 Hexagon SDK container.
# Usage: run-ptq1-dma-bench.sh /path/to/prism-llama.cpp
#
# Opt-in experiment helpers (default mode unchanged when unset):
#   PTQ1_BENCH_BATCH=1          Pass -DPTQ1_BENCH_BATCH=1; require batch cycle line.
#   PTQ1_BENCH_CACHE_DIR=/abs   ccache object cache (never caches test results).
#   PTQ1_BENCH_SHARD_INDEX/COUNT  Correctness-only shard of the 18-shape --worker matrix.
set -euo pipefail

ptq1_die() { printf '%s\n' "$*" >&2; exit 1; }

ptq1_validate_config() {
    case ${PTQ1_BENCH_BATCH:-0} in 0|1) ;; *) ptq1_die "PTQ1_BENCH_BATCH must be 0 or 1" ;; esac
    if [[ ${PTQ1_BENCH_BATCH:-0} == 1 ]]; then
        [[ ( ${2:-} == --worker || ${2:-} == --ffn-worker ) &&
           ${PTQ1_FFN_REJECT_CHECK:-0} == 0 ]] ||
            ptq1_die "batch mode requires a worker run without rejection checks"
    fi
    local index=${PTQ1_BENCH_SHARD_INDEX-0} count=${PTQ1_BENCH_SHARD_COUNT-1}
    [[ $index =~ ^(0|[1-9][0-9]{0,2})$ && $count =~ ^[1-9][0-9]{0,2}$ ]] ||
        ptq1_die "shards require decimal INDEX 0..998 and COUNT 1..999"
    [[ $index -lt $count ]] || ptq1_die "shard INDEX must be less than COUNT"
    if [[ -n ${PTQ1_BENCH_SHARD_INDEX+x} || -n ${PTQ1_BENCH_SHARD_COUNT+x} ]]; then
        [[ -n ${PTQ1_BENCH_SHARD_INDEX+x} && -n ${PTQ1_BENCH_SHARD_COUNT+x} &&
           ${2:-} == --worker && -z ${PTQ1_WORKER_K:-} &&
           -z ${PTQ1_WORKER_M:-} && -z ${PTQ1_WORKER_N:-} &&
           ${PTQ1_BENCH_BATCH:-0} == 0 ]] ||
            ptq1_die "shards apply only to the correctness worker matrix"
    fi
    if [[ -n ${PTQ1_BENCH_CACHE_DIR:-} ]]; then
        [[ $PTQ1_BENCH_CACHE_DIR == /* ]] || ptq1_die "cache directory must be absolute"
        PTQ1_BENCH_CACHE_DIR=$(realpath -ms -- "$PTQ1_BENCH_CACHE_DIR")
        [[ $PTQ1_BENCH_CACHE_DIR != / &&
           $(realpath -m -- "$PTQ1_BENCH_CACHE_DIR") == "$PTQ1_BENCH_CACHE_DIR" ]] ||
            ptq1_die "cache directory must not be root or contain symlinks"
        command -v ccache >/dev/null || ptq1_die "cache requested but ccache is unavailable"
        mkdir -p "$PTQ1_BENCH_CACHE_DIR"
        [[ -d $PTQ1_BENCH_CACHE_DIR && -w $PTQ1_BENCH_CACHE_DIR ]] ||
            ptq1_die "cache directory must be writable"
    fi
}

ptq1_shard_wants() {
    [[ $(($1 % ${PTQ1_BENCH_SHARD_COUNT:-1})) -eq ${PTQ1_BENCH_SHARD_INDEX:-0} ]]
}

ptq1_clang_tu() {
    local out=$1 src=$2
    shift 2
    local -a compiler=("$tools/hexagon-clang")
    # ccache hashes source, actual included headers, flags and compiler content.
    # Cache compiled objects only; every simulator check still executes.
    if [[ -n ${PTQ1_BENCH_CACHE_DIR:-} && $out == *.o ]]; then
        compiler=(env CCACHE_DIR="$PTQ1_BENCH_CACHE_DIR" CCACHE_COMPILERCHECK=content
            CCACHE_NOHARDLINK=true CCACHE_SLOPPINESS= CCACHE_BASEDIR="$build_dir"
            ccache "$tools/hexagon-clang")
    fi
    "${compiler[@]}" "$@" "$src" -o "$out"
}

ptq1_clang_link() {
    local out=$1
    shift
    "$tools/hexagon-clang" "$@" -o "$out"
}

ptq1_extract_batch_line() {
    grep -E '^PTQ1 batch repeat1_cycles=[0-9]+ repeat4_cycles=[0-9]+$' <<<"$1"
}

prism_htp=${1:?pass the Prism llama.cpp checkout path}
prism_htp=$prism_htp/ggml/src/ggml-hexagon/htp
ptq1_validate_config "$@"
if [[ ${2:-} == --ffn-worker && -z ${PTQ1_WORKER_N:-} ]]; then
    want_reject=${PTQ1_FFN_REJECT_CHECK:-0}
    for n in 256 1 33 81 97 129 257 273; do
        PTQ1_FFN_REJECT_CHECK=0 PTQ1_WORKER_N=$n PTQ1_WORKER_REPEATS=1 "$0" "${1}" --ffn-worker
    done
    if [[ ${PTQ1_BENCH_BATCH:-0} == 0 ]]; then
        PTQ1_FFN_REJECT_CHECK=0 PTQ1_WORKER_N=256 PTQ1_WORKER_REPEATS=4 "$0" "${1}" --ffn-worker
    fi
    if [[ $want_reject == 1 && ${PTQ1_FFN_FUSED:-0} == 1 ]]; then
        PTQ1_WORKER_N=256 PTQ1_WORKER_REPEATS=1 PTQ1_FFN_REJECT_CHECK=1 \
            "$0" "${1}" --ffn-worker
    fi
    exit 0
fi
if [[ ${2:-} == --worker && -z ${PTQ1_WORKER_K:-} ]]; then
    # Correctness-only 18-shape matrix. Optional shard: keep union identical; no timing arms here.
    i=0
    if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=256 "$0" "${1}" --worker; fi; i=$((i + 1))
    if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=5120 "$0" "${1}" --worker; fi; i=$((i + 1))
    if [[ -z ${PTQ1_WORKER_M:-} && -z ${PTQ1_WORKER_N:-} ]]; then
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=5120 PTQ1_WORKER_M=3 PTQ1_WORKER_N=273 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=5120 PTQ1_WORKER_M=4 PTQ1_WORKER_N=273 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=5120 PTQ1_WORKER_M=8 PTQ1_WORKER_N=273 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=5120 PTQ1_WORKER_M=9 PTQ1_WORKER_N=273 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=5120 PTQ1_WORKER_M=16 PTQ1_WORKER_N=256 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=6144 PTQ1_WORKER_M=8 PTQ1_WORKER_N=256 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=6144 PTQ1_WORKER_M=9 PTQ1_WORKER_N=273 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=6144 PTQ1_WORKER_M=16 PTQ1_WORKER_N=256 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=5120 PTQ1_WORKER_M=1 PTQ1_WORKER_N=273 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=17408 PTQ1_WORKER_M=1 PTQ1_WORKER_N=145 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=17408 PTQ1_WORKER_M=3 PTQ1_WORKER_N=97 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=17408 PTQ1_WORKER_M=3 PTQ1_WORKER_N=256 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=17408 PTQ1_WORKER_M=4 PTQ1_WORKER_N=256 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=17408 PTQ1_WORKER_M=5 PTQ1_WORKER_N=256 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=17408 PTQ1_WORKER_M=5 PTQ1_WORKER_N=273 "$0" "${1}" --worker; fi; i=$((i + 1))
        if ptq1_shard_wants "$i"; then env -u PTQ1_BENCH_SHARD_INDEX -u PTQ1_BENCH_SHARD_COUNT PTQ1_WORKER_K=17408 PTQ1_WORKER_M=8 PTQ1_WORKER_N=256 "$0" "${1}" --worker; fi; i=$((i + 1))
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
batch_flags=()
batch_libs=()
if [[ ${PTQ1_BENCH_BATCH:-0} == 1 ]]; then
    batch_flags=(-DPTQ1_BENCH_BATCH=1)
    batch_libs=(-lhexagon)
fi
ptq1_clang_tu "$build_dir/layout.elf" "$here/ptq1_output_layout.c" \
    "${includes[@]}" "${flags[@]}" "${layout_flags[@]}"
"$tools/hexagon-sim" --march v75na_1 -r "$build_dir/layout.elf" | grep -Fx 'PTQ1 output layout passed'
ptq1_clang_tu "$build_dir/queue.o" "$prism_htp/dma-queue.c" \
    "${includes[@]}" "${flags[@]}" -fpic -c

if [[ ${2:-} == --worker ]]; then
    worker_k=${PTQ1_WORKER_K:-256}
    worker_m=${PTQ1_WORKER_M:-3}
    worker_n=${PTQ1_WORKER_N:-81}
    worker_repeats=${PTQ1_WORKER_REPEATS:-1}
    includes+=(-I"$prism_htp/.." -I"$prism_htp/../..")
    ptq1_clang_tu "$build_dir/worker.o" "$here/ptq1_worker_sim.c" \
        "${includes[@]}" "${flags[@]}" -fpic -ffunction-sections -fdata-sections \
        -DPTQ1_WORKER_K="$worker_k" -DPTQ1_WORKER_M="$worker_m" -DPTQ1_WORKER_N="$worker_n" \
        -DPTQ1_WORKER_REPEATS="$worker_repeats" -DPTQ1_HALF_EXHAUSTIVE="${PTQ1_HALF_EXHAUSTIVE:-0}" \
        "${batch_flags[@]}" -c
    ptq1_clang_link "$build_dir/worker.elf" "${flags[@]}" -Wl,--gc-sections \
        "$build_dir/worker.o" "$build_dir/queue.o" -lm "${batch_libs[@]}"
    if ! output=$("$tools/hexagon-sim" --march v75na_1 -r "$build_dir/worker.elf" 2>&1); then
        printf '%s\n' "$output" >&2
        exit 1
    fi
    if [[ ${PTQ1_BENCH_BATCH:-0} == 1 ]]; then
        ptq1_extract_batch_line "$output"
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
        # ponytail: the temporary include path prevents worker cache hits in this
        # counter-only fixture; use COUNT_QUANT=0 for timings, keep counters for checks.
        cp "$prism_htp/matmul-ops.c" "$build_dir/matmul-ops.c"
        includes=("-I$build_dir" "-I$here/ptq1_ffn_quant_wrap" "${includes[@]}")
    fi
    ptq1_clang_tu "$build_dir/ffn_worker.o" "$here/ptq1_ffn_worker_sim.c" \
        "${includes[@]}" "${flags[@]}" -fpic -ffunction-sections -fdata-sections \
        -DPTQ1_WORKER_K="$worker_k" -DPTQ1_WORKER_M="$worker_m" -DPTQ1_WORKER_N="$worker_n" \
        -DPTQ1_WORKER_REPEATS="$worker_repeats" -DPTQ1_FFN_FUSED="$fused" \
        -DPTQ1_WORKER_THREADS="${PTQ1_WORKER_THREADS:-1}" \
        -DPTQ1_FFN_COUNT_QUANT="$count_quant" -DPTQ1_FFN_REJECT_CHECK="$reject_check" \
        "${batch_flags[@]}" -c
    ptq1_clang_link "$build_dir/ffn_worker.elf" "${flags[@]}" -Wl,--gc-sections \
        "$build_dir/ffn_worker.o" "$build_dir/queue.o" -lm "${batch_libs[@]}"
    if ! output=$("$tools/hexagon-sim" --march v75na_1 -r "$build_dir/ffn_worker.elf" 2>&1); then
        printf '%s\n' "$output" >&2
        exit 1
    fi
    if [[ ${PTQ1_BENCH_BATCH:-0} == 1 ]]; then
        ptq1_extract_batch_line "$output"
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
    ptq1_clang_tu "$build_dir/bench.o" "$here/ptq1_dma_sim.c" \
        "${includes[@]}" "${flags[@]}" -fpic "${define[@]}" -c
    ptq1_clang_link "$build_dir/bench.elf" "${flags[@]}" \
        "$build_dir/bench.o" "$build_dir/queue.o"
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
        ptq1_clang_tu "$build_dir/verify.o" "$here/ptq1_dma_sim.c" \
            "${includes[@]}" "${flags[@]}" -fpic "${define[@]}" \
            -DPTQ1_DMA_VERIFY -DPTQ1_DMA_TILES="$tiles" -c
        ptq1_clang_link "$build_dir/verify.elf" "${flags[@]}" \
            "$build_dir/verify.o" "$build_dir/queue.o"
        if ! output=$("$tools/hexagon-sim" --march v75na_1 -r "$build_dir/verify.elf" 2>&1); then
            printf '%s\n' "$output" >&2
            exit 1
        fi
        printf '%s: %s-tile output verification passed\n' "$variant" "$tiles"
    done
done
