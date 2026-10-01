#!/usr/bin/env bash
# Compare accepted baseline outputs under ordinary and vectorized compilation.
# Usage: run-ptq1-baseline-bits.sh /accepted/sdk/hexagon
set -euo pipefail
baseline=${1:?pass the accepted baseline SDK Hexagon include directory}
here=$(cd "$(dirname "$0")" && pwd)
candidate=$(cd "$here/../../sdk/hexagon" && pwd)
tools=${HEXAGON_TOOLS_ROOT:-/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07}/Tools/bin
build_dir=${PTQ1_BASELINE_BITS_BUILD_DIR:-$(mktemp -d /tmp/ptq1-baseline-bits.XXXXXX)}
mkdir -p "$build_dir"
test ! -e "$build_dir/base-off.log"
for arm in base candidate; do
    include=$baseline
    test "$arm" != candidate || include=$candidate
    for vectorize in off on; do
        flags=()
        test "$vectorize" != on || flags+=(-fvectorize)
        elf="$build_dir/$arm-$vectorize.elf"
        log="$build_dir/$arm-$vectorize.log"
        "$tools/hexagon-clang" -I"$include" -mcpu=v75 -mv75 -mhvx=v75 -mhmx \
            -O2 -Wall -Wextra -Werror "${flags[@]}" "$here/ptq1_baseline_bits_sim.c" -o "$elf"
        "$tools/hexagon-sim" --march v75na_1 -r "$elf" > "$log" 2>&1
        test "$(grep -c '^PTQ1_BASELINE_BITS_RAW K=' "$log")" = 27
        grep -Fxq PTQ1_BASELINE_BITS_CASES=27 "$log"
        grep '^PTQ1_BASELINE_BITS' "$log" > "$log.bits"
    done
done
for vectorize in off on; do
    cmp "$build_dir/base-$vectorize.log.bits" "$build_dir/candidate-$vectorize.log.bits"
done
printf 'PTQ1_54_BASELINE_OUTPUT_WORD_AND_PADDING_COMPARISONS_PASS\n'
printf 'Receipts: %s\n' "$build_dir"
