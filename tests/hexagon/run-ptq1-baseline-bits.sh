#!/usr/bin/env bash
# Compare accepted baseline and quad outputs under ordinary, vectorized and LTO compilation.
# Usage: run-ptq1-baseline-bits.sh /accepted/sdk/hexagon
set -euo pipefail
baseline=${1:?pass the accepted baseline SDK Hexagon include directory}
here=$(cd "$(dirname "$0")" && pwd)
candidate=$(cd "$here/../../sdk/hexagon" && pwd)
tools=${HEXAGON_TOOLS_ROOT:-/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07}/Tools/bin
build_dir=${PTQ1_BASELINE_BITS_BUILD_DIR:-$(mktemp -d /tmp/ptq1-baseline-bits.XXXXXX)}
mkdir -p "$build_dir"
test ! -e "$build_dir/base-off.log"
for mode in off on production; do
    flags=()
    if [[ $mode != off ]]; then flags+=(-fvectorize); fi
    if [[ $mode == production ]]; then flags+=(-flto); fi
    for arm in base candidate; do
        include=$baseline
        test "$arm" != candidate || include=$candidate
        elf="$build_dir/$arm-$mode.elf"
        log="$build_dir/$arm-$mode.log"
        "$tools/hexagon-clang" -I"$include" -mcpu=v75 -mv75 -mhvx=v75 -mhmx \
            -O2 -Wall -Wextra -Werror "${flags[@]}" -DGENIEX_PTQ1_QUAD_ORACLE \
            "$here/ptq1_baseline_bits_sim.c" -o "$elf"
        "$tools/hexagon-sim" --march v75na_1 -r "$elf" > "$log" 2>&1
        test "$(grep -c '^PTQ1_BASELINE_BITS_RAW K=' "$log")" = 27
        test "$(grep -c '^PTQ1_QUAD_ORACLE_BITS_RAW K=' "$log")" = 54
        grep -Fxq PTQ1_BASELINE_BITS_CASES=27 "$log"
        grep -Fxq PTQ1_QUAD_ORACLE_BITS_CASES=54 "$log"
        grep '^PTQ1_BASELINE_BITS' "$log" > "$log.bits"
        grep '^PTQ1_QUAD_ORACLE_BITS_RAW' "$log" > "$log.oracle"
    done
    elf="$build_dir/candidate-$mode-quad.elf"
    log="$build_dir/candidate-$mode-quad.log"
    "$tools/hexagon-clang" -I"$candidate" -mcpu=v75 -mv75 -mhvx=v75 -mhmx \
        -O2 -Wall -Wextra -Werror "${flags[@]}" -DGENIEX_PTQ1_QUAD_TEST \
        "$here/ptq1_baseline_bits_sim.c" -o "$elf"
    "$tools/hexagon-sim" --march v75na_1 -r "$elf" > "$log" 2>&1
    test "$(grep -c '^PTQ1_BASELINE_BITS_RAW K=' "$log")" = 27
    test "$(grep -c '^PTQ1_QUAD_ORACLE_BITS_RAW K=' "$log")" = 54
    grep -Fxq PTQ1_BASELINE_BITS_CASES=27 "$log"
    grep -Fxq PTQ1_QUAD_ORACLE_BITS_CASES=54 "$log"
    grep -Fxq PTQ1_QUAD_BITS_CASES=54 "$log"
    grep '^PTQ1_BASELINE_BITS' "$log" > "$log.bits"
    grep '^PTQ1_QUAD_ORACLE_BITS_RAW' "$log" > "$log.oracle"
    cmp "$build_dir/base-$mode.log.bits" "$build_dir/candidate-$mode.log.bits"
    cmp "$build_dir/base-$mode.log.bits" "$log.bits"
    cmp "$build_dir/base-$mode.log.oracle" "$build_dir/candidate-$mode.log.oracle"
    cmp "$build_dir/base-$mode.log.oracle" "$log.oracle"
done
printf 'PTQ1_BASELINE_AND_QUAD_OUTPUT_WORD_AND_PADDING_COMPARISONS_PASS\n'
printf 'Receipts: %s\n' "$build_dir"
