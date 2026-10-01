#!/usr/bin/env bash
# Run inside the shared board controller after staging the exact runtime.
# First apply gdn-output-line-cases.patch to a private Prism test source and
# build test-backend-ops. The existing CPU comparison covers attention+state.
set -euo pipefail
binary=${1:?pass the board-native test-backend-ops binary}
receipts=${2:?pass a new absolute receipt directory}
test -x "$binary"
[[ $binary = /* && $receipts = /* ]]
test "$(readlink /proc/$$/fd/9)" = /home/arduino/board.lock
flock -n 9
grep -Fxq 'Owner: GenieX/Bonsai root agent' /home/arduino/board-reservation.md
grep -Fxq 'Status: running' /home/arduino/board-reservation.md
test ! -e "$receipts"
mkdir -p "$receipts"
export GGML_HEXAGON_NDEV=1 GGML_HEXAGON_PROFILE=1 GGML_HEXAGON_VERBOSE=1
export GGML_HEXAGON_HOSTBUF=0
ulimit -c 0
for workers in 1 4; do
    export GGML_HEXAGON_NHVX=$workers
    log=$receipts/workers$workers.log
    result=0
    "$binary" test -b HTP0 -o GATED_DELTA_NET \
        -p 'head_count=(3|5),head_size=(3|10|31|32|33),n_seq_tokens=(1|2|3),n_seqs=(1|2),v_repeat=(1|2),permuted=0,kda=(0|1),K=(1|2|3),rows_mode=0,cache_rows=-1,raw_gates=0' \
        -j 1 > "$log" 2>&1 || result=$?
    printf '%s\n' "$result" > "$receipts/workers$workers.exit-status"
    test "$result" = 0
    grep -q '48/48 tests passed' "$log"
    grep -q "HTP0 hwinfo: threads $workers, hvx $workers," "$log"
    test "$(grep -c '^ggml-hex: HTP0 profile-op GATED_DELTA_NET|' "$log")" = 48
done
printf 'GDN board boundary CPU-reference and HTP placement gates passed\n'
