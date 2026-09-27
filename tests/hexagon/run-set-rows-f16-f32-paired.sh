#!/usr/bin/env bash
set -euo pipefail

base_prism=${1:?pass baseline Prism checkout}
candidate_prism=${2:?pass candidate Prism checkout}
mode=${3:-correctness}
here=$(cd "$(dirname "$0")" && pwd)
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT

for arm in base candidate; do
    prism=$base_prism
    [[ $arm == candidate ]] && prism=$candidate_prism
    bash "$here/run-set-rows-f16-f32-sim.sh" "$prism" "$mode" >"$build/$arm.log" 2>&1
    cat "$build/$arm.log"
    sed -n 's/^SET_ROWS output FNV64=//p' "$build/$arm.log" >"$build/$arm.hash"
done

if [[ ! -s $build/base.hash || ! -s $build/candidate.hash ]] || ! cmp -s "$build/base.hash" "$build/candidate.hash"; then
    echo "SET_ROWS baseline/candidate output hash mismatch" >&2
    exit 1
fi
echo "SET_ROWS baseline/candidate raw-output FNV64 matched: $(<"$build/base.hash")"
