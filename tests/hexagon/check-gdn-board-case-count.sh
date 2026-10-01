#!/usr/bin/env bash
# Host-only support enumeration. This does not run GDN computation.
set -euo pipefail
binary=${1:?pass the host-native test binary}
runner=${2:?pass the repository board runner}
log=${3:?pass a new log path}
expected=${4:?pass the expected case count}
test ! -e "$log"
pattern=$(sed -n "s/^[[:space:]]*-p '\([^']*\)'.*/\1/p" "$runner")
test -n "$pattern"
"$binary" support -b CPU -o GATED_DELTA_NET -p "$pattern" -j 1 > "$log" 2>&1
grep -q '^Backend 1/1: CPU$' "$log"
actual=$(grep -c '^  GATED_DELTA_NET(' "$log")
printf 'GDN host case count: actual=%s expected=%s\n' "$actual" "$expected"
test "$actual" = "$expected"
