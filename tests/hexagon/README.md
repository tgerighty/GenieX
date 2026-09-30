# PTQ1 experiment runner

Run in the existing v75 SDK environment. Default commands are unchanged.

```bash
python3 tests/hexagon/test_ptq1_dma_bench_orch.py
bash tests/hexagon/run-ptq1-dma-bench.sh /path/to/prism --worker
```

To reuse compiled objects, set `PTQ1_BENCH_CACHE_DIR` to an absolute writable
directory. The SDK image already has `ccache`. Mount the cache directory into
the container to reuse it across runs. Compiler content, flags, source and
included headers determine reuse. Linking and simulator checks still run.
Test results are never cached.
The FFN counter fixture copies source into a fresh temporary directory. Its
worker object does not get cache hits. Use `PTQ1_FFN_COUNT_QUANT=0` for timing;
keep counters enabled for the separate quantization-count checks.

```bash
PTQ1_BENCH_CACHE_DIR=/cache PTQ1_WORKER_K=5120 \
  bash tests/hexagon/run-ptq1-dma-bench.sh /path/to/prism --worker
CCACHE_DIR=/cache ccache -s
```

Batch timing initializes the fixture once, runs one warmup, then measures one
and four real operator calls with the SDK simulator cycle counter. It retains
scalar, bias, activation, output-padding, hash and VTCM checks. FFN quantization
counts cover all six calls when enabled.

```bash
PTQ1_BENCH_BATCH=1 PTQ1_WORKER_N=256 PTQ1_FFN_FUSED=1 \
  PTQ1_FFN_COUNT_QUANT=0 \
  bash tests/hexagon/run-ptq1-dma-bench.sh /path/to/prism --ffn-worker
```

`PTQ1 batch repeat1_cycles=... repeat4_cycles=...` times only the repeated
operator loops, not initialization or the scalar oracle. This is a new
measurement method: establish fresh baseline/candidate pairs. Do not subtract
these numbers from historical whole-process Pcycles. Keep the default runner
available for calibration. These measurements are not board tokens/s.

Split the ordinary correctness matrix with zero-based shard indices:

```bash
PTQ1_BENCH_SHARD_INDEX=0 PTQ1_BENCH_SHARD_COUNT=4 \
  bash tests/hexagon/run-ptq1-dma-bench.sh /path/to/prism --worker
```

Run indices 0 through 3 and retain every log. Their union is the original
18 shapes. Shards are not allowed for batch timing or explicit single shapes.
Keep paired timing serial. On Plex, retain the four-worker limit, one CPU per
container, low CPU shares, no GPU, and priority for Plex service load.
