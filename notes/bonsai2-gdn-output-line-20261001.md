# GDN output-line correctness prerequisite

Base: PR59, `4557e68b4fde3a5b2d5f8e1a51029b98ed108ecf`.
Target: VENTUNO Q HTP v75. No CPU/GPU model fallback added.
This is a correctness prerequisite for the scalar-four-row experiment,
not a serving-speed improvement.

Attention and returned state occupy one output tensor. When the state region
starts inside a 128-byte output line, HVX attention writes and DMA state writes
share that line. Captured S10/H3 failures affect its first two state floats;
S3/H3 failures affect its first 23. State beyond that line agrees with CPU.
Worker-count changes alone did not fix those two cases.

The patch reuses `hvx_copy_f32_uu` for state output when `state_out_base` is not
128-byte aligned. It keeps a completed zero-row DMA queue entry so ping-pong
accounting is unchanged. Aligned output retains DMA. Arithmetic, VTCM allocation,
model format and host backend are unchanged. Both decode and prefill workers
use the same condition; other K snapshots already use HVX copies.

## Verification

| Gate | Baseline | Candidate |
|---|---:|---:|
| Initial 6 board cases, 1/4 workers | 4/6 at both | 6/6 at both |
| Expanded 50 captured board cases, 1 worker | 24/50 | 50/50 |
| Expanded 50 captured board cases, 4 workers | 12/50 | 50/50 |
| Repository 48-case board gate | 24/48 at 1 worker | 48/48 at 1/4 workers |
| Paired v75 simulator, 12 shape/flag cases | Pass | Pass |

All 50 candidate full attention/state tensors match byte-for-byte across 1/4
workers. All 8 aligned-output cases match baseline at both worker counts.
Every board case has an actual GATED_DELTA_NET HTP profile record. The CPU
backend is the numerical reference only. The simulator also verifies guards,
full paired byte equality and real output/state/input mutation controls.
These results support the shared-line cause, not an isolated cache trace.

The board matrix covers S3/10/31/32/33, scalar/vector gates, T1/2/3, K1/2/3,
multiple sequences, repeated heads and uneven worker assignments. For K>T,
older snapshot slots remain caller-owned, as specified in the CPU source.
The 50-case diagnostic adds two duplicates to the repository's 48-case matrix.

Full ARM64/plugin and v73/v75/v79/v81 builds passed. Host SHA256 is unchanged:
`6f1c13191c5e272fca17a0b2711e7ff4786aa7d68965a9890bf5ce1fbfaf391a`.
Production and successful diagnostic v75 binaries are identical:
`9100d42ebaa8972528d34ff917034b7ed92eabbf1dd96b25a75d1c4ab432c079`.
There is no full-model serving A/B measurement or speed claim for this fix.

## Re-run and receipts

Apply `tests/hexagon/gdn-output-line-cases.patch` to a private Prism test source,
then build its `test-backend-ops` against the matching ARM64 libraries.
Inside the shared board controller, with its lock on inherited descriptor 9,
its reservation running, and the exact candidate runtime staged, run:

```bash
timeout --kill-after=2s 900 bash tests/hexagon/run-gdn-board-boundaries.sh /absolute/test-backend-ops /absolute/new-receipts
```

Read `/home/arduino/AGENTS.md` before board use. Hold the shared lock through
staging, tests and original-runtime restoration. The runner does not deploy
libraries or restore them. No weights were copied or clocks changed.

Board controller: `/tmp/geniex-b14-boundary50-window-20261001.sh`.
Board window: 14:22:28–14:23:48 UTC, 2026-10-01; original hashes verified at cleanup.
Local raw board receipts:
`/tmp/geniex-b14-board-receipts-20261001/b14-boundary50-20261001`.
Simulator controller: `/tmp/geniex-gdn-output-line-paired-sim-20261001.sh`.
Simulator receipts: `/tmp/geniex-gdn-output-line-local-receipts-20261001/geniex-gdn-output-line-sim-results-20261001`.
Build controller: `/tmp/geniex-gdn-output-line-production-build-20261001.sh`.
Earlier failures and abandoned, unbuilt `syncht` proposal remain recorded in
the repository's B14 note and handoff. No tolerance was relaxed.

CodeRabbit CLI review completed with three trivial findings. The profile check
now matches only actual HTP0 GDN records; note spacing is corrected. A helper
for two short state-copy blocks was not added. Independent security and test
reviews found no concrete defect; the test review clarified the parent timeout
in the rerun command. The checked-in runner reproduces 48 CPU-reference cases,
not the separate 50-case diagnostic capture and cross-worker byte comparison.
Snyk commands returned no HIGH findings, but their SARIF has no file inventory,
so these commands do not prove C-source analysis coverage.
Independent performance review confirmed zero-row queue accounting and bounded
copy size. It found no concrete defect. Base-aligned attention with odd per-head
state strides (for example S3/H32/T1) is not covered by the aligned-case matrix;
this is a coverage gap, not a reproduced failure. The fix targets the proved
attention/state shared-line failure, not every possible cache interaction.
