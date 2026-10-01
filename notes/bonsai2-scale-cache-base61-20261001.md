# B17 bounded activation-scale cache

Latest: all three corrected board pairs completed22:51; see the final
section for exact measurements and restoration. Median gains are
+0.548341786% prefill/+5.352255864% decode. Earlier entries below preserve
the failed, null and intermediate attempts; their live handles are historical.

Board pair1 completed22:32: baseline prefill0.285094908/decode0.240855021;
candidate prefill0.286658202/decode0.253746198 tokens/s. Exact raw times
prefill105228116→104654253us, decode33215002→31527566us give +0.548341786%
prefill/+5.352255864% decode. Each arm's warmup and measured output matches
expected count text. Main read terminal JSON and output markers; copied JSON
to /tmp/geniex-scale-cache-scalar-conversion-board-receipts-20261001.
Only one pair, not a retained gain or final median. Same92487 lock window
continues pair2 candidate413149; pairs2/3 and restoration pending. Accepted
board totals remain unchanged. No commit or PR yet.

Corrected board qualification, 2026-10-01 22:24 UTC: controller92487 holds
the shared lock from22:22:08 after TensorFold's explicit release. Both52-case
PTQ operator gates pass; main compared all52 output-hash lines exactly.
Full GDN output/state and one/four-worker snapshot bytes pass the audited
controller's comparisons. The previous six eligible decode mismatches are
resolved by scalar-conversion preservation. First serving benchmark412613
is live; three-pair measurements and runtime restoration remain pending.
No accepted speed gain, commit or PR yet. Window
/home/arduino/geniex-bonsai/b17-scale-cache-scalar-conversion-r2-20261001.

Fast repetition harness completed, 2026-10-01 22:20 UTC: session97364
ended0. Command: ssh plexserver 'bash /tmp/geniex-scale-cache-worker-fast-20261001/run.sh'.
Same frozen source/compiler/model shape as full-width worker benchmark:
K5120/M1/N5120/T4/bias1/seed7. Main read full runner and fixture delta;
source SHA pins, safety checks and full-output hash comparison retained.
One unchanged full scalar-oracle control completed in124s. All six repetitions
match control934b1e1d6f99f12e; deliberately changed output was rejected.
Repetition wall seconds in execution order:30,31,30,31,31,31.
Three pairs each7586196/7355430 worker Pcycles: +3.137355668% throughput.
Both counters are9 cycles above the old harness; codegen context changed,
so these are fresh baseline/candidate measurements, not identical binaries.

Total harness wall time339s (5m39s), versus original1036s (17m16s).
Original duration derives from Plex results-directory birth1790891451 to
controller final mtime1790892487; new duration uses Bash SECONDS around
the complete Docker harness. Observed test-process speedup3.056047198x,
67.277992278% less time. Runs were not paired under fixed host load; old
control146s versus new124s shows load/timing variability. Whole-repetition
cycles618351690/617890182 become140656824/140195316 after removal of the
repeated scalar reference. This is a test-process optimization, not an
application/NPU gain. No product change or new board improvement.
Raw local/Plex /tmp/geniex-scale-cache-worker-fast-20261001/results.
Runner44e5e9d/fixture0c4e93ea; complete source and exact commands retained.

Full-width B17 worker proxy completed, 2026-10-01 22:08 UTC: session23178
ended0. Command: ssh plexserver 'bash /tmp/geniex-scale-cache-worker-fullwidth-20261001/run.sh'.
K5120/M1/N5120/T4/bias1/seed7, SDK6.6/Tools19 O2/vectorization/LTO.
Three alternating pairs each report7586187 baseline/7355421 corrected worker
Pcycles: -3.041923433% cycles, +3.137359507% worker throughput. All six full
outputs match unchanged scalar-oracle control hash934b1e1d6f99f12e and checksum
-46839.7. Guards, padding and input/descriptor checks pass. Main copied and
independently compared every output. Raw local/Plex
/tmp/geniex-scale-cache-worker-fullwidth-20261001/results.
Whole-process counts618351690/617890182 include expensive scalar reference;
do not substitute these for worker counters. Sequential fixture workers,
not board serving. Corrected board gates still await TensorFold's release.
Accepted total remains +39.993116851% prefill/+37.594789140% decode.

Latest status, 2026-10-01 21:15 UTC: full build66523 and wider worker18534
are terminal0. Nine paired cases and54 baseline-bit comparisons pass, with
all values retained, including slower fallback fixtures. Corrected parent
frame3328, other ten MATVEC frames112, dot child2176, worker budget16384;
not a complete external call-chain bound. The corrected board controller is
prepared locally but not deployed. TensorFold keeps the next board turn.
No corrected board qualification, serving gain, commit or PR yet.

Repository runner end-to-end check21:23: default O2 session22120 ended0/all
three hashes47ce363af3a6732a; did not reproduce the old conversion failure.
Seeded regression mode now adds production vectorization/LTO; ordinary default
cases are unchanged. R2 session74505 ended0: base/fixed5fab53f3a0f55853 and
failed4e55d8ee3ae8954e. The documented command asserts baseline hash via grep.
Runner0838d713, source24aa0d6c. Raw local
/tmp/geniex-scale-cache-repo-runner{,-r2}-20261001/results. No board claim.

Seeded nine-case extension session35771 ended0, main independent paired
output/marker audit passes. Same original and corrected production sources,
fixture24aa0d6c, SDK6.6/Tools19 O2/vectorization/LTO. Raw local
/tmp/geniex-scale-cache-seeded-wide-20261001/results; exact controller run.sh
in its parent on both hosts. Whole-fixture Pcycles include setup and scalar
oracle, so these values are not serving gains or worker-only measurements.

| Case | Baseline whole-fixture Pcycles | Candidate whole-fixture Pcycles |
| --- | ---: | ---: |
| small K128/N513/T1 | 2510271 | 2509821 |
| threshold K5120/N385 | 51910494 | 51906807 |
| odd N513/no bias | 68262771 | 68262315 |
| max K17408 | 229932762 | 229930842 |
| large K17536 fallback | 231615747 | 231684054 |
| small/empty partition fallback | 20744880 | 20750670 |
| M3 fallback | 45064056 | 45081231 |
| padded weights | 68270874 | 68270394 |
| VTCM rejection | 7965417 | 7965537 |

Worker-only timing trial40442 now uses existing SDK HAP_perf_get_pcycles,
one warmup then a timed public op_matmul call with compiler barriers. Three
alternating pairs enforce original exact output hash5fab53f3a0f55853. Source
and controller /tmp/geniex-scale-cache-worker-cycles-20261001 on both hosts.
No result yet; sequential fixture workers remain a simulator limitation.

Counter follow-up21:46: attempt40442 ended1; the hardware counter returns0
in this simulator setup and the positive-count guard rejects it. Baseline
correctness hash remains5fab53f3a0f55853. R2 session95397 ended1 at link:
hexagon_sim_read_pcycles needs existing SDK libhexagon.a. Both failed results
are retained in their /tmp/geniex-scale-cache-worker-cycles{,-r2}-20261001
roots on both hosts. Main verified the installed header and library symbol.
R3 session43123 now uses that simulator API/library; three pairs pending,
same fixture/warmup/output gates. No speed result from either failed attempt.

R3 session43123 ended0 at21:47. Three alternating baseline/corrected pairs
each report642468/638661 worker-only Pcycles. This is -0.592558696% cycles
or +0.596090884% worker throughput for K5120/M1/N385/T4/bias1/seed7.
Main independently compared all paired outputs and the original known hash;
all six match5fab53f3a0f55853. Raw local and Plex
/tmp/geniex-scale-cache-worker-cycles-r3-20261001/results. Count excludes
the scalar oracle and includes production op_matmul/quant/DMA with sequential
fixture workers. It is not board serving performance and does not change
accepted board totals. All runs retained, including the two failed counter
attempts. Next full-width proxy needs a plain-fixture baseline output anchor.

Full-width trial23178 started21:50: K5120/M1/N5120/T4/bias1/seed7,
same corrected production source, oneCPU/2GiB/nice10/shares128. Unchanged
plain fixture24aa0d6c first establishes output hash; timed baseline/corrected
builds each warm up and must match it through three alternating pairs.
Controller/source /tmp/geniex-scale-cache-worker-fullwidth-20261001 on both
hosts. Exact job remains live; do not restart. No full-width gain yet.

Base eac46bf0ccd9a7ed5870e349449009a97776a960, PR61, open and unmerged.
Target Bonsai2 PTQ1_0 on VENTUNO Q HTP v75.
Candidate: prepare FP32 activation scales once per decode worker rather than
once per pair of output tiles. Use the same FP16-to-FP32 conversion and
ascending K/scale order. Keep the existing uncached NPU path for small
partitions, K above17408, odd tiles, prefill and other formats.

For N5120 with four workers, K5120 conversion count is12,800 to640.
For K17408 it is43,520 to2,176. These are source counts, not speed gains.
The table requires640 or2,176 bytes per worker. Inspect actual compiler stack
frames before use. VTCM, weight DMA, dispatch and quantization are unchanged.
The earlier profile placed PTQ1 matmul at96.4% of summed operator time.
That is not a current decode wall-time fraction or a forecast for this change.

## Unchanged production-worker baseline

Command:
`ssh plexserver 'bash /tmp/geniex-scale-cache-base61-worker-base-20261001.sh'`.
Session54028 completed with exit0. K5120/M1/N5120, four sequential stub
workers, bias1, real DMA, O2/vectorization/LTO, SDK6.6/tools19.0.07.
The existing op_matmul fixture checks scalar output, padding, descriptors,
input immutability and VTCM guards. It reports output hash9b9ee3cf6f546438.
Pcycles610711236 include fixture setup and scalar reference. They are not
a kernel-only speed measurement or board latency.

Receipts on Plex and locally:
`/tmp/geniex-scale-cache-base61-worker-base-results-20261001`.
Input hashes, compilation output, objects, ELF and complete raw.log are kept.
Do not restart into this directory. Candidate outputs must match this hash
before wider gates. The fixture's workers are sequential, not true concurrency.

## Required next checks

Inspect the focused source port and every generated-format stack frame.
Apply to a separate owned Prism source. Check first and repeated SDK configure.
Use existing production-worker gates for eligible partitions, partial tiles,
K5120/17408, K17536 fallback, bias and no bias, padding and VTCM rejection.
Compare complete outputs to accepted baseline. Run real board gates and
three paired serving runs after the current scalar4 window is complete.
Do not combine the candidates or claim a gain before that result.

## Initial candidate worker result

Header SHA4266e0b0329dd4b9b75881cdee608f20f688c67499f9ae3fd79a152d1615a8cf.
Patch SHA476bae9121f287e8e33719edc99493fc16335924953d088face11c6b538ea1ec.
Applied matmul source SHA334f6286f92fe459753636a9d67c0a910169005f3aacd59aa7b95d9b381888fc.
Owned-source patch replacement and both SDK configure passes succeeded.

Command:
`ssh plexserver 'bash /tmp/geniex-scale-cache-base61-worker-candidate-20261001.sh'`.
Session66731 completed with exit0. Same K5120/M1/N5120/four sequential
workers/bias1/real DMA/O2/vectorization/LTO fixture and compiler flags.
Candidate hash9b9ee3cf6f546438 exactly matches baseline. Whole-fixture
Pcycles610711236 to610456116; this includes setup and scalar reference.
A single run per arm is not a retained performance result or board gain.
Both complete receipt directories exist locally and on Plex.

Full cross-build27446 completed with exit0 at19:50 UTC. ARM64 host/plugin
and v73/v75/v79/v81 passed. Host backend is byte-identical to baseline:
6f1c13191c5e272fca17a0b2711e7ff4786aa7d68965a9890bf5ce1fbfaf391a.
Candidate v75 SHA:
111d14e9e0c7e65f92fbcdc5daaa7b3d35b1cd105a32ff63bd0ff7ba160e14bc.
Compiled v75 frame audit: PTQ1 decode1152 to3584 bytes; all ten other
MATVEC expansions remain112 bytes. The dot_tile child remains2176 bytes.
The runtime work queue allocates16384 bytes per worker. Main independently
checked all eleven prologues and runtime source. Alignment and external
calls mean these frames are not a complete worst-case call-chain bound.
Full disassemblies are preserved locally and on Plex at
`/tmp/geniex-scale-cache-stack-audit-20261001`.

Expanded paired-worker gate75408 completed with exit0 at19:58 UTC: nine
existing-fixture shape/configuration cases and54 baseline-bit comparisons.
Independent local result/hash/count checks passed. The controller is
`/tmp/geniex-scale-cache-base61-paired-worker-gates-20261001.sh`.
Preserve `/tmp/geniex-scale-cache-base61-paired-worker-results-20261001`
on both hosts. All paired fixture measurements follow; these include
initialization and the scalar oracle, so they are not kernel-only timings.

| Existing worker case | Base Pcycles | Candidate Pcycles |
|---|---:|---:|
|K128/M1/N513/one worker/no bias|2508867|2508417|
|K5120/M1/N385/four workers/bias|51866235|51850587|
|K5120/M1/N513/four workers/no bias|68219064|68206536|
|K17408/M1/N513/four workers/bias|229786695|229736604|
|K17536/M1/N513/K-limit fallback|231469395|231534807|
|K5120/M1/N145/empty-worker fallback|20701137|20706660|
|K5120/M3/N129/multirow control|44934768|44951151|
|K5120/M1/N513/padded weights|68227344|68214792|
|K5120/M1/N513/VTCM rejection|7922730|7922868|

### Board coverage gap and required gate

The existing45-case board gate has small output widths for M1; its three
full-width cases use M30 and the unchanged true-batch route. Thus it does
not qualify the eligible decode cache at full model width.
Reuse the same public MUL_MAT fixture and CPU comparison with seven extra
M1 cases: (N,M,K)=(385,1,5120),(513,1,5120),(513,1,17408),
(513,1,17536),(5120,1,5120),(5120,1,17408),(17408,1,5120).
Total52 selected cases. N385 partitions4,4,4,1 tiles; N513 partitions5,5,5,2.
Those exercise both cached and small-partition workers. K17536 is fallback.

Private test patch notes/ptq1-scale-cache-board-cases.patch SHA
b0f342e80bf8fc991394889ff297217c611e20cf05b6aaf0d467246295c459c8.
Apply/reverse checks pass on owned test source, SHA
cf2d7d76fff1747fb593256988a9e8e3691e60fc120efb1ce0b5e65ef53f77e8.
The board-native executable build14133 is live, command
`ssh plexserver 'bash /tmp/geniex-scale-cache-base61-board-test-build-20261001.sh'`.
No new operator/private test boundary or product code change was added.
Local board controller requires a separately verified52-case executable,
its third SHA argument, and52 full-output hashes per arm. It is prepared
but not deployed or run. Preserve TensorFold's next turn.

No board measurement, commit, PR or accepted gain exists for this candidate.

### 2026-10-01 20:55 UTC — B17 scalar conversion correction and regression

One pragma disables vectorization only of prepare_flat_scales; caching,
worker layout and all other math remain unchanged. Header SHA4a17fc39.
First launcher39446 failed before candidate compilation: an escaped newline
joined objdump and simulator commands, producing an unknown --march argument
for objdump. Failed results and run-attempt1.sh remain in
/tmp/geniex-scale-cache-scalar-conversion-main-20261001. This is a controller
failure, not a failed kernel fix. Corrected runner SHA bcedf191 uses a new
results-r2 directory; no failed receipt was overwritten.

Session32965 ended0: captured-input public op_matmul output baseline-exact,
both hash8ec2448939ceb374. Candidate whole-fixture51872586Pcycles versus
baseline51876273; not a speed gain claim. Main disassembly audit confirms
four scalar __extendhfsf2 calls in cache loop0x10ac4–0x10b24 and no half-vector
multiply in PTQ1 MATVEC. An initial overly broad whole-ELF absence check failed
because unrelated existing MATVEC kernels still contain half instructions;
the corrected changed-function audit passes. Keep that failed audit visible.
Raw local /tmp/geniex-scale-cache-scalar-conversion-r2-results-20261001.

Compact deterministic regression reuses the existing op_matmul fixture:
LCG seed7, upper24 bits mapped to [-1,1), PTQ1_WORKER_RANDOM_ACT=1.
Session71646 ended0, O2/vectorize/LTO, K5120/M1/N385/T4/bias1:
baseline5fab53f3a0f55853; failed cache4e55d8ee3ae8954e; corrected cache
5fab53f3a0f55853. All guards/scalar tolerance pass. This detects the old
failure without storing5120 literals in the repo. Command:
ssh plexserver 'bash /tmp/geniex-scale-cache-random-regression-main-20261001/run.sh'.
Controllera5215904; source24aa0d6c. Raw local
/tmp/geniex-scale-cache-random-regression-results-20261001.
Source option promoted to existing repository fixture and runner; default0
preserves old input generation. Repository runner SHA5546b684, syntax passes.
Luna read-only audit found no concrete bounds/caller/default-input defect.

Full build66523 ended0: two actual SDK configure passes, ARM64/plugin and
v73/v75/v79/v81. Frozen corrected source/Prism/build prefixes
/tmp/geniex-scale-cache-scalar-conversion-{source,prism,build}-20261001.
Header4a17fc39, patch476bae91 and applied matmul334f6286.
Host unchanged6f1c1319; v755a742fdc29315074c8f83f5dcc2006e55b122f3f52aa59ffd897195ed51e0810.
Local verified /tmp/geniex-scale-cache-scalar-conversion-artifacts-20261001.
Native52-case executable remains25b19a40. Wider paired-worker controller18534
is still live at last check20:55, first five pairs pass. Do not restart it.
No corrected board test, serving measurement, commit, PR or retained gain yet.


### 2026-10-01 20:38 UTC — B17 conversion root cause reproduced

Private diagnostic retains the cache vector preparation and compares it to
one-half scalar conversion through the existing op_matmul fixture.
All three eligible workers fail at block0/group0: half1fe7 expands exactly
to FP32 3bfce000, but cache holds3bfce001 (one ULP high). This is a normal
FP16 value, not a subnormal. The public output retains the original failed
candidate hash e3b30d4b6e368321. Concurrent workers are not required.

Independent main log/disassembly check passes: scalar diagnostic at0x23d38
calls __extendhfsf2 at0x2f1d4; cache at0x10be8 performs half multiplication
into qf32, followed by qf32-to-sf conversion at0x10bf0. This identifies a
conversion discrepancy in the local failing path, not a board speed result.

Private applied C SHA525296f1; diagnostic patchd0b24821; runner3ec5f879;
ELFef29e26bbeebb45dcc94b63d079b4da9da58e7b46bcb383fb88ebe0ebece5274.
Command: ssh plexserver 'bash /tmp/geniex-scale-cache-conversion-bits-diag-run-20261001.sh'.
Raw local /tmp/geniex-scale-cache-conversion-bits-diag-results-main-20261001,
Plex /tmp/geniex-scale-cache-conversion-bits-diag-results-20261001.
The earlier patch738a1d41 lacked completion markers; its main-review copy
is preserved. Final patchd0b24821 logs checked count for each cached worker.
Next isolated fix trial disables vectorization only for bulk scale conversion,
retaining caching and all other math. It must make the captured-input public
operator output baseline-exact before wider gates/build or board timing.
Frozen failed source and receipts remain unchanged; no production fix yet.


### 2026-10-01 20:32 UTC — B17 simulator reproduction with captured inputs

Observation-only public MUL_MAT fixture built in session64871, terminal0.
Source /tmp/geniex-scale-cache-board52-observe-20261001.cpp SHA3bb6cdf9;
private patch notes/ptq1-scale-cache-board-observe.patch SHA05ddb3a4.
Executable /tmp/geniex-scale-cache-board-observe-test-backend-ops-20261001
SHAc67e2f75ac6d0081c54e18b05f24616cc31eb884db36dcb94516d0dae898a8ad.
Builder /tmp/geniex-scale-cache-board-observe-build-20261001.sh SHA43fbe61c.
Local ARM CPU control selected one N385/M1/K5120 case and passed. It emits
385 output/reference bit pairs,5120 activation words and two input hashes.
This validates observation output, not NPU correctness. First launch exited127
because copied libraries lacked the .so.0 CPU name; the missing dependencies
were copied and the new cpu-control-complete.log run exited0. Both logs remain
in /tmp/geniex-scale-cache-board-observe-native-20261001.

Captured CPU activation words were inserted into the existing production
op_matmul simulator fixture, with no product-source change. Command:
ssh plexserver 'bash /tmp/geniex-scale-cache-cpu-act-diag-20261001/run.sh'.
Session53874 ended1 because exact-output comparison FAILED:
base8ec2448939ceb374 versus candidatee3b30d4b6e368321. Both guards and scalar
reference tolerance passed, checksum267.0. Base whole-fixture51876273Pcycles,
candidate51860634; not a speed claim. This reproduces an exact-output difference
with sequential stub workers, so concurrent workers are not necessary to trigger
this local failure. It is not the same activation set as the board, whose
initializer uses eight threads; the local CPU initializer uses four threads.

Frozen production base/candidate are unchanged. Diagnostic source SHA8521d89e,
activation header015a3cf2, runner2ea19b27. Raw local receipts:
/tmp/geniex-scale-cache-cpu-act-diag-results-20261001.
Source and receipts also on Plex under /tmp/geniex-scale-cache-cpu-act-diag-20261001.
Sol is preparing cache-vs-scalar bit logging using this reproduced failure.
Astra found matching paired accumulation instructions and K reduction order.
No production fix was attempted. Board is reserved for TensorFold's next turn.
Local one-case board controller is prepared but not deployed/run:
/tmp/geniex-scale-cache-board-diagnostic-r2-window-20261001.sh.
It requires a released reservation explicitly naming GenieX as next owner.


### 2026-10-01 20:21 UTC — B17 board failure and local null diagnostic

The 52-case native executable build14133 completed0, SHA
25b19a4010a34b02199c1d181abde84b8d1b8f78ab1806054a6b07ad4b7ee054.
Command: ssh plexserver 'bash /tmp/geniex-scale-cache-base61-board-test-build-20261001.sh'.
B17 board session47417 ended1, window20:05:08–20:06:29. Both runtime arms
passed52 CPU-reference comparisons. Exact full-output comparison failed
on six eligible decode cases. In (output N, activation M, input K) notation:

| Shape | Baseline hash | Candidate hash |
|---|---|---|
|17408,1,5120|809f26369c1d91f7|20b626303eb8ec13|
|385,1,5120|4b37143b8297e88d|d16e4f9f801d945a|
|5120,1,17408|7fb4060a31ceb045|8c190b176a53e4c4|
|5120,1,5120|044bde6330a9d93a|de1413da66b8984b|
|513,1,17408|5971dab9e18183c4|8ae9998a51b8b522|
|513,1,5120|46a3359ab264ab02|1e2e9af199d13f89|

The old45 cases and K17536 fallback match. No serving pairs ran. The
controller restored the original host and v75 runtime; independent hash
checks passed20:08. Reservation released, next TensorFold. B17 is not
retained, no PR or speed gain. Raw local receipts:
/tmp/geniex-scale-cache-base61-board-r1-receipts-20261001.
Controller /tmp/geniex-board-scale-cache-base61-window-20261001.sh SHA
934de9123879447c470fde0fe0b85431e4729fd85351093f04e41c39a1796377.
Do not restart this failed window or overwrite its receipts.

Compiler audit found vector FP16 conversion in bulk cache preparation,
where the original path uses scalar __extendhfsf2. This is a hypothesis,
not a proved root cause. Source audit found no ordinary missing barrier
or table-lifetime error. HVX store visibility remains unproved.

Local diagnostic reused the existing production op_matmul fixture, changing
only activation denominator64 to67. K5120/M1/N385/four sequential stub
workers/bias1/real DMA, same compiler flags and frozen runtime source.
Command: ssh plexserver 'bash /tmp/geniex-scale-cache-nonbinary-act-diag-20261001/run.sh'.
Session17459 ended0. Both output hashes d78c5192df682fd1; guards and scalar
reference passed. This did not reproduce the board failure and does not
prove board correctness. Source SHA
f136a2645437ad21f81d6ef8b84ff15f929450912b529f7f750644e2eef551df;
controller SHAa5c2c983b769fb2408003643fbbcfc8a9014297a89d95771bebfe2c38113acdc.
Raw local receipts /tmp/geniex-scale-cache-nonbinary-act-diag-results-20261001.
No production fix was attempted. Next: compare cached and scalar-converted
scale bits on one failing public MUL_MAT case when the board is released.
User reports TensorFold has started; leave the board to it. Accepted totals
remain +39.993116851% prefill/+37.594789140% decode across separate steps.

## Corrected board A/B — 2026-10-01 22:51 UTC

Base PR61 eac46bf0ccd9a7ed5870e349449009a97776a960. Candidate files:
header4a17fc39, patch476bae91, fixture24aa0d6c, runner0838d713.
Disable vectorization only in cache preparation: scalar FP16 conversion
preserves the original bits. Captured-input and seeded regression reproduced
the old one-ULP failure and passed with this correction. Default-O2 diagnostic
did not reproduce it; preserve that null result.

Controller /tmp/geniex-board-scale-cache-scalar-conversion-r2-window-20261001.sh
(SHAab44f773) ran under the shared lock, root session92487 ended0.
Model Bonsai2 PTQ1_0 SHA53107f53; SDK6.6/Tools19.0.07/v75, four virtual
HTP sessions/four HVX workers, diagnostics off. Exact command per arm, from
runtime /home/arduino/geniex-bonsai (window is the receipt directory):

```bash
unset GGML_HEXAGON_HOSTBUF
export LD_LIBRARY_PATH="$runtime/src:$runtime/bin:$runtime/plugins/llama_cpp"
export GGML_HEXAGON_NDEV=4 GGML_HEXAGON_VERBOSE=0 GGML_HEXAGON_PROFILE=0
./geniex-bench --plugin llama_cpp --device npu --device-id HTP0,HTP1,HTP2,HTP3 \
  -m "$runtime/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
  --prompt-file "$runtime/count-templated.txt" -n 8 -c 512 -t 4 --warmup 1 -r 1 \
  --output-json "$window/pair-$pair-$arm.json"
```

Actual prompt30tokens; output8; seed42/temp0; reset and warmup each arm.
Alternating order base/candidate, candidate/base, base/candidate.

| Pair | Base prefill us | Candidate prefill us | Base decode us | Candidate decode us | Prefill gain | Decode gain |
|---|---:|---:|---:|---:|---:|---:|
|1|105228116|104654253|33215002|31527566|0.548341786%|5.352255864%|
|2|105146099|104661736|33082428|31619111|0.462788999%|4.627951115%|
|3|105257514|104587348|33508570|31225024|0.640771578%|7.313192137%|

Median rates: prefill0.285094907525→0.286658202032 tokens/s;
decode0.240855020873→0.253746197851 tokens/s. Ratio-of-medians gains
+0.548341786%/+5.352255864%; paired median gains are the same here.
Every candidate rate exceeds every baseline rate in this series. There is
no separate unchanged-code noise series; this is a short count workload,
not a general-chat throughput result. All twelve warmup/measurement outputs
are the expected count text; both PTQ52 gates and all52 hashes match.
Six full GDN outputs/state,56 boundary cases, and snapshot worker1/4 bytes
match. No model compute fallback change was introduced.

Original runtime hashes independently match65e2bd27/cee1a3e8 after cleanup.
Reservation released22:51:34, nextTensorFold. Complete receipts copied to
/tmp/geniex-scale-cache-scalar-conversion-board-receipts-20261001/
b17-scale-cache-scalar-conversion-r2-20261001. Retain this candidate for PR
review; no commit/PR at this entry. Previous accepted totals remain separate
until PR qualification. Failed first-board hashes above remain in the record.

Fresh pre-PR cross-build54849 ended0 at23:00: ARM64/plugin and all four
HTP targets rebuild. Whitespace and runner bash syntax pass; source hashes
match the board-qualified snapshot. Focused independent bounds/callers/test
audit found no concrete issue. Snyk17668 ended1 for one LOW finding in the
unchanged tests/qdc/run_qdc_pytest.py XML parser (Python≤3.10), no HIGH/MEDIUM.
Do not count that as a clean whole-repository Snyk result. Canonical fleet
Sonar and manual bot reviews apply to the PR head after creation.
