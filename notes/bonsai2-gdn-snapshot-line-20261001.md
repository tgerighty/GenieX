# GDN snapshot-line correctness prerequisite

Base: PR60, `6c034c259944c6b9231ede390919e25e7d0ee140`.
Branch: `fix/bonsai-gdn-snapshot-line-base60-20261001`.
Target: VENTUNO Q Hexagon HTP v75. This is not a speed improvement.

## Reproduced failure

Private six-case controller 85667 completed with exit 0 at 15:24:07 UTC.
PR59 and PR60 each pass 4/6 cases at one and four workers. Numerical exits
remain 1. Both scalar and vector S3/H8/T4/K2 cases fail. S3/H32/T1/K1 and
S3/H8/T4/K1 pass. Each arm has six actual HTP profiles. All six captured
output tensors are byte-identical across the four arms. This predates PR60.

Raw receipts:
`/tmp/geniex-gdn-output-line-serving-local-receipts-20261001/gdn-snapshot-line-20261001`.
Native binary SHA256: `9cccc11a9ce66f149692c1aba9f6eb6165d54815413a9a3ba9ad948806b8b2b6`.
Original runtime hashes independently pass after cleanup. TensorFold got the
next turn and released it at 16:44:29 after its separate diagnostic.

## Cause and single-variable hypothesis

The failing shape has these byte regions in the contiguous output:

| Region | Inclusive byte offsets | Writer |
|---|---:|---|
| Attention | 0–383 | HVX |
| Most recent state, slot 0 | 384–671 | DMA |
| Prior state, slot 1 | 672–959 | HVX |

The last two regions share cache line 640–767. K1 and K2 captures match
through byte 639. Only 30 bytes differ at offsets 640–670 within the shared
slot-0 region. This pattern is identical in all four arms and both gate modes.
One worker also fails, so this does not require a race between workers.

Extend the prefill `hvx_state_copy` condition to K>1 when a whole snapshot
stride is not 128-byte aligned. Reuse the existing HVX copy and zero-row DMA
queue entry. Keep snapshot order, numerical operations and VTCM unchanged.
Decode has no prior-snapshot write. Its guard stays unchanged. Aligned
snapshot strides and the S128 production model keep their existing path.

Production patch SHA256:
`d12d2534e48716446a2ae6d2d26e7982a7f22826c78b8a3850a5396c8661e04a`.
The production patch passes `git apply --check` on the actual PR60 source.

## Gates and commands

Build controller: `/tmp/geniex-gdn-snapshot-line-production-build-20261001.sh`.
On Plex: `bash /tmp/geniex-gdn-snapshot-line-production-build-20261001.sh`.
Session 78549 completed with exit 0. Full ARM64/plugin/all HTP build passes.
Host SHA256 remains `6f1c13191c5e272fca17a0b2711e7ff4786aa7d68965a9890bf5ce1fbfaf391a`.
v75 SHA256: `ed83c95395e1d1a7f4e3e960a2c20474ef6102ca13bc8eacb429122f2b5f37b6`.
The existing SDK6.6 / Tools19.0.07 toolchain and one-CPU/4GiB limit remain.

Paired simulator controller: `/tmp/geniex-gdn-snapshot-line-paired-sim-20261001.sh`.
On Plex: `bash /tmp/geniex-gdn-snapshot-line-paired-sim-20261001.sh`.
Session 31204 completed with exit 0. Twelve K1 shape/compiler-flag cases
pass full output/state equality, CPU references, guards and fault controls.
The test took 118.201408398 seconds. This is not a speed result and does not
exercise the new K2 condition. K2 qualification requires the board test.
Copied receipts: `/tmp/geniex-gdn-snapshot-line-local-receipts-20261001`.

The repository test patch now adds six cases to the previous 48. The native
suite binary compiled with O2/Wall/Wextra/Werror and the existing unused
helper exemption. Maximum required GLIBC is 2.38.
Binary SHA256: `4095ffe5c6cff2aa29730c2ae065fb1f8f9199e322abef6b771911106fdfd9f1`.
Runner SHA256: `4fc49fd1bd64ee20a49693fdb79933a8328d2193af27ace9703f888abc7f8583`.
The test patch reverse-checks against the compiled source. Shell syntax passes.

Board controller: `/tmp/geniex-gdn-snapshot-line-fix-r2-window-20261001.sh`.
Arguments: candidate v75 SHA256, suite binary SHA256, runner SHA256.
It requires the shared lock and GenieX turn. It compares baseline/candidate
six-case numerical results and captures at one/four workers, then runs the
repository suite. Expected baseline: 4/6 private and 54/56 repository.
Required candidate: 6/6 private and 56/56 repository at both worker counts.
Every repository result requires actual HTP profiles. The controller has a
900-second timeout per test command and restores original runtime hashes.
The revised board controller is not yet run. Candidate fix is not yet qualified.

## Failed launcher and count assertion — 17:13 UTC

First launcher 44388 exited 127 before the controller started. The SSH command
used the wrong basename. No board ownership or runtime change occurred.
Corrected launcher 72390 ran 17:12:32–17:13:15, then exited 1. Both baseline
private runs reproduced 4/6. The repository baseline reported 54/56. The
expanded filter also selects two existing aligned S32/H8/T4/two-sequence
cases, and both pass. The incorrect 54-case count stopped the controller
before candidate deployment. Keep these useful existing controls and require
56 cases. This changes the gate count, not its tolerance or production code.
Original host/DSP hashes pass cleanup. Next turn is TensorFold. Preserve the
failed window at `/home/arduino/geniex-bonsai/gdn-snapshot-line-fix-20261001`
and its local copy under `/tmp/geniex-gdn-snapshot-line-local-receipts-20261001`.
The renamed `regression56-test-backend-ops` binary has identical bytes and
SHA256 to the earlier `regression54-test-backend-ops` artifact.

## Reporting

Preserve all failures and complete captures. Do not relax the 1e-7 reference
limit. Do not add this correctness result to performance totals. Current
accepted compounded gain stays +39.993116851% prefill/+37.594789140% decode.
No PR exists for this candidate yet. No merge is permitted.

## Local reviews — 17:23 UTC

CodeRabbit reservation `rsv_633e2bcc99184895` was successfully claimed for CLI
at 17:16:51.997769Z. Review 1096 completed with exit 0 and zero findings on
three tracked files. It did not review the new untracked production patch or
this note. The coordinator recorded success at 17:19:54.292776Z. No retry ran.
Independent Sol6 performance/logic and Luna6 security reviews explicitly
included the production patch. Neither found a concrete defect. The Luna6
test review found the filter-count error and a stale binary name in the r2
argument prompt. Both are corrected. Main verified the count in the real
baseline log. The revised runner and controller pass shell syntax checks.

Snyk repo scan94454 and applied-source scan20866 completed with exit 0 and
zero high-severity findings. Both SARIF reports lack artifact inventories.
Do not claim native C coverage from these results. The actual applied C source
SHA256 is `f5411a85b1febc773b00a88b7e4e907b144d5a0a04f9edf9260a047aae9a2c3c`.
The production patch reverse-checks against that built source. The canonical
SonarQube and CodeRabbit guides were read completely before these operations.
No Sonar result exists for this uncommitted candidate. Wait for actual board
red/green qualification before commit and fork PR creation.

## Host case-count preflight — 17:28 UTC

Reuse the native test binary's `support` mode to enumerate the exact board
filter before reserving hardware. The helper reads the filter from the board
runner instead of maintaining a second filter. It requires a CPU-only backend
inventory and does not run GDN computation. Invoke it with the host-native
binary, repository runner, new log path and expected count 56:

```sh
bash tests/hexagon/check-gdn-board-case-count.sh \
    /tmp/geniex-b14-board-test-build-20261001/regression56-test-backend-ops \
    tests/hexagon/run-gdn-board-boundaries.sh /tmp/new-case-count.log 56
```

Set `LD_LIBRARY_PATH` and `GGML_BACKEND_PATH` to the matching local GGML and
CPU libraries. This ARM host used `/tmp/tensorfold-prism-gpu-20261001/build/bin`
and its `libggml-cpu.so`. No board runtime or peer files changed.
The 56-count preflight exited 0 in 0.674 seconds. A genuine wrong-count
control using 54 exited 1 in 0.611 seconds. Logs remain at
`/tmp/geniex-gdn-case-count-preflight-{positive,negative}-20261001.log`.
These are local command wall times, not serving gains or an overall experiment
speedup. This preflight detects the count error before a board window starts.

Fresh repository-helper checks, verified by17:50 UTC: expected54 exits1 in0.606s;
expected56 exits0 in0.675s. The existing-log control exits1 before enumeration.
The positive-confirm receipt SHA256 remains
`5c77a3163392cc947bbd779c8d1b33259f758c7be131e9f14232746ab3cc0248`.
Logs `/tmp/geniex-gdn-case-count-repository-{negative,positive-confirm}-20261001.log`.
Candidate still awaits the board turn; these tests do not qualify its K2 path.
The previous17:58 heading used an estimated time ahead of the clock. The
17:50:13 UTC clock query corrects that reporting error; raw results unchanged.

## Extended K2 simulator coverage — verified by18:08 UTC

Session56868 completed with exit0. Controller
`/tmp/geniex-gdn-k2-paired-sim-20261001.sh` runs20 shapes/flags per arm in two
oneCPU/2GiB low-priority containers. New isolated worker75867f43 and runner
fe88ca1b remain under `/tmp/geniex-gdn-k2-sim-fixture-20261001` on both hosts.
The fixture adds K<=T, four-token capacity, both latest-first snapshots,
whole output/state guards and K in the result marker. It retains K1 hash pins.
Source baseline0fd9e6c4/candidatef5411a85 are pinned; no production edits.
All20 complete binary output/state comparisons pass, including S3/H8/T4/K1
and K2 with scalar/vector gates under ordinary and vectorized builds.
Seven negatives pass: two genuine slot1 mutations, output/state/input faults,
and wrong scalar4 source-state checks for both arms. Reference limit stays1e-7.
Gate wall157.282022091s is not speed. This runner does not enable LTO.
Baseline K2 also passes in this simulator, so it does not reproduce the board
cache-line failure. Board qualification remains required, not replaced.

Full local/Plex receipts `/tmp/geniex-gdn-k2-sim-results-20261001`.
Independent check: `node notes/check-gdn-k2-sim.cjs RECEIPT_DIR` exits0 and
verifies all20 pairs/seven negatives. Passing only the baseline subdirectory
exits1 with missing arm receipt, as expected. Luna6 static audit found no
fixture/controller defect; it did not verify remote source pins. Main did.

## Board red-green qualification — 18:27 UTC

TensorFold released its window18:24:45. Controller38357 acquired board.lock
and ran18:27:15–18:27:37. ControllerSHA
b7f10005164fd091cc4c1d3eb2c45cb04f3dc6f39f8e46c435b9a83c03d0c24a.
Candidate v75ed83c953, native4095ffe5, runner4fc49fd1. Host case-count
preflight freshly selected56 before board ownership. Same pinned baseline
and candidate as the recorded full cross-build and simulator coverage.

Baseline private cases pass4/6 at one and four workers. Candidate passes6/6
at both counts. Baseline repository suite passes54/56 at one worker.
Candidate repository suite passes56/56 at one and four workers, with one
actual HTP GATED_DELTA_NET profile per case. CPU reference includes both
attention and all state snapshots. No tolerance changed.

Main independently checked every captured output length and finite value.
All six candidate outputs match byte-for-byte across worker counts. All
six baseline outputs also match across worker counts. Four already-passing
controls remain byte-for-byte unchanged. Only the two known K2 cases change.

Controller exited0 and restored both original library hashes. Independent
SSH hash check confirmed restoration. No test process remains. Reservation
released18:27:37 with next ownerTensorFold. Receipts retained locally at
/tmp/geniex-gdn-snapshot-line-r2-receipts-20261001 and on board at
/home/arduino/geniex-bonsai/gdn-snapshot-line-fix-r2-20261001.
This is a correctness qualification, not a serving or speed measurement.
Aligned S128 model routing remains unchanged. No new performance gain.
