# B12 board result: unused QH high-half work

Base: PR58,3190d784f76e2bf68d2aa001bb99a2678b44fc33.
Candidate removes QH high-half stores AND reads. Store-only removal fails
poisoned-scratch correctness. QS high halves, DMA, dispatch and VTCM unchanged.

Target: VENTUNO Q HTP v75, Bonsai2 PTQ1_0 GGUF5946648928 bytes, SHA256
53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3.
SDK6.6/Tools19.0.07; ARM64 host and all HTP cross-built. Same production flags.
Host6f1c1319 both arms; baseline DSP3296b398, candidate DSPf98ff4be.
Header SHA8d26e491311a9881f39469010065c7a67027fde87caa742ba8124cf358f9f812.

## Serving result

Actual30 prompt tokens/8 generated, context512, temperature0/seed42,
four HTP sessions/four workers, one warmup and one measured run per process.
Alternating B/C,C/B,B/C. Exact text and length stop matched in every run.
No clocks changed. Shared board.lock held11:00:01–11:29:27 UTC.

| Pair | Base prompt/decode us | Candidate prompt/decode us | Prefill gain | Decode gain |
|---|---|---|---|---|
| 1 | 106300880/33939605 | 105190730/33151225 | 1.055369% | 2.378132% |
| 2 | 106233300/33085835 | 105184872/33082342 | 0.996748% | 0.010559% |
| 3 | 106304463/33479288 | 105237466/33445601 | 1.013895% | 0.100722% |

Median-rate prefill0.2822177954→0.2851962335 tokens/s: +1.055368662%.
Median-rate decode0.2389537077→0.2413183827 tokens/s: +0.989595407%.
Paired median gains: +1.013894614% prefill/+0.100721766% decode.
Median TTFT106313517→105194492us. Warmups/thermal snapshots retained below.
Unchanged-baseline time spread:0.066987470% prefill/2.580469860% decode.
This is observed repeat spread, not a fixed noise floor or confidence bound.
Keep for consistent prefill improvement. Decode gain is less stable; do not
present the first pair or median-rate gain as a guaranteed repeatable gain.
This workload does not establish performance at longer contexts/concurrency.

Command per arm, from the board runtime with the pinned libraries installed:

```bash
./geniex-bench --plugin llama_cpp --device npu --device-id HTP0,HTP1,HTP2,HTP3 \
  -m models/Ternary-Bonsai-2-27B-PTQ1_0.gguf --prompt-file count-templated.txt \
  -n 8 -c 512 -t 4 --warmup 1 -r 1 --output-json ARM.json
```

Full controller: /tmp/geniex-board-qh-dead-high-window-20261001.sh.
Board receipts: /home/arduino/geniex-bonsai/b12-20261001.
Local full copy: /tmp/geniex-b12-board-receipts-20261001.
Local six-arm JSON/log prefix: /tmp/geniex-b12-pair-N-ARM-20261001.
Verifier: `node /tmp/geniex-board-direct-two-row-base58-results-20261001.js /tmp/geniex-b12`.
Controller exit0; original host65e2bd27/DSPcee1a3e8 restored and independently
verified. Reservation released to TensorFold root agent. No merge.

## Correctness and review

Both board gates:45/45 numerical cases,45 exact output hashes,33 batch routes,
four real workers. Strict receipt checker and negative self-tests pass.
Simulator worker:32 exact outputs/8 tile controls/6 VTCM rejects.
Four ordinary poison/prep/reduce controls pass; unchanged-store and store-only
consumer negative controls fail. Command on Plex:
`QH_VECTOR_MODES=off bash /tmp/geniex-b12-qh-poison-gate-20261001.sh SOURCE`.
New repo runner: `tests/hexagon/run-ptq1-baseline-bits.sh BASELINE_SDK_HEXAGON`.
It verifies54 within-flag cases/3672 exact output and padding words.

Independent scalar comparison fails under explicit-fvectorize for BOTH accepted
baseline and candidate. These failures remain recorded, not waived. Accepted
baseline changes all27 cases across ordinary/explicit-vectorization flags;
compare baseline/candidate within flags only. Production flags were not changed.
Simulator Pcycles gains0.016644–0.080743% are not board serving gains.
Supported-file Snyk high0; local CodeRabbit all4 code files/findings0; read-only
CursorAuto findings checked against actual source/test receipts. No full-repo
or exact-PR-head review claim. Request exact-head Sonar/Gitar/Bugbot after PR.
