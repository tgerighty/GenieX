# Bonsai 2 software rig

Target: text-only `Ternary-Bonsai-2-27B-PTQ1_0.gguf` on the VENTUNO Q Hexagon HTP v75. [Arduino's board guide](https://docs.arduino.cc/tutorials/ventuno-q/llama-cpp/) identifies its NPU skeleton as `libQnnHtpV75Skel.so`; [Qualcomm's QCS8275 page](https://www.qualcomm.com/internet-of-things/products/iq8-series/iq-8275) also lists a V73 DSP core. The skeleton target, not that separate DSP label, sets the simulator and HTP build architecture here. PTQ1_0 is 5.95 GB; PQ2_0 is 7.21 GB. The F16 file is 53.8 GB and cannot fit in the board's 16 GB memory. Vision needs a separate projector. See the [model files](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/main).

The model needs [Prism's `prism` llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp/tree/prism) for its PTQ1_0 format and activation transform. The tested fork revision is `adfffbe`. GenieX can select that source with `GENIEX_LLAMA_CPP_SOURCE`. `GENIEX_PRISM_PTQ1_HEXAGON=ON` applies `sdk/patches/prism-ptq1-hexagon.patch` to that checkout during configuration; use a separate writable clone checked out at `adfffbe`. The vendored llama.cpp remains unchanged. GenieX's other vendored patches are not applied to an external source, so its extra HTP session handoff and per-model power-mode setter are absent from this build.

## Cross-build on an Intel host

Clone GenieX with submodules. Clone the Prism fork beside it and run `git -C /absolute/path/to/Prism-llama.cpp checkout adfffbe`. From the GenieX root, start the toolchain container (replace the Prism path):

```sh
docker run --rm -it --user "$(id -u):$(id -g)" --platform linux/amd64 \
  --volume "$(pwd):/workspace" \
  --volume "/absolute/path/to/Prism-llama.cpp:/prism" \
  --workdir /workspace/sdk \
  docker.io/qualcomm/geniex-toolchain-linux:v0.1.0 bash
```

Run these commands inside the container:

```sh
cmake -S . -B build-bonsai2 \
  -DCMAKE_TOOLCHAIN_FILE=/workspace/sdk/cmake/arm64-linux-gnu.cmake \
  -DCMAKE_BUILD_TYPE=Release -DPREBUILT_LIB_DIR=linux_aarch64 \
  -DGENIEX_LLAMA_CPP_SOURCE=/prism -DGENIEX_PRISM_PTQ1_HEXAGON=ON \
  -DGENIEX_PLUGIN_QAIRT=OFF -DGGML_CCACHE=OFF \
  -DGGML_HEXAGON=ON -DGGML_OPENCL=OFF \
  -DHEXAGON_SDK_ROOT=/opt/hexagon/6.6.0.0 \
  -DHEXAGON_TOOLS_ROOT=/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07
cmake --build build-bonsai2 --target geniex_llama_cpp --parallel 8
cmake --install build-bonsai2 --prefix pkg-bonsai2
```

The result contains ARM64 GenieX libraries and HTP skeletons for v73, v75, v79, and v81, including the VENTUNO Q target `libggml-htp-v75.so`. An Intel host cannot run these binaries. The PTQ1_0 HTP path supports single-token matrix-vector operations when K is a multiple of 128 and the VTCM budget fits. Prefill, fused operations, and unsupported shapes stay on CPU. The code has been cross-built, but it has not run full inference on an HTP device.

With a GenieX CLI built against this SDK, use `geniex pull prism-ml/Ternary-Bonsai-2-27B-gguf:PTQ1_0 --model-type llm`. The explicit model type keeps the vision projector out of the download.

## Software checks

The Intel GenieX build with Prism loaded the real PTQ1_0 GGUF and answered `4` to `What is 2 plus 2? Answer with one digit.` using `geniex-bench --plugin llama_cpp --device cpu --accuracy --no-think -c 2048 -t 8`.

`sdk/hexagon/ptq1_tile.h` defines the 28-byte block, 896-byte tile, and reversible host pack/unpack conversion. `sdk/hexagon/ptq1_hvx.h` contains the measured HVX kernel and a wrapper for flat Q8 activations across K blocks. `tests/hexagon/ptq1_sim.c` includes both headers and checks tile round-trips across two K blocks and a padded row, a 32-row HVX dot tile, all 128 activation basis positions, and a two-block flat Q8 dot with FP16 scales against a scalar reference. Build it with `hexagon-clang -I sdk/hexagon -mcpu=v75 -mv75 -mhvx=v75 -mhmx -O2 -Wall -Wextra -Werror` and run it with `hexagon-sim --march v75na_1 -r ptq1_sim_v75.elf`. The simulator may need an isolated `libncurses.so.5` compatibility library.

The same source has a `PTQ1_BENCH` build mode and a `PTQ1_SCALAR_PREP` comparison mode. In the v75 simulator at `-O2`, 256 preparation-plus-tile operations used 1,042,536 instructions and 1,317,132 simulated cycles with HVX activation preparation, versus 4,773,781 instructions and 5,118,024 simulated cycles with scalar preparation. Both gave checksum `-17701.5`. This test repeats preparation for each tile, so it does not model production reuse or predict model throughput.

`PTQ1_BENCH_STREAM` checks that reuse pattern: it prepares one activation block and runs 256 packed-weight tiles. Build the four variants below from the same source with the v75 `-O2` flags above, then run each with `hexagon-sim --march v75na_1 -r <file>.elf`. All four produced checksum `-23373.5`.

| Extra compiler flags | Decoder | Reduction | Instructions | Simulated cycles |
| --- | --- | --- | ---: | ---: |
| `-DPTQ1_SHIFT_DECODE -DPTQ1_SCALAR_REDUCE` | Shift/add | Scalar | 1,644,750 | 1,882,077 |
| `-DPTQ1_SCALAR_REDUCE` | HVX multiply | Scalar | 1,411,058 | 1,647,117 |
| `-DPTQ1_SHIFT_DECODE` | Shift/add | HVX | 1,389,702 | 1,646,205 |
| None | HVX multiply | HVX | 1,130,428 | 1,411,269 |

The two retained changes cut simulated cycles by 25.0% and instructions by 31.3% at v75 `-O2`. An instruction histogram showed 51,200 to 17,408 vector halfword adds and shifts and 158,019 to 25,923 scalar 32-bit loads. At `-O3` on v75, the same before/after source used 1,533,948/894,948 cycles; `-O3` is not the current HTP Release setting. A standalone `-O2 -flto -fvectorize` executable without `-fpic` used 1,791,870/1,011,981 cycles; `-flto -fpic` did not link as a standalone executable, so this is not a full production-flags result. The kernel also passed exhaustive packed-byte decode tests (256 codes at five powers), all 128 activation basis positions, and a scaled dot comparison. The synthetic tile test excludes GenieX, FastRPC, VTCM, memory transfers, and other model operations; these numbers cannot establish board latency or tokens per second.

The `PTQ1_BENCH_FULLK` mode uses K=5,120, 32 weight tiles (1,024 output rows), and flat Q8 activation data. A v75 `-O2` run gave 8,386,594 instructions and 9,592,770 simulated cycles for shift decoding plus scalar reduction, versus 5,815,056 instructions and 7,238,826 cycles for the optimized code. Unrolling the fixed-length decoder loops reduced the optimized result further to 5,640,994 instructions and 7,027,650 cycles: 26.7% fewer cycles than the original baseline. All three returned checksum `182651.0`. The v75 correctness test still matches the scalar reference, and `-fstack-usage` still reports 3,328 bytes for the dot function. An earlier `PTQ1_BENCH_REUSE` run prepared once per K block instead of once per weight tile: 5,217,298 instructions and 6,660,045 cycles with the same checksum. This form keeps 348,160 bytes of prepared activation data; integration must fit or chunk that data in VTCM. The synthetic run does not include activation quantization, DMA, FastRPC, or other model operations.

An earlier v73 `-O3` experiment built activation vectors inside each dot tile instead of preparing them once. It gave the same checksum but increased the streamed test from 894,924 to 972,408 simulated cycles (8.7% slower); the on-the-fly path was discarded pending a v75 comparison. The prepared form occupies 8,704 bytes per 128-activation block. The model's GGUF metadata gives embedding length 5,120 and feed-forward length 17,408, so a full prepared activation for those K sizes would occupy 348,160 or 1,183,744 bytes. Production integration must check VTCM capacity and may need K chunking.

The HTP worker stack is 16,384 bytes in Prism's `htp/main.c`. The v75 `-O2 -fstack-usage` simulator build reports 3,328 bytes for the HVX dot function and 11,776 bytes for the full-K benchmark's main function, which includes its activation scratch. A `-O2 -fpic -fstack-usage` compile of the paired HTP matvec source reports a 1,152-byte worker frame and the same 3,328-byte dot frame. These figures do not establish the final HTP call stack, but they make an 8,704-byte on-stack prepared activation unsafe to assume. The HTP path reserves explicit per-worker scratch and includes it in the VTCM budget.

The SDK patch adds PTQ1_0 weight repacking and a 128-wide HTP decode matvec path. It reserves 8,704 bytes of per-worker VTCM scratch for activation preparation. The paired path now prepares each activation block once for two weight tiles, using the existing two-tile DMA prefetch buffers. In the kernel-only `PTQ1_BENCH_PAIR` test with the production pair helper, the per-tile path used 5,641,084 instructions and 7,027,770 cycles; the paired path used 5,310,023 instructions and 6,708,048 cycles, a 4.5% cycle reduction. Both returned checksum `182651.0`.

`tests/hexagon/ptq1_dma_sim.c` adds Prism's real `dma-queue.c` to the v75 simulator test. Compile it at `-O2` with the Prism HTP headers, the SDK's `computev75/include/qurt` headers, and the same v75 flags as above. Compile once without extra flags and once with `-DPTQ1_DMA_PAIR`; both use 32 weight tiles, K=5,120, and two DMA buffers. The one-tile path took 5,509,607 instructions and 6,892,710 simulator Pcycles. The paired path took 5,178,701 instructions and 6,571,959 Pcycles, a 4.7% cycle reduction. Both returned checksum `-4122.0`. A closer `-fpic` comparison, with position-independent objects linked into standalone executables, gave 6,888,819 versus 6,573,687 Pcycles, a 4.6% reduction. Builds with `-DPTQ1_DMA_VERIFY` compared every output with the direct kernel, and paired builds with `-DPTQ1_DMA_TILES=1` and `=3` passed the single and odd-final-tile cases. The simulator target buffer is static memory, not a VENTUNO Q VTCM allocation, so these measurements do not predict board bandwidth or model throughput.

From the toolchain container above, run `/workspace/tests/hexagon/run-ptq1-dma-bench.sh /prism` to repeat the `-fpic` comparison and compare DMA outputs with the direct kernel for 32, 1, and 3 tiles. The simulator may need a compatible `libncurses.so.5` path.

The faster full-K reuse experiment above is not integrated because it needs much more per-worker scratch. These checks do not run full model inference on the Hexagon. A VENTUNO Q run is required before NPU support, model accuracy, latency, or tokens per second can be claimed.

The real GGUF has an `output.weight` PTQ1_0 matrix of shape `[5120, 248320]`. The HTP eligibility check currently caps PTQ1_0 at 32,768 output rows, so the output head remains on CPU. This cap is also a VTCM safeguard: the current flat matvec layout reserves a full output row per HTP worker. At eight workers, that output buffer alone would use 7,946,240 bytes, before weight tiles, activation scratch, and other buffers, exceeding the default 8 MiB budget in total. A per-worker output slice could remove that memory limit, but moving this large matrix to HTP may still be slower. Keep the cap until a device can compare complete decode latency with and without output-head offload.
