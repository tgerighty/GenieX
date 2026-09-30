#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Check real runner dispatch/configuration; does not test NPU arithmetic."""
import os
from pathlib import Path
import subprocess
import tempfile

RUNNER = Path(__file__).with_name("run-ptq1-dma-bench.sh")
SHAPES = [
    (256, 3, 81), (5120, 3, 81), (5120, 3, 273), (5120, 4, 273),
    (5120, 8, 273), (5120, 9, 273), (5120, 16, 256), (6144, 8, 256),
    (6144, 9, 273), (6144, 16, 256), (5120, 1, 273), (17408, 1, 145),
    (17408, 3, 97), (17408, 3, 256), (17408, 4, 256), (17408, 5, 256),
    (17408, 5, 273), (17408, 8, 256),
]


def main():
    # Replace only recursive invocations with a shape recorder. The actual
    # parent's dispatch, validation and shard selection execute unchanged.
    with tempfile.TemporaryDirectory(prefix="ptq1-dispatch-") as tmp:
        root = Path(tmp)
        recorder = root / "record"
        recorder.write_text(
            '#!/usr/bin/env bash\n'
            'printf "%s %s %s\\n" "${PTQ1_WORKER_K:-256}" '
            '"${PTQ1_WORKER_M:-3}" "${PTQ1_WORKER_N:-81}"\n'
        )
        recorder.chmod(0o700)
        runner = root / "runner"
        runner.write_text(RUNNER.read_text().replace(
            '"$0" "${1}" --worker', f'"{recorder}" "${{1}}" --worker'
        ))
        runner.chmod(0o700)
        env = {k: v for k, v in os.environ.items() if not k.startswith("PTQ1_")}

        def run(**options):
            return subprocess.run(
                [str(runner), "/unused", "--worker"], env={**env, **options},
                capture_output=True, text=True, check=False,
            )

        default = run()
        assert default.returncode == 0, default.stderr
        assert [tuple(map(int, row.split())) for row in default.stdout.splitlines()] == SHAPES
        for count in (1, 3, 4):
            union = []
            for index in range(count):
                result = run(PTQ1_BENCH_SHARD_INDEX=str(index), PTQ1_BENCH_SHARD_COUNT=str(count))
                assert result.returncode == 0, result.stderr
                shapes = [tuple(map(int, row.split())) for row in result.stdout.splitlines()]
                assert shapes == SHAPES[index::count], shapes
                union.extend(shapes)
            assert sorted(union) == sorted(SHAPES)
        for options in (
            {"PTQ1_BENCH_SHARD_INDEX": "0"},
            {"PTQ1_BENCH_SHARD_INDEX": "", "PTQ1_BENCH_SHARD_COUNT": ""},
            {"PTQ1_BENCH_SHARD_INDEX": "4", "PTQ1_BENCH_SHARD_COUNT": "4"},
            {"PTQ1_BENCH_SHARD_INDEX": "00", "PTQ1_BENCH_SHARD_COUNT": "4"},
            {"PTQ1_BENCH_SHARD_INDEX": "0", "PTQ1_BENCH_SHARD_COUNT": "0"},
            {"PTQ1_BENCH_SHARD_INDEX": "0", "PTQ1_BENCH_SHARD_COUNT": "4", "PTQ1_BENCH_BATCH": "1"},
            {"PTQ1_BENCH_SHARD_INDEX": "0", "PTQ1_BENCH_SHARD_COUNT": "4", "PTQ1_WORKER_M": "1"},
            {"PTQ1_BENCH_SHARD_INDEX": "0", "PTQ1_BENCH_SHARD_COUNT": "4", "PTQ1_WORKER_N": "33"},
            {"PTQ1_BENCH_BATCH": "2"},
            {"PTQ1_BENCH_CACHE_DIR": "relative"},
        ):
            result = run(**options)
            assert result.returncode != 0, options
        link = root / "cache-link"
        link.symlink_to(root)
        assert run(PTQ1_BENCH_CACHE_DIR=str(link)).returncode != 0
        for suffix in ("/", "//", "/."):
            assert run(PTQ1_BENCH_CACHE_DIR=str(link) + suffix).returncode != 0
    print("PTQ1 dispatch/configuration checks passed; numerical/cache checks require the SDK")


if __name__ == "__main__":
    main()
