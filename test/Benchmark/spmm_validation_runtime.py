# REQUIRES: amdgpu-runtime
# RUN: %python %s %S/../../benchmark %t sparsewave-opt %amdgpu_chip %mlir_rocm_runtime %mlir_runner_utils %sparsewave_benchmark_utils

import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, sys.argv[1])
import run_spmm_benchmark as benchmark

root = Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
opt, chip, runtime, runner_utils, benchmark_utils = sys.argv[3:]
matrix_path = root / "cancellation.mtx"
matrix_path.write_text(
    "%%MatrixMarket matrix coordinate real general\n"
    "3 15 4\n1 1 16777216\n1 8 1\n1 15 -16777216\n2 1 2\n"
)
matrix = benchmark.common.read_matrix_market(matrix_path)

for sparse_format in ["csr", "bsr"]:
    binary = root / f"matrix.{sparse_format}"
    if sparse_format == "csr":
        benchmark.common.write_csr_binary(binary, matrix)
        source_text = benchmark.render_mlir(
            Path(sys.argv[1]) / "spmm.mlir.in", matrix, 8, 3
        )
        mappings = benchmark.MAPPINGS
    else:
        bsr = benchmark.common.convert_to_bsr(matrix, 2)
        benchmark.common.write_bsr_binary(binary, bsr)
        source_text = benchmark.render_bsr_mlir(
            Path(sys.argv[1]) / "bsr_spmm.mlir.in", bsr, 8, 3
        )
        mappings = [benchmark.BASELINE_MAPPING]
    source = root / f"{sparse_format}.mlir"
    source.write_text(source_text)
    for mapping in mappings:
        compiled = root / f"{sparse_format}-{mapping}.mlir"
        pipeline = (
            "builtin.module(convert-scf-to-cf,"
            "sparsewave-to-amdgpu-pipeline{"
            f"chip={chip} wavefront-size=32 spmm-block-size=64 "
            f"spmm-mapping={mapping}"
            "},reconcile-unrealized-casts)"
        )
        subprocess.run(
            [opt, str(source), f"--pass-pipeline={pipeline}",
             "-o", str(compiled)],
            check=True,
        )
        environment = os.environ.copy()
        environment[f"SPARSEWAVE_BENCHMARK_{sparse_format.upper()}"] = str(binary)
        result = subprocess.run(
            [
                "mlir-runner", str(compiled), "--entry-point-result=void",
                f"--shared-libs={runtime}", f"--shared-libs={runner_utils}",
                f"--shared-libs={benchmark_utils}",
            ],
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        assert result.stdout.strip().endswith("[0]"), (
            sparse_format, mapping, result.stdout, result.stderr
        )
