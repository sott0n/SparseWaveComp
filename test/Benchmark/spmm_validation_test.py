# RUN: %python %s %S/../../benchmark %sparsewave_benchmark_utils %t

import ctypes as ct
import math
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, sys.argv[1])
import benchmark_utils as common

LIBRARY = ct.CDLL(sys.argv[2])
ROOT = Path(sys.argv[3])
sys.argv = [sys.argv[0]]


def memref(dtype, shape):
    rank = len(shape)

    class Descriptor(ct.Structure):
        _fields_ = [
            ("base", ct.POINTER(dtype)),
            ("data", ct.POINTER(dtype)),
            ("offset", ct.c_int64),
            ("sizes", ct.c_int64 * rank),
            ("strides", ct.c_int64 * rank),
        ]

    storage = (dtype * math.prod(shape))()
    strides = [math.prod(shape[i + 1 :]) for i in range(rank)]
    descriptor = Descriptor(
        storage,
        storage,
        0,
        (ct.c_int64 * rank)(*shape),
        (ct.c_int64 * rank)(*strides),
    )
    return storage, descriptor


class VerificationTest(unittest.TestCase):
    def check_format(self, sparse_format):
        ROOT.mkdir(parents=True, exist_ok=True)
        path = ROOT / "cancellation.mtx"
        # RHS column zero is one at all three positions. f32 sequential
        # accumulation loses the middle term, while the f64 reference is one.
        path.write_text(
            "%%MatrixMarket matrix coordinate real general\n"
            "3 15 4\n1 1 16777216\n1 8 1\n1 15 -16777216\n2 1 2\n"
        )
        matrix = common.read_matrix_market(path)
        binary = ROOT / f"matrix.{sparse_format}"
        if sparse_format == "csr":
            common.write_csr_binary(binary, matrix)
            inputs = [
                memref(ct.c_int32, [4]),
                memref(ct.c_int32, [4]),
                memref(ct.c_float, [4]),
            ]
            rows, columns = 3, 15
            loader = LIBRARY._mlir_ciface_loadSpMMBenchmarkInputs
        else:
            bsr = common.convert_to_bsr(matrix, 2)
            common.write_bsr_binary(binary, bsr)
            inputs = [
                memref(ct.c_int32, [len(bsr["block_row_offsets"])]),
                memref(ct.c_int32, [bsr["nnzb"]]),
                memref(ct.c_float, [len(bsr["block_values"])]),
            ]
            rows, columns = bsr["rows"], bsr["columns"]
            loader = LIBRARY._mlir_ciface_loadBSRSpMMBenchmarkInputs
        os.environ[f"SPARSEWAVE_BENCHMARK_{sparse_format.upper()}"] = str(binary)
        rhs = memref(ct.c_float, [columns, 1])
        expected = memref(ct.c_double, [rows, 1])
        bounds = memref(ct.c_double, [rows, 1])
        loader.restype = None
        loader(*(ct.byref(item[1]) for item in inputs + [rhs, expected, bounds]))
        self.assertEqual(expected[0][0], 1.0)
        self.assertEqual(expected[0][1], 2.0)
        self.assertEqual(expected[0][2], 0.0)
        ku = 4 * 2**-24
        self.assertAlmostEqual(bounds[0][0], ku / (1 - ku) * (2**25 + 1))
        self.assertEqual(bounds[0][1], 2e-4)
        self.assertEqual(bounds[0][2], 1e-4)
        actual = memref(ct.c_float, [rows, 1])
        actual[0][1] = 2.0
        verify = LIBRARY._mlir_ciface_verifySpMMBenchmarkOutput
        verify.restype = ct.c_int64

        def count():
            return verify(
                *(ct.byref(item[1]) for item in [actual, expected, bounds])
            )

        # Both legal accumulation orders pass, without changing the floor
        # for well-conditioned or empty outputs.
        self.assertEqual(count(), 0)
        actual[0][0] = 1.0
        self.assertEqual(count(), 0)
        for invalid in [32.0, float("nan"), float("inf"), -float("inf")]:
            actual[0][0] = invalid
            self.assertEqual(count(), 1)
        actual[0][0] = 1.0
        actual[0][1] = 2.01
        actual[0][2] = 0.01
        self.assertEqual(count(), 2)
        actual[0][1] = 2.0
        actual[0][2] = 0.0
        for invalid in [float("nan"), float("inf"), -1.0]:
            bounds[0][0] = invalid
            self.assertEqual(count(), 1)

    def test_csr_reference_and_validation(self):
        self.check_format("csr")

    def test_bsr_reference_and_validation(self):
        self.check_format("bsr")


unittest.main()
