# CSR elementwise block-size sweep on gfx1101

This report records the effect of GPU block size on SparseWave's symbolic,
prefix-scan, and numeric phases for CSR elementwise add and multiply. It is a
snapshot for the compiler and software revisions below, not a general
performance guarantee.

## Environment

| Component | Value |
| --- | --- |
| Date | 2026-11-01 |
| GPU | AMD Radeon RX 7800 XT |
| Target chip | gfx1101 |
| Wavefront size | 32 |
| SparseWave commit | `183e05fb196284004bb9815853f918b0711af347` |
| LLVM commit | `64b593c2371b7f7225b0ec190a37cd8b672e4c5d` |
| ROCm version | 6.4.4-129 |
| Warmup dispatches | 20 |
| Measured dispatches | 100 |

All 48 measured configurations passed output validation. The inputs have
65,536 rows and columns, 4 or 32 nonzeros per input row, and 25%, 50%, or 100%
structural overlap. The sweep compares block sizes 32, 64, 128, and 256.

## Benchmark method

The runner generates two sorted CSR inputs with the requested overlap. Each
dispatch executes five kernels grouped into three logical phases:

1. symbolic intersection or union counting;
2. block-local scan, block-total scan, and block-offset addition;
3. numeric intersection or union materialization.

`rocprofv3` kernel timestamps exclude compilation, allocation, transfer, and
validation. Total latency is the sum of all five kernels for each dispatch;
the table reports the median of those dispatch totals.

## Reproduction

```sh
python3 benchmark/run_elementwise_benchmark.py \
  --chip=gfx1101 \
  --rows=65536 \
  --columns=65536 \
  --nnz-per-row=4,32 \
  --overlaps=25,50,100 \
  --operations=add,multiply \
  --block-sizes=32,64,128,256 \
  --warmup=20 \
  --iterations=100
```

Add `--keep-artifacts` to retain generated MLIR, profiler traces, and embedded
GPU binaries for resource and ISA inspection.

## Results

Median complete-dispatch latency in microseconds is shown below. Bold marks
the fastest block size for each workload.

| Operation | NNZ/row | Overlap | Block 32 | Block 64 | Block 128 | Block 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Add | 4 | 25% | 129.62 | 132.20 | 94.44 | **87.66** |
| Add | 4 | 50% | 76.64 | 74.56 | 71.36 | **70.84** |
| Add | 4 | 100% | 62.24 | 51.88 | 47.42 | **37.32** |
| Add | 32 | 25% | **1,909.83** | 2,029.89 | 1,966.73 | 1,957.47 |
| Add | 32 | 50% | **1,617.53** | 1,744.31 | 1,659.67 | 1,650.07 |
| Add | 32 | 100% | **965.04** | 1,005.81 | 983.60 | 987.00 |
| Multiply | 4 | 25% | 48.60 | 31.56 | 25.60 | **25.04** |
| Multiply | 4 | 50% | 51.24 | 32.76 | 27.64 | **27.32** |
| Multiply | 4 | 100% | 57.32 | 53.20 | 46.88 | **46.44** |
| Multiply | 32 | 25% | **657.80** | 683.50 | 669.70 | 674.68 |
| Multiply | 32 | 50% | **757.02** | 782.40 | 759.82 | 771.22 |
| Multiply | 32 | 100% | 963.75 | 968.71 | **953.25** | 980.30 |

The following values are geometric means of each block size divided by block
32 for the same workload. Lower is better.

| Block size | Symbolic | Prefix | Numeric | Total |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1.000 | 1.000 | 1.000 | 1.000 |
| 64 | 1.001 | 0.481 | 1.202 | 0.929 |
| 128 | 0.966 | 0.335 | 1.160 | 0.844 |
| 256 | 0.979 | 0.316 | 1.122 | 0.822 |

Block 256 reduces prefix time to 31.6% of block 32, a 3.17x improvement. The
complete dispatch improves by 1.22x in the geometric mean, but the phase data
shows that the prefix gain does not determine every workload's optimum:
numeric time is higher on average for the larger blocks.

## GPU resources

The following code-object metadata was extracted from a representative
32-NNZ/row, 50%-overlap input. Counts are unchanged across the measured block
sizes except for prefix LDS allocation.

| Kernel | Add VGPR | Add SGPR | Multiply VGPR | Multiply SGPR | LDS bytes at blocks 32/64/128/256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Symbolic | 22 | 22 | 16 | 15 | 0 / 0 / 0 / 0 |
| Prefix local | 21 | 26 | 21 | 26 | 4 / 8 / 16 / 32 |
| Prefix blocks | 24 | 30 | 24 | 30 | 4 / 8 / 16 / 32 |
| Prefix add | 12 | 24 | 12 | 24 | 0 / 0 / 0 / 0 |
| Numeric | 22 | 26 | 18 | 24 | 0 / 0 / 0 / 0 |

No configuration uses scratch memory or spills VGPRs or SGPRs. The prefix
kernels store one 32-bit total per wave, so fixed LDS grows from one value at
block 32 to eight values at block 256 and remains negligible.

## Observations

- Short-row workloads benefit most from a larger block because prefix-scan
  overhead is a significant fraction of their total work. Block 256 wins all
  six 4-NNZ/row cases.
- Symbolic and numeric merging dominate the 32-NNZ/row cases. Block 32 wins
  five of those six workloads; reducing prefix latency cannot offset the
  numeric-phase slowdown.
- Block 128 and 256 have similar aggregate latency, while block 256 has the
  lowest prefix cost. The result supports block 256 as a reasonable static
  default for short rows, but not as a universal choice.
- A future matrix-profile decision should consider row length and operation,
  rather than selecting block size from scan performance alone.
