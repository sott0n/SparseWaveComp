//===- SparseBenchmarkUtils.cpp - Benchmark runner utilities -----*- C++
//-*-===//
//
// Part of the SparseWave project.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#ifdef _WIN32
#define SPARSEWAVE_BENCHMARK_EXPORT __declspec(dllexport)
#else
#define SPARSEWAVE_BENCHMARK_EXPORT __attribute__((visibility("default")))
#endif

namespace {

constexpr char magic[] = "SWCSR001";

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "SparseWave benchmark input error: " << message << '\n';
  std::abort();
}

template <typename T>
void readArray(std::ifstream &stream, StridedMemRefType<T, 1> *destination,
               int64_t expectedSize, const char *name) {
  if (destination->sizes[0] != expectedSize || destination->strides[0] != 1)
    fail(std::string(name) + " memref has an unexpected layout");
  stream.read(reinterpret_cast<char *>(destination->data + destination->offset),
              expectedSize * sizeof(T));
  if (!stream)
    fail(std::string("could not read ") + name);
}

uint64_t readU64(std::ifstream &stream, const char *name) {
  uint64_t value = 0;
  stream.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!stream)
    fail(std::string("could not read ") + name);
  return value;
}

struct CSRDimensions {
  uint64_t rows;
  uint64_t columns;
  uint64_t nnz;
};

CSRDimensions loadCSRInputs(StridedMemRefType<int32_t, 1> *rowOffsets,
                            StridedMemRefType<int32_t, 1> *columnIndices,
                            StridedMemRefType<float, 1> *values) {
  const char *path = std::getenv("SPARSEWAVE_BENCHMARK_CSR");
  if (!path)
    fail("SPARSEWAVE_BENCHMARK_CSR is not set");

  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail(std::string("could not open ") + path);

  char fileMagic[sizeof(magic) - 1];
  stream.read(fileMagic, sizeof(fileMagic));
  if (!stream || !std::equal(std::begin(fileMagic), std::end(fileMagic),
                             std::begin(magic)))
    fail("invalid CSR binary header");

  CSRDimensions dimensions{
      readU64(stream, "row count"),
      readU64(stream, "column count"),
      readU64(stream, "NNZ count"),
  };
  if (dimensions.rows > std::numeric_limits<int64_t>::max() ||
      dimensions.columns > std::numeric_limits<int64_t>::max() ||
      dimensions.nnz > std::numeric_limits<int64_t>::max())
    fail("CSR dimensions exceed the runner limits");

  readArray(stream, rowOffsets, static_cast<int64_t>(dimensions.rows + 1),
            "row offsets");
  readArray(stream, columnIndices, static_cast<int64_t>(dimensions.nnz),
            "column indices");
  readArray(stream, values, static_cast<int64_t>(dimensions.nnz), "values");
  return dimensions;
}

template <typename T>
T &element(StridedMemRefType<T, 2> *memref, int64_t row, int64_t column) {
  return memref->data[memref->offset + row * memref->strides[0] +
                      column * memref->strides[1]];
}

} // namespace

extern "C" SPARSEWAVE_BENCHMARK_EXPORT void
_mlir_ciface_loadSpMVBenchmarkInputs(
    StridedMemRefType<int32_t, 1> *rowOffsets,
    StridedMemRefType<int32_t, 1> *columnIndices,
    StridedMemRefType<float, 1> *values, StridedMemRefType<float, 1> *vector,
    StridedMemRefType<float, 1> *expected) {
  CSRDimensions dimensions = loadCSRInputs(rowOffsets, columnIndices, values);

  if (vector->sizes[0] != static_cast<int64_t>(dimensions.columns) ||
      vector->strides[0] != 1)
    fail("vector memref has an unexpected layout");
  std::fill_n(vector->data + vector->offset, dimensions.columns, 1.0f);

  if (expected->sizes[0] != static_cast<int64_t>(dimensions.rows) ||
      expected->strides[0] != 1)
    fail("expected-output memref has an unexpected layout");
  for (uint64_t row = 0; row < dimensions.rows; ++row) {
    float sum = 0.0f;
    for (int32_t position = rowOffsets->data[rowOffsets->offset + row];
         position < rowOffsets->data[rowOffsets->offset + row + 1]; ++position)
      sum += values->data[values->offset + position];
    expected->data[expected->offset + row] = sum;
  }
}

extern "C" SPARSEWAVE_BENCHMARK_EXPORT void
_mlir_ciface_loadSpMMBenchmarkInputs(
    StridedMemRefType<int32_t, 1> *rowOffsets,
    StridedMemRefType<int32_t, 1> *columnIndices,
    StridedMemRefType<float, 1> *values, StridedMemRefType<float, 2> *rhs,
    StridedMemRefType<float, 2> *expected) {
  CSRDimensions dimensions = loadCSRInputs(rowOffsets, columnIndices, values);
  if (rhs->sizes[0] != static_cast<int64_t>(dimensions.columns) ||
      rhs->sizes[1] < 1 || rhs->strides[1] != 1 ||
      rhs->strides[0] != rhs->sizes[1])
    fail("RHS memref has an unexpected layout");
  if (expected->sizes[0] != static_cast<int64_t>(dimensions.rows) ||
      expected->sizes[1] != rhs->sizes[1] || expected->strides[1] != 1 ||
      expected->strides[0] != expected->sizes[1])
    fail("expected-output memref has an unexpected layout");

  for (uint64_t reduction = 0; reduction < dimensions.columns; ++reduction)
    for (int64_t column = 0; column < rhs->sizes[1]; ++column)
      element(rhs, reduction, column) =
          1.0f + static_cast<float>((reduction + column) % 7) * 0.125f;

  for (uint64_t row = 0; row < dimensions.rows; ++row) {
    for (int64_t column = 0; column < rhs->sizes[1]; ++column) {
      float sum = 0.0f;
      for (int32_t position = rowOffsets->data[rowOffsets->offset + row];
           position < rowOffsets->data[rowOffsets->offset + row + 1];
           ++position) {
        int32_t reductionIndex =
            columnIndices->data[columnIndices->offset + position];
        sum += values->data[values->offset + position] *
               element(rhs, reductionIndex, column);
      }
      element(expected, row, column) = sum;
    }
  }
}
