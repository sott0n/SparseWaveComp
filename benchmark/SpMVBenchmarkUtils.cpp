//===- SpMVBenchmarkUtils.cpp - SpMV benchmark runner utilities -*- C++ -*-===//
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

} // namespace

extern "C" SPARSEWAVE_BENCHMARK_EXPORT void
_mlir_ciface_loadSpMVBenchmarkInputs(
    StridedMemRefType<int32_t, 1> *rowOffsets,
    StridedMemRefType<int32_t, 1> *columnIndices,
    StridedMemRefType<float, 1> *values, StridedMemRefType<float, 1> *vector,
    StridedMemRefType<float, 1> *expected) {
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

  const uint64_t rows = readU64(stream, "row count");
  const uint64_t columns = readU64(stream, "column count");
  const uint64_t nnz = readU64(stream, "NNZ count");
  if (rows > std::numeric_limits<int64_t>::max() ||
      columns > std::numeric_limits<int64_t>::max() ||
      nnz > std::numeric_limits<int64_t>::max())
    fail("CSR dimensions exceed the runner limits");

  readArray(stream, rowOffsets, static_cast<int64_t>(rows + 1), "row offsets");
  readArray(stream, columnIndices, static_cast<int64_t>(nnz), "column indices");
  readArray(stream, values, static_cast<int64_t>(nnz), "values");

  if (vector->sizes[0] != static_cast<int64_t>(columns) ||
      vector->strides[0] != 1)
    fail("vector memref has an unexpected layout");
  std::fill_n(vector->data + vector->offset, columns, 1.0f);

  if (expected->sizes[0] != static_cast<int64_t>(rows) ||
      expected->strides[0] != 1)
    fail("expected-output memref has an unexpected layout");
  for (uint64_t row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (int32_t position = rowOffsets->data[rowOffsets->offset + row];
         position < rowOffsets->data[rowOffsets->offset + row + 1]; ++position)
      sum += values->data[values->offset + position];
    expected->data[expected->offset + row] = sum;
  }
}
