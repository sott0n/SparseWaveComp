//===- SparseBenchmarkUtils.cpp - Benchmark runner utilities -----*- C++
//-*-===//
//
// Part of the SparseWave project.
//
//===----------------------------------------------------------------------===//

#include "BenchmarkVerification.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifdef _WIN32
#define SPARSEWAVE_BENCHMARK_EXPORT __declspec(dllexport)
#else
#define SPARSEWAVE_BENCHMARK_EXPORT __attribute__((visibility("default")))
#endif

namespace {

constexpr char csrMagic[] = "SWCSR001";
constexpr char cooMagic[] = "SWCOO001";
constexpr char bsrMagic[] = "SWBSR001";

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

struct SparseDimensions {
  uint64_t rows;
  uint64_t columns;
  uint64_t nnz;
};

struct BSRDimensions : SparseDimensions {
  uint64_t blockSize;
  uint64_t originalRows;
  uint64_t originalColumns;
  uint64_t originalNnz;
  std::vector<int32_t> inputRows;
  std::vector<int32_t> inputColumns;
  std::vector<float> inputValues;
};

template <typename T>
std::vector<T> readVector(std::ifstream &stream, uint64_t size,
                          const char *name) {
  if (size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    fail(std::string(name) + " exceeds the runner limits");
  std::vector<T> values(size);
  stream.read(reinterpret_cast<char *>(values.data()), size * sizeof(T));
  if (!stream)
    fail(std::string("could not read ") + name);
  return values;
}

SparseDimensions readSparseHeader(std::ifstream &stream, const char *magic,
                                  const char *format) {
  char fileMagic[8];
  stream.read(fileMagic, sizeof(fileMagic));
  if (!stream || !std::equal(std::begin(fileMagic), std::end(fileMagic), magic))
    fail(std::string("invalid ") + format + " binary header");

  SparseDimensions dimensions{
      readU64(stream, "row count"),
      readU64(stream, "column count"),
      readU64(stream, "NNZ count"),
  };
  if (dimensions.rows > std::numeric_limits<int64_t>::max() ||
      dimensions.columns > std::numeric_limits<int64_t>::max() ||
      dimensions.nnz > std::numeric_limits<int64_t>::max())
    fail(std::string(format) + " dimensions exceed the runner limits");
  return dimensions;
}

SparseDimensions loadCSRInputs(StridedMemRefType<int32_t, 1> *rowOffsets,
                               StridedMemRefType<int32_t, 1> *columnIndices,
                               StridedMemRefType<float, 1> *values) {
  const char *path = std::getenv("SPARSEWAVE_BENCHMARK_CSR");
  if (!path)
    fail("SPARSEWAVE_BENCHMARK_CSR is not set");

  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail(std::string("could not open ") + path);

  SparseDimensions dimensions = readSparseHeader(stream, csrMagic, "CSR");

  readArray(stream, rowOffsets, static_cast<int64_t>(dimensions.rows + 1),
            "row offsets");
  readArray(stream, columnIndices, static_cast<int64_t>(dimensions.nnz),
            "column indices");
  readArray(stream, values, static_cast<int64_t>(dimensions.nnz), "values");
  return dimensions;
}

SparseDimensions loadCOOInputs(StridedMemRefType<int32_t, 1> *rowIndices,
                               StridedMemRefType<int32_t, 1> *columnIndices,
                               StridedMemRefType<float, 1> *values) {
  const char *path = std::getenv("SPARSEWAVE_BENCHMARK_COO");
  if (!path)
    fail("SPARSEWAVE_BENCHMARK_COO is not set");

  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail(std::string("could not open ") + path);

  SparseDimensions dimensions = readSparseHeader(stream, cooMagic, "COO");
  readArray(stream, rowIndices, static_cast<int64_t>(dimensions.nnz),
            "row indices");
  readArray(stream, columnIndices, static_cast<int64_t>(dimensions.nnz),
            "column indices");
  readArray(stream, values, static_cast<int64_t>(dimensions.nnz), "values");
  return dimensions;
}

BSRDimensions loadBSRInputs(StridedMemRefType<int32_t, 1> *blockRowOffsets,
                            StridedMemRefType<int32_t, 1> *blockColumnIndices,
                            StridedMemRefType<float, 1> *blockValues) {
  const char *path = std::getenv("SPARSEWAVE_BENCHMARK_BSR");
  if (!path)
    fail("SPARSEWAVE_BENCHMARK_BSR is not set");

  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail(std::string("could not open ") + path);

  SparseDimensions sparse = readSparseHeader(stream, bsrMagic, "BSR");
  uint64_t blockSize = readU64(stream, "block size");
  uint64_t originalRows = readU64(stream, "original row count");
  uint64_t originalColumns = readU64(stream, "original column count");
  uint64_t originalNnz = readU64(stream, "original NNZ count");
  if (blockSize == 0 || sparse.rows % blockSize != 0 ||
      sparse.columns % blockSize != 0)
    fail("BSR dimensions must be divisible by a positive block size");
  if (originalRows > sparse.rows || originalColumns > sparse.columns)
    fail("original BSR dimensions exceed the padded dimensions");
  if (blockSize > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      sparse.nnz > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                       blockSize / blockSize)
    fail("BSR block values exceed the runner limits");

  readArray(stream, blockRowOffsets,
            static_cast<int64_t>(sparse.rows / blockSize + 1),
            "block-row offsets");
  readArray(stream, blockColumnIndices, static_cast<int64_t>(sparse.nnz),
            "block-column indices");
  readArray(stream, blockValues,
            static_cast<int64_t>(sparse.nnz * blockSize * blockSize),
            "block values");
  return {sparse,
          blockSize,
          originalRows,
          originalColumns,
          originalNnz,
          readVector<int32_t>(stream, originalNnz, "input row indices"),
          readVector<int32_t>(stream, originalNnz, "input column indices"),
          readVector<float>(stream, originalNnz, "input values")};
}

void initializeVector(StridedMemRefType<float, 1> *vector,
                      uint64_t columnCount) {
  if (vector->sizes[0] != static_cast<int64_t>(columnCount) ||
      vector->strides[0] != 1)
    fail("vector memref has an unexpected layout");
  std::fill_n(vector->data + vector->offset, columnCount, 1.0f);
}

void initializeExpected(StridedMemRefType<float, 1> *expected,
                        uint64_t rowCount) {
  if (expected->sizes[0] != static_cast<int64_t>(rowCount) ||
      expected->strides[0] != 1)
    fail("expected-output memref has an unexpected layout");
  std::fill_n(expected->data + expected->offset, rowCount, 0.0f);
}

template <typename T>
T &element(StridedMemRefType<T, 2> *memref, int64_t row, int64_t column) {
  return memref->data[memref->offset + row * memref->strides[0] +
                      column * memref->strides[1]];
}

} // namespace

extern "C" SPARSEWAVE_BENCHMARK_EXPORT void
_mlir_ciface_loadCSRSpMVBenchmarkInputs(
    StridedMemRefType<int32_t, 1> *rowOffsets,
    StridedMemRefType<int32_t, 1> *columnIndices,
    StridedMemRefType<float, 1> *values, StridedMemRefType<float, 1> *vector,
    StridedMemRefType<float, 1> *expected) {
  SparseDimensions dimensions =
      loadCSRInputs(rowOffsets, columnIndices, values);

  initializeVector(vector, dimensions.columns);
  initializeExpected(expected, dimensions.rows);
  for (uint64_t row = 0; row < dimensions.rows; ++row) {
    float sum = 0.0f;
    for (int32_t position = rowOffsets->data[rowOffsets->offset + row];
         position < rowOffsets->data[rowOffsets->offset + row + 1]; ++position)
      sum += values->data[values->offset + position];
    expected->data[expected->offset + row] = sum;
  }
}

extern "C" SPARSEWAVE_BENCHMARK_EXPORT void
_mlir_ciface_loadCOOSpMVBenchmarkInputs(
    StridedMemRefType<int32_t, 1> *rowIndices,
    StridedMemRefType<int32_t, 1> *columnIndices,
    StridedMemRefType<float, 1> *values, StridedMemRefType<float, 1> *vector,
    StridedMemRefType<float, 1> *expected) {
  SparseDimensions dimensions =
      loadCOOInputs(rowIndices, columnIndices, values);

  initializeVector(vector, dimensions.columns);
  initializeExpected(expected, dimensions.rows);
  for (uint64_t position = 0; position < dimensions.nnz; ++position) {
    int32_t row = rowIndices->data[rowIndices->offset + position];
    int32_t column = columnIndices->data[columnIndices->offset + position];
    if (row < 0 || static_cast<uint64_t>(row) >= dimensions.rows ||
        column < 0 || static_cast<uint64_t>(column) >= dimensions.columns)
      fail("COO coordinate is out of bounds");
    expected->data[expected->offset + row] +=
        values->data[values->offset + position];
  }
}

extern "C" SPARSEWAVE_BENCHMARK_EXPORT void
_mlir_ciface_loadSpMMBenchmarkInputs(
    StridedMemRefType<int32_t, 1> *rowOffsets,
    StridedMemRefType<int32_t, 1> *columnIndices,
    StridedMemRefType<float, 1> *values, StridedMemRefType<float, 2> *rhs,
    StridedMemRefType<double, 2> *expected,
    StridedMemRefType<double, 2> *tolerances) {
  SparseDimensions dimensions =
      loadCSRInputs(rowOffsets, columnIndices, values);
  if (rhs->sizes[0] != static_cast<int64_t>(dimensions.columns) ||
      rhs->sizes[1] < 1 || rhs->strides[1] != 1 ||
      rhs->strides[0] != rhs->sizes[1])
    fail("RHS memref has an unexpected layout");
  if (expected->sizes[0] != static_cast<int64_t>(dimensions.rows) ||
      expected->sizes[1] != rhs->sizes[1] || expected->strides[1] != 1 ||
      expected->strides[0] != expected->sizes[1])
    fail("expected-output memref has an unexpected layout");

  if (tolerances->sizes[0] != expected->sizes[0] ||
      tolerances->sizes[1] != expected->sizes[1] ||
      tolerances->strides[1] != 1 ||
      tolerances->strides[0] != tolerances->sizes[1])
    fail("tolerance memref has an unexpected layout");

  for (uint64_t reduction = 0; reduction < dimensions.columns; ++reduction)
    for (int64_t column = 0; column < rhs->sizes[1]; ++column)
      element(rhs, reduction, column) =
          1.0f + static_cast<float>((reduction + column) % 7) * 0.125f;

  for (uint64_t row = 0; row < dimensions.rows; ++row) {
    for (int64_t column = 0; column < rhs->sizes[1]; ++column) {
      double sum = 0.0;
      double absoluteProducts = 0.0;
      for (int32_t position = rowOffsets->data[rowOffsets->offset + row];
           position < rowOffsets->data[rowOffsets->offset + row + 1];
           ++position) {
        int32_t reductionIndex =
            columnIndices->data[columnIndices->offset + position];
        double product =
            static_cast<double>(values->data[values->offset + position]) *
            element(rhs, reductionIndex, column);
        sum += product;
        absoluteProducts += std::abs(product);
      }
      element(expected, row, column) = sum;
      uint64_t terms = rowOffsets->data[rowOffsets->offset + row + 1] -
                       rowOffsets->data[rowOffsets->offset + row];
      element(tolerances, row, column) =
          sparsewave::benchmark::referenceTolerance(sum, absoluteProducts,
                                                    terms);
    }
  }
}

extern "C" SPARSEWAVE_BENCHMARK_EXPORT void
_mlir_ciface_loadBSRSpMMBenchmarkInputs(
    StridedMemRefType<int32_t, 1> *blockRowOffsets,
    StridedMemRefType<int32_t, 1> *blockColumnIndices,
    StridedMemRefType<float, 1> *blockValues, StridedMemRefType<float, 2> *rhs,
    StridedMemRefType<double, 2> *expected,
    StridedMemRefType<double, 2> *tolerances) {
  BSRDimensions dimensions =
      loadBSRInputs(blockRowOffsets, blockColumnIndices, blockValues);
  if (rhs->sizes[0] != static_cast<int64_t>(dimensions.columns) ||
      rhs->sizes[1] < 1 || rhs->strides[1] != 1 ||
      rhs->strides[0] != rhs->sizes[1])
    fail("RHS memref has an unexpected layout");
  if (expected->sizes[0] != static_cast<int64_t>(dimensions.rows) ||
      expected->sizes[1] != rhs->sizes[1] || expected->strides[1] != 1 ||
      expected->strides[0] != expected->sizes[1])
    fail("expected-output memref has an unexpected layout");

  if (tolerances->sizes[0] != expected->sizes[0] ||
      tolerances->sizes[1] != expected->sizes[1] ||
      tolerances->strides[1] != 1 ||
      tolerances->strides[0] != tolerances->sizes[1])
    fail("tolerance memref has an unexpected layout");

  for (uint64_t reduction = 0; reduction < dimensions.columns; ++reduction)
    for (int64_t column = 0; column < rhs->sizes[1]; ++column)
      element(rhs, reduction, column) =
          1.0f + static_cast<float>((reduction + column) % 7) * 0.125f;

  std::fill_n(expected->data + expected->offset,
              dimensions.rows * rhs->sizes[1], 0.0);
  std::fill_n(tolerances->data + tolerances->offset,
              dimensions.rows * rhs->sizes[1], 0.0);
  std::vector<uint64_t> rowTerms(dimensions.rows, 0);
  for (uint64_t position = 0; position < dimensions.originalNnz; ++position) {
    int32_t row = dimensions.inputRows[position];
    int32_t reduction = dimensions.inputColumns[position];
    if (row < 0 || static_cast<uint64_t>(row) >= dimensions.originalRows ||
        reduction < 0 ||
        static_cast<uint64_t>(reduction) >= dimensions.originalColumns)
      fail("original BSR input coordinate is out of bounds");
    ++rowTerms[row];
    for (int64_t column = 0; column < rhs->sizes[1]; ++column) {
      double product = static_cast<double>(dimensions.inputValues[position]) *
                       element(rhs, reduction, column);
      element(expected, row, column) += product;
      element(tolerances, row, column) += std::abs(product);
    }
  }
  for (uint64_t row = 0; row < dimensions.rows; ++row) {
    for (int64_t column = 0; column < rhs->sizes[1]; ++column) {
      element(tolerances, row, column) =
          sparsewave::benchmark::referenceTolerance(
              element(expected, row, column), element(tolerances, row, column),
              rowTerms[row]);
    }
  }
}

extern "C" SPARSEWAVE_BENCHMARK_EXPORT int64_t
_mlir_ciface_verifySpMMBenchmarkOutput(
    StridedMemRefType<float, 2> *actual, StridedMemRefType<double, 2> *expected,
    StridedMemRefType<double, 2> *tolerances) {
  if (actual->sizes[0] != expected->sizes[0] ||
      actual->sizes[1] != expected->sizes[1] ||
      tolerances->sizes[0] != expected->sizes[0] ||
      tolerances->sizes[1] != expected->sizes[1])
    fail("verification memref shapes differ");
  int64_t mismatches = 0;
  for (int64_t row = 0; row < actual->sizes[0]; ++row) {
    for (int64_t column = 0; column < actual->sizes[1]; ++column) {
      float value = element(actual, row, column);
      double reference = element(expected, row, column);
      double tolerance = element(tolerances, row, column);
      if (sparsewave::benchmark::referenceMatches(value, reference, tolerance))
        continue;
      if (mismatches++ < 8)
        std::cerr << std::setprecision(17) << "SpMM mismatch at (" << row
                  << ", " << column << "): expected=" << reference
                  << " actual=" << value << " tolerance=" << tolerance << '\n';
    }
  }
  return mismatches;
}
