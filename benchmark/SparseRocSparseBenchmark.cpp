//===- SparseRocSparseBenchmark.cpp - rocSPARSE benchmark runner -*- C++
//-*-===//
//
// Part of the SparseWave project.
//
//===----------------------------------------------------------------------===//

#include "hip/hip_runtime_api.h"
#include "rocsparse/rocsparse.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr char magic[] = "SWCSR001";

void checkHip(hipError_t status, const char *operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
}

void checkRocSparse(rocsparse_status status, const char *operation) {
  if (status != rocsparse_status_success)
    throw std::runtime_error(std::string(operation) +
                             " failed with rocSPARSE status " +
                             std::to_string(status));
}

template <typename T> T *allocateDevice(size_t count) {
  T *pointer = nullptr;
  if (count != 0)
    checkHip(hipMalloc(reinterpret_cast<void **>(&pointer), count * sizeof(T)),
             "hipMalloc");
  return pointer;
}

template <typename T>
void copyToDevice(T *destination, const std::vector<T> &source) {
  if (!source.empty())
    checkHip(hipMemcpy(destination, source.data(), source.size() * sizeof(T),
                       hipMemcpyHostToDevice),
             "hipMemcpy host to device");
}

struct CSRMatrix {
  uint64_t rows;
  uint64_t columns;
  uint64_t nnz;
  std::vector<int32_t> rowOffsets;
  std::vector<int32_t> columnIndices;
  std::vector<float> values;
};

uint64_t readU64(std::ifstream &stream, const char *name) {
  uint64_t value = 0;
  stream.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!stream)
    throw std::runtime_error(std::string("could not read ") + name);
  return value;
}

template <typename T>
std::vector<T> readArray(std::ifstream &stream, size_t count,
                         const char *name) {
  std::vector<T> values(count);
  stream.read(reinterpret_cast<char *>(values.data()), count * sizeof(T));
  if (!stream)
    throw std::runtime_error(std::string("could not read ") + name);
  return values;
}

CSRMatrix readCSR(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("could not open " + path);

  char fileMagic[sizeof(magic) - 1];
  stream.read(fileMagic, sizeof(fileMagic));
  if (!stream || !std::equal(std::begin(fileMagic), std::end(fileMagic),
                             std::begin(magic)))
    throw std::runtime_error("invalid CSR binary header");

  CSRMatrix matrix;
  matrix.rows = readU64(stream, "row count");
  matrix.columns = readU64(stream, "column count");
  matrix.nnz = readU64(stream, "NNZ count");
  if (matrix.rows > std::numeric_limits<int32_t>::max() ||
      matrix.columns > std::numeric_limits<int32_t>::max() ||
      matrix.nnz > std::numeric_limits<int32_t>::max())
    throw std::runtime_error("CSR dimensions exceed the i32 benchmark limits");

  matrix.rowOffsets =
      readArray<int32_t>(stream, matrix.rows + 1, "row offsets");
  matrix.columnIndices =
      readArray<int32_t>(stream, matrix.nnz, "column indices");
  matrix.values = readArray<float>(stream, matrix.nnz, "values");
  return matrix;
}

struct Arguments {
  std::string operation;
  std::string csrPath;
  int warmup = 0;
  int iterations = 0;
  int rhsColumns = 1;
};

int parseNonnegative(const std::string &value, const char *name) {
  size_t consumed = 0;
  long parsed = std::stol(value, &consumed);
  if (consumed != value.size() || parsed < 0 ||
      parsed > std::numeric_limits<int>::max())
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  return static_cast<int>(parsed);
}

Arguments parseArguments(int argc, char **argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    std::string option = argv[index];
    if (index + 1 == argc)
      throw std::runtime_error("missing value for " + option);
    std::string value = argv[++index];
    if (option == "--operation")
      arguments.operation = value;
    else if (option == "--csr")
      arguments.csrPath = value;
    else if (option == "--warmup")
      arguments.warmup = parseNonnegative(value, "warmup count");
    else if (option == "--iterations")
      arguments.iterations = parseNonnegative(value, "iteration count");
    else if (option == "--rhs-columns")
      arguments.rhsColumns = parseNonnegative(value, "RHS column count");
    else
      throw std::runtime_error("unknown option: " + option);
  }
  if (arguments.operation != "spmv" && arguments.operation != "spmm")
    throw std::runtime_error("--operation must be spmv or spmm");
  if (arguments.csrPath.empty())
    throw std::runtime_error("--csr is required");
  if (arguments.iterations == 0)
    throw std::runtime_error("--iterations must be positive");
  if (arguments.rhsColumns == 0)
    throw std::runtime_error("--rhs-columns must be positive");
  return arguments;
}

std::vector<float> createSpMVExpected(const CSRMatrix &matrix) {
  std::vector<float> expected(matrix.rows, 0.0f);
  for (uint64_t row = 0; row < matrix.rows; ++row)
    for (int32_t position = matrix.rowOffsets[row];
         position < matrix.rowOffsets[row + 1]; ++position)
      expected[row] += matrix.values[position];
  return expected;
}

std::vector<float> createSpMMRhs(const CSRMatrix &matrix, int rhsColumns) {
  std::vector<float> rhs(matrix.columns * rhsColumns);
  for (uint64_t reduction = 0; reduction < matrix.columns; ++reduction)
    for (int column = 0; column < rhsColumns; ++column)
      rhs[reduction * rhsColumns + column] =
          1.0f + static_cast<float>((reduction + column) % 7) * 0.125f;
  return rhs;
}

std::vector<float> createSpMMExpected(const CSRMatrix &matrix,
                                      const std::vector<float> &rhs,
                                      int rhsColumns) {
  std::vector<float> expected(matrix.rows * rhsColumns, 0.0f);
  for (uint64_t row = 0; row < matrix.rows; ++row)
    for (int column = 0; column < rhsColumns; ++column)
      for (int32_t position = matrix.rowOffsets[row];
           position < matrix.rowOffsets[row + 1]; ++position)
        expected[row * rhsColumns + column] +=
            matrix.values[position] *
            rhs[matrix.columnIndices[position] * rhsColumns + column];
  return expected;
}

int64_t countMismatches(const std::vector<float> &actual,
                        const std::vector<float> &expected) {
  int64_t mismatches = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    float tolerance = 1.0e-4f * std::max(1.0f, std::abs(expected[index]));
    if (!std::isfinite(actual[index]) ||
        std::abs(actual[index] - expected[index]) > tolerance)
      ++mismatches;
  }
  return mismatches;
}

template <typename Compute>
std::vector<float> runIterations(rocsparse_handle handle, int count,
                                 Compute compute) {
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t end = nullptr;
  checkHip(hipStreamCreate(&stream), "hipStreamCreate");
  checkHip(hipEventCreate(&start), "hipEventCreate start");
  checkHip(hipEventCreate(&end), "hipEventCreate end");
  checkRocSparse(rocsparse_set_stream(handle, stream), "rocsparse_set_stream");
  std::vector<float> timings(count);
  for (float &timing : timings) {
    checkHip(hipEventRecord(start, stream), "hipEventRecord start");
    compute();
    checkHip(hipEventRecord(end, stream), "hipEventRecord end");
    checkHip(hipEventSynchronize(end), "hipEventSynchronize");
    checkHip(hipEventElapsedTime(&timing, start, end), "hipEventElapsedTime");
  }
  checkHip(hipEventDestroy(start), "hipEventDestroy start");
  checkHip(hipEventDestroy(end), "hipEventDestroy end");
  checkHip(hipStreamDestroy(stream), "hipStreamDestroy");
  return timings;
}

struct DeviceCSR {
  int32_t *rowOffsets = nullptr;
  int32_t *columnIndices = nullptr;
  float *values = nullptr;
  rocsparse_spmat_descr descriptor = nullptr;
};

DeviceCSR createDeviceCSR(const CSRMatrix &matrix) {
  DeviceCSR device;
  device.rowOffsets = allocateDevice<int32_t>(matrix.rowOffsets.size());
  device.columnIndices = allocateDevice<int32_t>(matrix.columnIndices.size());
  device.values = allocateDevice<float>(matrix.values.size());
  copyToDevice(device.rowOffsets, matrix.rowOffsets);
  copyToDevice(device.columnIndices, matrix.columnIndices);
  copyToDevice(device.values, matrix.values);
  checkRocSparse(rocsparse_create_csr_descr(
                     &device.descriptor, matrix.rows, matrix.columns,
                     matrix.nnz, device.rowOffsets, device.columnIndices,
                     device.values, rocsparse_indextype_i32,
                     rocsparse_indextype_i32, rocsparse_index_base_zero,
                     rocsparse_datatype_f32_r),
                 "rocsparse_create_csr_descr");
  return device;
}

void destroyDeviceCSR(DeviceCSR &device) {
  checkRocSparse(rocsparse_destroy_spmat_descr(device.descriptor),
                 "rocsparse_destroy_spmat_descr");
  checkHip(hipFree(device.rowOffsets), "hipFree row offsets");
  checkHip(hipFree(device.columnIndices), "hipFree column indices");
  checkHip(hipFree(device.values), "hipFree values");
}

template <typename Preprocess>
double runPreprocess(hipStream_t stream, Preprocess preprocess) {
  auto start = std::chrono::steady_clock::now();
  preprocess();
  checkHip(hipStreamSynchronize(stream), "preprocess synchronization");
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - start).count();
}

struct BenchmarkResult {
  double preprocessMicroseconds;
  std::vector<float> timingMilliseconds;
  int64_t mismatches;
};

BenchmarkResult runSpMV(rocsparse_handle handle, hipStream_t setupStream,
                        const CSRMatrix &matrix, const DeviceCSR &deviceMatrix,
                        int dispatches) {
  std::vector<float> vector(matrix.columns, 1.0f);
  std::vector<float> expected = createSpMVExpected(matrix);
  std::vector<float> output(matrix.rows, 0.0f);
  float *deviceVector = allocateDevice<float>(vector.size());
  float *deviceOutput = allocateDevice<float>(output.size());
  copyToDevice(deviceVector, vector);
  copyToDevice(deviceOutput, output);

  rocsparse_dnvec_descr vectorDescriptor = nullptr;
  rocsparse_dnvec_descr outputDescriptor = nullptr;
  checkRocSparse(rocsparse_create_dnvec_descr(&vectorDescriptor, matrix.columns,
                                              deviceVector,
                                              rocsparse_datatype_f32_r),
                 "rocsparse_create_dnvec_descr vector");
  checkRocSparse(rocsparse_create_dnvec_descr(&outputDescriptor, matrix.rows,
                                              deviceOutput,
                                              rocsparse_datatype_f32_r),
                 "rocsparse_create_dnvec_descr output");

  float alpha = 1.0f;
  float beta = 0.0f;
  size_t bufferSize = 0;
  auto invoke = [&](rocsparse_spmv_stage stage, void *buffer) {
    checkRocSparse(
        rocsparse_spmv(handle, rocsparse_operation_none, &alpha,
                       deviceMatrix.descriptor, vectorDescriptor, &beta,
                       outputDescriptor, rocsparse_datatype_f32_r,
                       rocsparse_spmv_alg_default, stage, &bufferSize, buffer),
        "rocsparse_spmv");
  };
  invoke(rocsparse_spmv_stage_buffer_size, nullptr);
  void *buffer = allocateDevice<std::byte>(bufferSize);
  double preprocessMicroseconds = runPreprocess(
      setupStream, [&] { invoke(rocsparse_spmv_stage_preprocess, buffer); });
  std::vector<float> timings = runIterations(handle, dispatches, [&] {
    invoke(rocsparse_spmv_stage_compute, buffer);
  });
  checkHip(hipMemcpy(output.data(), deviceOutput, output.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "hipMemcpy SpMV output");

  int64_t mismatches = countMismatches(output, expected);
  checkHip(hipFree(buffer), "hipFree SpMV buffer");
  checkRocSparse(rocsparse_destroy_dnvec_descr(vectorDescriptor),
                 "rocsparse_destroy_dnvec_descr vector");
  checkRocSparse(rocsparse_destroy_dnvec_descr(outputDescriptor),
                 "rocsparse_destroy_dnvec_descr output");
  checkHip(hipFree(deviceVector), "hipFree vector");
  checkHip(hipFree(deviceOutput), "hipFree output");
  return {preprocessMicroseconds, std::move(timings), mismatches};
}

BenchmarkResult runSpMM(rocsparse_handle handle, hipStream_t setupStream,
                        const CSRMatrix &matrix, const DeviceCSR &deviceMatrix,
                        int rhsColumns, int dispatches) {
  std::vector<float> rhs = createSpMMRhs(matrix, rhsColumns);
  std::vector<float> expected = createSpMMExpected(matrix, rhs, rhsColumns);
  std::vector<float> output(matrix.rows * rhsColumns, 0.0f);
  float *deviceRhs = allocateDevice<float>(rhs.size());
  float *deviceOutput = allocateDevice<float>(output.size());
  copyToDevice(deviceRhs, rhs);
  copyToDevice(deviceOutput, output);

  rocsparse_dnmat_descr rhsDescriptor = nullptr;
  rocsparse_dnmat_descr outputDescriptor = nullptr;
  checkRocSparse(rocsparse_create_dnmat_descr(
                     &rhsDescriptor, matrix.columns, rhsColumns, rhsColumns,
                     deviceRhs, rocsparse_datatype_f32_r, rocsparse_order_row),
                 "rocsparse_create_dnmat_descr RHS");
  checkRocSparse(rocsparse_create_dnmat_descr(
                     &outputDescriptor, matrix.rows, rhsColumns, rhsColumns,
                     deviceOutput, rocsparse_datatype_f32_r,
                     rocsparse_order_row),
                 "rocsparse_create_dnmat_descr output");

  float alpha = 1.0f;
  float beta = 0.0f;
  size_t bufferSize = 0;
  auto invoke = [&](rocsparse_spmm_stage stage, void *buffer) {
    checkRocSparse(rocsparse_spmm(handle, rocsparse_operation_none,
                                  rocsparse_operation_none, &alpha,
                                  deviceMatrix.descriptor, rhsDescriptor, &beta,
                                  outputDescriptor, rocsparse_datatype_f32_r,
                                  rocsparse_spmm_alg_default, stage,
                                  &bufferSize, buffer),
                   "rocsparse_spmm");
  };
  invoke(rocsparse_spmm_stage_buffer_size, nullptr);
  void *buffer = allocateDevice<std::byte>(bufferSize);
  double preprocessMicroseconds = runPreprocess(
      setupStream, [&] { invoke(rocsparse_spmm_stage_preprocess, buffer); });
  std::vector<float> timings = runIterations(handle, dispatches, [&] {
    invoke(rocsparse_spmm_stage_compute, buffer);
  });
  checkHip(hipMemcpy(output.data(), deviceOutput, output.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "hipMemcpy SpMM output");

  int64_t mismatches = countMismatches(output, expected);
  checkHip(hipFree(buffer), "hipFree SpMM buffer");
  checkRocSparse(rocsparse_destroy_dnmat_descr(rhsDescriptor),
                 "rocsparse_destroy_dnmat_descr RHS");
  checkRocSparse(rocsparse_destroy_dnmat_descr(outputDescriptor),
                 "rocsparse_destroy_dnmat_descr output");
  checkHip(hipFree(deviceRhs), "hipFree RHS");
  checkHip(hipFree(deviceOutput), "hipFree output");
  return {preprocessMicroseconds, std::move(timings), mismatches};
}

} // namespace

int main(int argc, char **argv) {
  try {
    Arguments arguments = parseArguments(argc, argv);
    CSRMatrix matrix = readCSR(arguments.csrPath);

    rocsparse_handle handle = nullptr;
    checkRocSparse(rocsparse_create_handle(&handle), "rocsparse_create_handle");
    hipStream_t setupStream = nullptr;
    checkHip(hipStreamCreate(&setupStream), "hipStreamCreate setup");
    checkRocSparse(rocsparse_set_stream(handle, setupStream),
                   "rocsparse_set_stream setup");
    DeviceCSR deviceMatrix = createDeviceCSR(matrix);

    int version = 0;
    char revision[64] = {};
    checkRocSparse(rocsparse_get_version(handle, &version),
                   "rocsparse_get_version");
    checkRocSparse(rocsparse_get_git_rev(handle, revision),
                   "rocsparse_get_git_rev");

    int dispatches = arguments.warmup + arguments.iterations;
    BenchmarkResult result =
        arguments.operation == "spmv"
            ? runSpMV(handle, setupStream, matrix, deviceMatrix, dispatches)
            : runSpMM(handle, setupStream, matrix, deviceMatrix,
                      arguments.rhsColumns, dispatches);

    checkRocSparse(rocsparse_set_stream(handle, setupStream),
                   "rocsparse_set_stream cleanup");
    destroyDeviceCSR(deviceMatrix);
    checkHip(hipStreamDestroy(setupStream), "hipStreamDestroy setup");
    checkRocSparse(rocsparse_destroy_handle(handle),
                   "rocsparse_destroy_handle");

    std::cout << "rocsparse_version=" << version << '\n'
              << "rocsparse_git_rev=" << revision << '\n'
              << "preprocess_us=" << result.preprocessMicroseconds << '\n'
              << "timings_us=";
    for (size_t index = 0; index < result.timingMilliseconds.size(); ++index) {
      if (index != 0)
        std::cout << ',';
      std::cout << result.timingMilliseconds[index] * 1000.0f;
    }
    std::cout << '\n' << '[' << result.mismatches << "]\n";
    return result.mismatches == 0 ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "rocSPARSE benchmark error: " << error.what() << '\n';
    return 1;
  }
}
