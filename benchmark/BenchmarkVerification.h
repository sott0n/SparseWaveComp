//===- BenchmarkVerification.h - Floating-point validation -------*- C++
//-*-===//
//
// Part of the SparseWave project.
//
//===----------------------------------------------------------------------===//

#ifndef SPARSEWAVE_BENCHMARK_VERIFICATION_H
#define SPARSEWAVE_BENCHMARK_VERIFICATION_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace sparsewave::benchmark {

// Bound f32 product rounding and accumulation in any order by
// gamma_(n+1) * sum(abs(a_i * b_i)), where gamma_k = k*u/(1-k*u).
// Compute the reference and absolute-product sum in f64 from f32 inputs.
// Keep the existing tolerance floor for well-conditioned outputs. This model
// assumes finite arithmetic without overflow or flush-to-zero underflow.
inline double referenceTolerance(double expected, double absoluteProducts,
                                 uint64_t terms) {
  double ku = (static_cast<double>(terms) + 1) *
              (std::numeric_limits<float>::epsilon() / 2);
  if (ku >= 1 || !std::isfinite(expected) || !std::isfinite(absoluteProducts))
    return std::numeric_limits<double>::quiet_NaN();
  return std::max(1.0e-4 * std::max(1.0, std::abs(expected)),
                  ku / (1 - ku) * absoluteProducts);
}

inline bool referenceMatches(float actual, double expected, double tolerance) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::isfinite(tolerance) && tolerance >= 0 &&
         std::abs(static_cast<double>(actual) - expected) <= tolerance;
}

} // namespace sparsewave::benchmark

#endif // SPARSEWAVE_BENCHMARK_VERIFICATION_H
