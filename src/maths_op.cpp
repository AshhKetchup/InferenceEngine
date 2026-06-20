#include "maths_op.h"
#include <cmath>

using std::size_t;

float dot_product(const float *a, const float *b, size_t n) {
  double prod = 0.0;
  for (size_t i = 0; i < n; ++i) {
    prod += static_cast<double>(a[i]) * b[i];
  }
  return static_cast<float>(prod);
}

void matvec(float *out, const float *W, const float *x, size_t rows,
            size_t cols) {
  for (size_t r = 0; r < rows; ++r) {
    out[r] = dot_product(W + r * cols, x, cols);
  }
}

void rmsnorm(float *out, const float *x, const float *g, size_t n, float eps) {
  double sumsq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sumsq += static_cast<double>(x[i]) * x[i];
  }

  float mean_sq = static_cast<float>(sumsq / static_cast<double>(n));
  float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

  for (size_t i = 0; i < n; ++i) {
    out[i] = x[i] * inv_rms * g[i];
  }
}
