#include "maths_op.h"
#include <assert.h>
#include <cmath>
#include <iostream>

using namespace std;
float dot_product(const float *a, const float *b, std::size_t n) {
  double prod = 0.0;

  for (size_t i = 0; i < n; i++) {
    prod += static_cast<double>(a[i]) * b[i];
  }

  return static_cast<float>(prod);
}

void matvec(float *out, const float *W, const float *x, size_t rows,
            size_t cols) {
  for (size_t r = 0; r < rows; r++) {
    out[r] = dot_product(W + r * cols, x, cols);
  }
}

//==================== test =============================//

static void test_dot_product() {
  float a1[] = {1, 2, 3}, b1[] = {4, 5, 6};
  assert(std::fabs(dot_product(a1, b1, 3) - 32.0f) < 1e-5f);

  float a2[] = {0.5f, -1.0f}, b2[] = {2.0f, 3.0f};
  assert(std::fabs(dot_product(a2, b2, 2) - (-2.0f)) < 1e-5f);

  float a3[] = {1, 0}, b3[] = {0, 1};
  assert(dot_product(a3, b3, 2) == 0.0f);

  assert(dot_product(nullptr, nullptr, 0) == 0.0f);
  float a5[] = {3, 4};
  assert(dot_product(a5, a5, 2) == 25.0f);
  std::cout << "dot_product tests OK\n";
}

static void test_matvec() {
  // Example 1
  float W1[] = {1, 2, 3, 4};
  float x1[] = {5, 6};
  float out1[2];
  matvec(out1, W1, x1, 2, 2);
  assert(out1[0] == 17.0f && out1[1] == 39.0f);

  // Example 2 (identity)
  float W2[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float x2[] = {7, 8, 9};

  // Example 2 (identity)
  float out2[3];
  matvec(out2, W2, x2, 3, 3);
  assert(out2[0] == 7 && out2[1] == 8 && out2[2] == 9);

  // Example 3 (non-square)
  float W3[] = {1, 2, 3, 4, 5, 6};
  float x3[] = {10, 20};
  float out3[3];
  matvec(out3, W3, x3, 3, 2);
  assert(out3[0] == 50 && out3[1] == 110 && out3[2] == 170);
  // Example 4 (1 row) float W4[] = {1,2,3,4}; float x4[] = {1,1,1,1}; float
  // out4[1]; matvec(out4, W4, x4, 1, 4); assert(out4[0] == 10.0f);
  printf("matvec tests OK\n");
}
