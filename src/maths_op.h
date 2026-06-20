#pragma once
#include <cstddef>

float dot_product(const float *a, const float *b, std::size_t n);

void matvec(float *out, const float *W, const float *x, std::size_t rows,
            std::size_t cols);

void rmsnorm(float *out, const float *x, const float *g, std::size_t n,
             float eps);
