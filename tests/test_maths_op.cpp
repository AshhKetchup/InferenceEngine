#include "../src/maths_op.h"
#include <cassert>
#include <cmath>
#include <cstdio>

static bool approx(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) < tol;
}

static void test_dot_product() {
    float a1[] = {1, 2, 3}, b1[] = {4, 5, 6};
    assert(approx(dot_product(a1, b1, 3), 32.0f));

    float a2[] = {0.5f, -1.0f}, b2[] = {2.0f, 3.0f};
    assert(approx(dot_product(a2, b2, 2), -2.0f));

    float a3[] = {1, 0}, b3[] = {0, 1};
    assert(dot_product(a3, b3, 2) == 0.0f);

    assert(dot_product(nullptr, nullptr, 0) == 0.0f);

    float a5[] = {3, 4};
    assert(dot_product(a5, a5, 2) == 25.0f);

    printf("dot_product tests OK\n");
}

static void test_matvec() {
    float W1[] = {1, 2, 3, 4};
    float x1[] = {5, 6};
    float out1[2];
    matvec(out1, W1, x1, 2, 2);
    assert(out1[0] == 17.0f && out1[1] == 39.0f);

    float W2[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float x2[] = {7, 8, 9};
    float out2[3];
    matvec(out2, W2, x2, 3, 3);
    assert(out2[0] == 7 && out2[1] == 8 && out2[2] == 9);

    float W3[] = {1, 2, 3, 4, 5, 6};
    float x3[] = {10, 20};
    float out3[3];
    matvec(out3, W3, x3, 3, 2);
    assert(out3[0] == 50 && out3[1] == 110 && out3[2] == 170);

    float W4[] = {1, 2, 3, 4};
    float x4[] = {1, 1, 1, 1};
    float out4[1];
    matvec(out4, W4, x4, 1, 4);
    assert(out4[0] == 10.0f);

    printf("matvec tests OK\n");
}

static void test_rmsnorm() {
    float x1[] = {1, 2, 2};
    float g1[] = {1, 1, 1};
    float out1[3];
    rmsnorm(out1, x1, g1, 3, 0.0f);
    assert(approx(out1[0], 0.57735f));
    assert(approx(out1[1], 1.15470f));
    assert(approx(out1[2], 1.15470f));

    float x2[] = {1, 2, 2};
    float g2[] = {2, 0, 1};
    float out2[3];
    rmsnorm(out2, x2, g2, 3, 0.0f);
    assert(approx(out2[0], 1.15470f));
    assert(approx(out2[1], 0.0f));
    assert(approx(out2[2], 1.15470f));

    float x3[] = {0, 0, 0};
    float g3[] = {1, 1, 1};
    float out3[3];
    rmsnorm(out3, x3, g3, 3, 1e-6f);
    assert(out3[0] == 0.0f && out3[1] == 0.0f && out3[2] == 0.0f);

    float x4[] = {3, 0, 0};
    float g4[] = {1, 1, 1};
    float out4[3];
    rmsnorm(out4, x4, g4, 3, 0.0f);
    assert(approx(out4[0], 1.73205f));
    assert(approx(out4[1], 0.0f));
    assert(approx(out4[2], 0.0f));

    float x5[] = {1, 2, 2};
    float g5[] = {1, 1, 1};
    rmsnorm(x5, x5, g5, 3, 0.0f);
    assert(approx(x5[0], 0.57735f));
    assert(approx(x5[1], 1.15470f));
    assert(approx(x5[2], 1.15470f));

    printf("rmsnorm tests OK\n");
}

void test_maths_op() {
    test_dot_product();
    test_matvec();
    test_rmsnorm();
}
