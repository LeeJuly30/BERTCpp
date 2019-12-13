#include<iostream>
#include<memory>
#include<fstream>
#include<cmath>
#include<random>
#include"../src/gelu.h"
#include"gtest/gtest.h"

using namespace lh;
using namespace std;

TEST(CommonTest, gelu){
    float input[5] = {-2, -1, 0, 1, 2};

    float expect[5];
    for (int i = 0; i < 5; ++i) {
        expect[i] = input[i] * 0.5 * (1.0 + erf(input[i] / sqrt(2.0)));
    }
    Gelu<float> l;
    float output[9];
    l.compute(5, input);
    EXPECT_NEAR(input[0], expect[0], 1e-5);
    EXPECT_NEAR(input[1], expect[1], 1e-5);
    EXPECT_NEAR(input[2], expect[2], 1e-5);
    EXPECT_NEAR(input[3], expect[3], 1e-5);
    EXPECT_NEAR(input[4], expect[4], 1e-5);
}