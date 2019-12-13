#include<iostream>
#include<memory>
#include<fstream>
#include<random>
#include"../src/softmax.h"
#include"gtest/gtest.h"

using namespace lh;
using namespace std;

TEST(CommonTest, softmax){
    float input[9] = {1, 2, 3, 6, 6, 6, 8, 8, -10000.0f};
    Softmax<float> l;
    float* output = input;
    l.compute(3*1, 3, input, output);
    EXPECT_FLOAT_EQ(output[0], 0.090030573);
    EXPECT_FLOAT_EQ(output[1], 0.244728478);
    EXPECT_FLOAT_EQ(output[2], 0.66524094);
    EXPECT_FLOAT_EQ(output[3], 0.33333334);
    EXPECT_FLOAT_EQ(output[4], 0.33333334);
    EXPECT_FLOAT_EQ(output[5], 0.33333334);
    EXPECT_NEAR(output[6], 0.5, 1e-4);
    EXPECT_NEAR(output[7], 0.5, 1e-4);
    EXPECT_NEAR(output[8], 0, 1e-4);
}