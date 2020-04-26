#include<iostream>
#include<memory>
#include<fstream>
#include<random>
#include"../src/layernorm.h"
#include"gtest/gtest.h"

using namespace lh;
using namespace std;

TEST(CommonTest, layernorm){
    float beta[3] = {-1, 0, 1};
    float gamma[3] = {1, 2, 3};
    Graph graph;
    graph["beta"] = new tensor(static_cast<void*>(beta), vector<size_t>({3}));
    graph["gamma"] = new tensor(static_cast<void*>(gamma), vector<size_t>({3}));
    vector<string> names = {"gamma", "beta"};
    Layernorm<float> l(names, graph, 5, 128);
    float input[6] = {9, 10, 11, 5, 4, 3};
    float output[6];
    l.compute(2, 1, input, output);
    EXPECT_NEAR(output[0], -2.224744871391589, 1e-5);
    EXPECT_FLOAT_EQ(output[1], 0);
    EXPECT_NEAR(output[2], 4.674234614174766, 1e-5);
    EXPECT_NEAR(output[3], 0.22474487139158894, 1e-5);
    EXPECT_FLOAT_EQ(output[4], 0);
    EXPECT_NEAR(output[5], -2.674234614174767, 1e-5);
}