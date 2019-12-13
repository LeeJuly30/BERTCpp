#include<iostream>
#include<memory>
#include<fstream>
#include<random>
#include"../src/dense.h"
#include"gtest/gtest.h"

using namespace lh;
using namespace std;

TEST(CommonTest, dense){
    float kernel[6] = {1, 2, 3, 4, 5, 6};
    float bias[3] = {-3, -2, -1};
    Graph<float> graph;
    graph["weight"] = make_pair(vector<size_t>({2, 3}), kernel);
    graph["bias"] = make_pair(vector<size_t>({3}), bias);
    vector<string> names = {"weight", "bias"};
    Dense<float> l(names, graph);
    float input[6] = {1, 2, 3, 4, 5, 6};
    float output[9];
    l.compute(2, 1, input, output);
    EXPECT_FLOAT_EQ(output[0], 9 - 3);
    EXPECT_FLOAT_EQ(output[1], 12 - 2);
    EXPECT_FLOAT_EQ(output[2], 15 - 1);
    EXPECT_FLOAT_EQ(output[3], 19 - 3);
    EXPECT_FLOAT_EQ(output[4], 26 - 2);
    EXPECT_FLOAT_EQ(output[5], 33 - 1);
}
