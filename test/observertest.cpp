#include<iostream>
#include<memory>
#include<fstream>
#include<random>
#include"../src/dense.h"
#include"gtest/gtest.h"

using namespace lh;
using namespace std;

TEST(CommonTest, observer_dense){
    float kernel[6] = {1, 2, 3, 4, 5, 6};
    float bias[3] = {-3, -2, -1};
    Graph<float> graph;
    graph["weight"] = make_pair(vector<size_t>({2, 3}), kernel);
    graph["bias"] = make_pair(vector<size_t>({3}), bias);
    vector<string> names = {"weight", "bias"};
    Dense<float> l(names, graph);
    float averger = 0.5f;
    l.addobserver(averger);
    float input[6] = {1, 2, 3, 4, 5, 6};
    float output[9];
    for(int i = 0; i < 2; i++){
        l.calibration(2, 1, input, output);
        input[0] = 10.0f;
    }
    EXPECT_FLOAT_EQ(l.weight_observer->min_val_, 13.5);
    EXPECT_FLOAT_EQ(l.weight_observer->max_val_, 37.5);
}