#include<iostream>
#include<memory>
#include<fstream>
#include<random>
#include"../src/embedding.h"
#include"gtest/gtest.h"

using namespace lh;
using namespace std;

TEST(CommonTest, embedding){
    float embedding_table[12] = {
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            9, 10, 11,
    };
    Graph<float> graph;
    graph["weight"] = make_pair(vector<size_t>({4, 3}), embedding_table);
    vector<string> names = {"weight"};
    Embedding<float> l(names, graph);
    uint64_t input_ids[4] = {2, 2, 1, 3};
    float out[12];
    l.compute(2, 2, input_ids, out);
    EXPECT_FLOAT_EQ(out[0], 6);
    EXPECT_FLOAT_EQ(out[1], 7);
    EXPECT_FLOAT_EQ(out[2], 8);
    EXPECT_FLOAT_EQ(out[3], 6);
    EXPECT_FLOAT_EQ(out[4], 7);
    EXPECT_FLOAT_EQ(out[5], 8);
    EXPECT_FLOAT_EQ(out[6], 3);
    EXPECT_FLOAT_EQ(out[7], 4);
    EXPECT_FLOAT_EQ(out[8], 5);
    EXPECT_FLOAT_EQ(out[9], 9);
    EXPECT_FLOAT_EQ(out[10], 10);
    EXPECT_FLOAT_EQ(out[11], 11);
}