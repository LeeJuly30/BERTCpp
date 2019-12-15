#include"../src/bertembedding.h"
#include"gtest/gtest.h"
#include <cmath>

using namespace std;
using namespace lh;

TEST(CommonTest, bertembedding){
    size_t vocab_size = 5;
    size_t type_vocab_size = 2;
    size_t hidden_size = 3;
    size_t seq_length = 4;
    size_t batch_size = 2;

    float word_embeddings[15] = {
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            9, 10, 11,
            12, 13, 14,
    };
    float token_type_embeddings[6] = {
            -2, -1, 0,
            -5, -4, -3,
    };
    float position_embeddings[12] = {
            1, 1, 1,
            -1, -1, -1,
            2, 2, 2,
            -2, -2, -2,
    };

    float beta[3] = {1, 0, -1};
    float gamma[3] = {1, 1, 1};

    Graph<float> graph = {
            {"bert/embeddings/word_embeddings",         make_pair(vector<size_t>({5, 3}), word_embeddings)},
            {"bert/embeddings/position_embeddings",     make_pair(vector<size_t>({4, 3}), position_embeddings)},
            {"bert/embeddings/token_type_embeddings",   make_pair(vector<size_t>({2, 3}), token_type_embeddings)},
            {"bert/embeddings/LayerNorm/gamma",         make_pair(vector<size_t>({3}), gamma)},
            {"bert/embeddings/LayerNorm/beta",          make_pair(vector<size_t>({3}), beta)},
    };

    vector<string> names = {
        "bert/embeddings/word_embeddings",
        "bert/embeddings/position_embeddings",
        "bert/embeddings/token_type_embeddings", 
        "bert/embeddings/LayerNorm/gamma",
        "bert/embeddings/LayerNorm/beta", 
    };

    BertEmbedding<float> l(names, graph, 5, 10, 3);
    uint64_t input_ids[8] = {3, 0, 1, 4,
                        2, 0, 2, 1};
    uint64_t segment_ids[8] = {1, 1, 0, 1,
                          0, 1, 0, 0};

    uint64_t position_ids[8];
    for(int i=0; i<batch_size;i++){
        for(int j=0;j<seq_length;j++){
            position_ids[i*seq_length + j] = j;
        }
    }

    float out[24];

    l.compute(batch_size, seq_length, input_ids, position_ids, segment_ids, out);

    EXPECT_NEAR(out[0], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[1], 0);
    EXPECT_NEAR(out[2], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[3], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[4], 0);
    EXPECT_NEAR(out[5], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[6], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[7], 0);
    EXPECT_NEAR(out[8], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[9], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[10], 0);
    EXPECT_NEAR(out[11], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[12], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[13], 0);
    EXPECT_NEAR(out[14], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[15], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[16], 0);
    EXPECT_NEAR(out[17], std::sqrt(1.5) - 1, 1e-6);
    EXPECT_NEAR(out[18], 1 - std::sqrt(1.5), 2e-6);
    EXPECT_FLOAT_EQ(out[19], 0);
    EXPECT_NEAR(out[20], std::sqrt(1.5) - 1, 2e-6);
    EXPECT_NEAR(out[21], 1 - std::sqrt(1.5), 1e-6);
    EXPECT_FLOAT_EQ(out[22], 0);
    EXPECT_NEAR(out[23], std::sqrt(1.5) - 1, 1e-6);
}