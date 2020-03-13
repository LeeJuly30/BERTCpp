#include"../src/batchgemm.h"
#include"gtest/gtest.h"

using namespace std;
using namespace lh;

TEST(CommonTest, qk){
    size_t seq_length = 2;
    size_t num_attention_heads = 4;
    size_t size_per_head = 3;
    float query[48] = {
            0, 1, 2,
            -1, -2, 0,
            2, -1, 0,
            0, 1, -2,

            2, 1, -2,
            0, 2, -1,
            -1, -1, 0,
            2, 1, -1,

            0, 1, 2,
            -1, -2, 0,
            2, -1, 0,
            0, 1, -2,

            2, 1, -2,
            0, 2, -1,
            -1, -1, 0,
            2, 1, -1,
    };
    float key[48] = {
            3, 0, -1,
            2, 3, -3,
            0, -1, -1,
            1, 1, 1,

            -2, 3, 0,
            -1, 2, 2,
            2, 0, 0,
            0, 0, 0,

            3, 0, -1,
            2, 3, -3,
            0, -1, -1,
            1, 1, 1,

            -2, 3, 0,
            -1, 2, 2,
            2, 0, 0,
            0, 0, 0,
    };
    float out[32];
    const float* q_array[8];
    const float* k_array[8];
    float* pointer_qk_array[8];
    attn_qk(2, num_attention_heads, seq_length, size_per_head, query, key, out, q_array, k_array, pointer_qk_array);
    EXPECT_FLOAT_EQ(out[0], -2);
    EXPECT_FLOAT_EQ(out[1], 3);
    EXPECT_FLOAT_EQ(out[2], -8);
    EXPECT_FLOAT_EQ(out[3], -3);
    EXPECT_FLOAT_EQ(out[4], 1);
    EXPECT_FLOAT_EQ(out[5], 4);
    EXPECT_FLOAT_EQ(out[6], -1);
    EXPECT_FLOAT_EQ(out[7], 0);
    EXPECT_FLOAT_EQ(out[8], 8);
    EXPECT_FLOAT_EQ(out[9], -1);
    EXPECT_FLOAT_EQ(out[10], 9);
    EXPECT_FLOAT_EQ(out[11], 2);
    EXPECT_FLOAT_EQ(out[12], 1);
    EXPECT_FLOAT_EQ(out[13], -2);
    EXPECT_FLOAT_EQ(out[14], 2);
    EXPECT_FLOAT_EQ(out[15], 0);
    EXPECT_FLOAT_EQ(out[16], -2);
    EXPECT_FLOAT_EQ(out[17], 3);
    EXPECT_FLOAT_EQ(out[18], -8);
    EXPECT_FLOAT_EQ(out[19], -3);
    EXPECT_FLOAT_EQ(out[20], 1);
    EXPECT_FLOAT_EQ(out[21], 4);
    EXPECT_FLOAT_EQ(out[22], -1);
    EXPECT_FLOAT_EQ(out[23], 0);
    EXPECT_FLOAT_EQ(out[24], 8);
    EXPECT_FLOAT_EQ(out[25], -1);
    EXPECT_FLOAT_EQ(out[26], 9);
    EXPECT_FLOAT_EQ(out[27], 2);
    EXPECT_FLOAT_EQ(out[28], 1);
    EXPECT_FLOAT_EQ(out[29], -2);
    EXPECT_FLOAT_EQ(out[30], 2);
    EXPECT_FLOAT_EQ(out[31], 0);
}

TEST(CommonTest, qkv){
    size_t seq_length = 2;
    size_t num_attention_heads = 4;
    size_t size_per_head = 3;
    float qk[32] = {
            0, 1,
            -1, -2,
            2, -1,
            0, 1,

            2, 1,
            0, 2,
            -1, -1,
            2, 1,

            0, 1,
            -1, -2,
            2, -1,
            0, 1,

            2, 1,
            0, 2,
            -1, -1,
            2, 1,
    };
    float value[48] = {
            3, 0, -1,
            2, 3, -3,
            0, -1, -1,
            1, 1, 1,

            -2, 3, 0,
            -1, 2, 2,
            2, 0, 0,
            0, 0, 0,

            3, 0, -1,
            2, 3, -3,
            0, -1, -1,
            1, 1, 1,

            -2, 3, 0,
            -1, 2, 2,
            2, 0, 0,
            0, 0, 0,
    };
    float out[48];
    const float* sim_array[8];
    const float* value_array[8];
    float* pointer_sv_array[8];
    attn_sv(2, num_attention_heads, seq_length, size_per_head, qk, value, out, sim_array, value_array, pointer_sv_array);
    EXPECT_FLOAT_EQ(out[0], -2);
    EXPECT_FLOAT_EQ(out[1], 3);
    EXPECT_FLOAT_EQ(out[2], 0);
    EXPECT_FLOAT_EQ(out[3], 0);
    EXPECT_FLOAT_EQ(out[4], -7);
    EXPECT_FLOAT_EQ(out[5], -1);
    EXPECT_FLOAT_EQ(out[6], -2);
    EXPECT_FLOAT_EQ(out[7], -2);
    EXPECT_FLOAT_EQ(out[8], -2);
    EXPECT_FLOAT_EQ(out[9], 0);
    EXPECT_FLOAT_EQ(out[10], 0);
    EXPECT_FLOAT_EQ(out[11], 0);
    EXPECT_FLOAT_EQ(out[12], 4);
    EXPECT_FLOAT_EQ(out[13], 3);
    EXPECT_FLOAT_EQ(out[14], -2);
    EXPECT_FLOAT_EQ(out[15], -2);
    EXPECT_FLOAT_EQ(out[16], 4);
    EXPECT_FLOAT_EQ(out[17], 4);
    EXPECT_FLOAT_EQ(out[18], -2);
    EXPECT_FLOAT_EQ(out[19], 1);
    EXPECT_FLOAT_EQ(out[20], 1);
    EXPECT_FLOAT_EQ(out[21], 2);
    EXPECT_FLOAT_EQ(out[22], 2);
    EXPECT_FLOAT_EQ(out[23], 2);
    EXPECT_FLOAT_EQ(out[24], -2);
    EXPECT_FLOAT_EQ(out[25], 3);
    EXPECT_FLOAT_EQ(out[26], 0);
    EXPECT_FLOAT_EQ(out[27], 0);
    EXPECT_FLOAT_EQ(out[28], -7);
    EXPECT_FLOAT_EQ(out[29], -1);
    EXPECT_FLOAT_EQ(out[30], -2);
    EXPECT_FLOAT_EQ(out[31], -2);
    EXPECT_FLOAT_EQ(out[32], -2);
    EXPECT_FLOAT_EQ(out[33], 0);
    EXPECT_FLOAT_EQ(out[34], 0);
    EXPECT_FLOAT_EQ(out[35], 0);
    EXPECT_FLOAT_EQ(out[36], 4);
    EXPECT_FLOAT_EQ(out[37], 3);
    EXPECT_FLOAT_EQ(out[38], -2);
    EXPECT_FLOAT_EQ(out[39], -2);
    EXPECT_FLOAT_EQ(out[40], 4);
    EXPECT_FLOAT_EQ(out[41], 4);
    EXPECT_FLOAT_EQ(out[42], -2);
    EXPECT_FLOAT_EQ(out[43], 1);
    EXPECT_FLOAT_EQ(out[44], 1);
    EXPECT_FLOAT_EQ(out[45], 2);
    EXPECT_FLOAT_EQ(out[46], 2);
    EXPECT_FLOAT_EQ(out[47], 2);
}