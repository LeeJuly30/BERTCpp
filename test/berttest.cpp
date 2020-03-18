#include "../src/bert.h"
#include "../src/tokenizer.h"
#include "../src/model.pb.h"
#include <fstream>
#include <iostream>
#include"gtest/gtest.h"

using namespace lh;
using namespace std;

class BERT_TEST : public ::testing::Test {
protected:
    void SetUp() override {
        Model model;
        fstream input("/root/BERTCpp/model/model.proto", ios::in | ios::binary);
        if (!model.ParseFromIstream(&input)) {
            throw std::invalid_argument("can not read protofile");
        }
        for(int i=0;i<model.param_size();i++){
            Model_Paramter paramter = model.param(i);
            int size = 1;
            vector<size_t> dims(paramter.n_dim());
            for(int j=0;j<paramter.n_dim();j++){
                int dim = paramter.dim(j);
                size *= dim;
                dims[j] = dim;
            }
            float* data = new float[size];
            for(int i=0;i<size;i++){
                data[i] = paramter.data(i);
            }
            graph[paramter.name()] = make_pair(dims, data);
        }
        google::protobuf::ShutdownProtobufLibrary();

    }

    void TearDown() override {
        for(auto var_param:graph){
            delete [] var_param.second.second;
        }
    }
    Graph<float> graph;
};

TEST_F(BERT_TEST, readmodel){

    EXPECT_EQ(graph["bert.encoder.layer.10.intermediate.dense.weight"].first[0], 768);
    EXPECT_EQ(graph["bert.encoder.layer.10.intermediate.dense.weight"].first[1], 3072);
    EXPECT_NEAR(graph["bert.encoder.layer.10.intermediate.dense.weight"].second[1], -0.03554476797580719, 5e-5);
    EXPECT_NEAR(graph["bert.encoder.layer.10.intermediate.dense.weight"].second[2359294], 0.014271574094891548, 5e-5);
    EXPECT_NEAR(graph["bert.embeddings.word_embeddings.weight"].second[1], 0.019025683403015137, 5e-5);
    EXPECT_NEAR(graph["bert.embeddings.word_embeddings.weight"].second[770], 0.013949137181043625, 5e-5);
}

TEST_F(BERT_TEST, loadmodel){
    size_t pre_batch_size = 12;
    size_t pre_seq_len = 512;
    size_t num_heads = 12;
    size_t embedding_size = 768;
    size_t head_hidden_size = 64;
    size_t intermediate_ratio = 4;
    size_t num_layers = 12;
    vector<string> names;
    names.push_back("bert.embeddings.word_embeddings.weight");
    names.push_back("bert.embeddings.position_embeddings.weight");
    names.push_back("bert.embeddings.token_type_embeddings.weight");
    names.push_back("bert.embeddings.LayerNorm.weight");
    names.push_back("bert.embeddings.LayerNorm.bias");
    for(int idx;idx<num_layers;idx++){
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.query.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.query.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.key.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.key.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.value.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.value.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.output.dense.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.output.dense.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.output.LayerNorm.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.output.LayerNorm.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".intermediate.dense.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".intermediate.dense.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".output.dense.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".output.dense.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".output.LayerNorm.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".output.LayerNorm.bias");
    }
    names.push_back("bert.pooler.dense.weight");
    names.push_back("bert.pooler.dense.bias");

    Bert<float> bert(names, graph, pre_batch_size, pre_seq_len, embedding_size, num_heads, head_hidden_size, intermediate_ratio, num_layers);
    FullTokenizer tokenizer("/root/BERTCpp/model/vocab.txt");

    vector<string> input_string = {u8"同学们，今天我们来学习一个新词汇，叫做量化交易，好了我们开始吧！", u8"因为有些算法还是不容易理解的，你得知道什么地方用什么，还得知道为啥那么用。单词就无脑背诵都记不下来，那LeetCode自然一次记不住就太正常了。其实上面的类比你懂了的话，你就知道，刷LeetCode也是无他，多刷两遍就好了，多总结总复习，常用的东西还真得背下来。"};
    
    vector<vector<string>> input_tokens(2);
    for(int i=0; i<2;i++){
        tokenizer.tokenize(input_string[i].c_str(), &input_tokens[i], 128);
        input_tokens[i].insert(input_tokens[i].begin(), "[CLS]");
        input_tokens[i].push_back("[SEP]");
    }
    uint64_t mask[2];
    for(int i=0; i<2;i++){
        mask[i] = input_tokens[i].size();
        for(int j=input_tokens[i].size();j<128;j++){
            input_tokens[i].push_back("[PAD]");
        }
    }
    uint64_t input_ids[256];
    uint64_t position_ids[256];
    uint64_t type_ids[256];
    for(int i=0; i<2;i++){
        tokenizer.convert_tokens_to_ids(input_tokens[i], input_ids + i*128);
        for(int j=0;j<128;j++){
            position_ids[i*128 + j] = j;
            type_ids[i*128 + j] = 0;
        }
    }
    EXPECT_EQ(input_ids[0], 101);
    EXPECT_EQ(input_ids[1], 1398);
    EXPECT_EQ(input_ids[128], 101);
    EXPECT_EQ(input_ids[129], 1728);

    float out[2*128*embedding_size];
    float pool_out[2*embedding_size];
    bert.compute(2, 128, input_ids, position_ids, type_ids, mask, out, pool_out);

    EXPECT_NEAR(out[0], 2.4571e-01, 1e-4);
    EXPECT_NEAR(out[1], -6.2416e-02, 1e-4);
    EXPECT_NEAR(out[2], -5.1928e-01, 1e-4);
    EXPECT_NEAR(out[765], -1.0644e-01, 1e-4);
    EXPECT_NEAR(out[766], -1.6784e-01, 1e-4);
    EXPECT_NEAR(out[767], -2.3593e-01, 1e-4);
    EXPECT_NEAR(out[768], 7.9876e-01, 1e-4);

    EXPECT_NEAR(out[98304], 5.9381e-01, 1e-4);
    EXPECT_NEAR(out[98305], -2.6037e-01, 1e-4);
    EXPECT_NEAR(out[98306], -1.1703e-02, 1e-4);
    EXPECT_NEAR(out[99071], -1.9830e-01, 1e-4);
    EXPECT_NEAR(out[99070], -3.1941e-01, 1e-4);
    EXPECT_NEAR(out[99072], 6.6202e-01, 1e-4);

    EXPECT_NEAR(pool_out[0], 0.9999, 1e-4);
    EXPECT_NEAR(pool_out[1], 0.9995, 1e-4);
    EXPECT_NEAR(pool_out[767], 0.7718, 1e-4);
    EXPECT_NEAR(pool_out[766], -0.9933, 1e-4);
    EXPECT_NEAR(pool_out[768], 0.9998, 1e-4);
    EXPECT_NEAR(pool_out[769], 0.9995, 1e-4);
    EXPECT_NEAR(pool_out[1535], 0.5488, 1e-4);
    EXPECT_NEAR(pool_out[1534], -0.9922, 1e-4);
}