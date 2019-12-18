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
        fstream input("/home/dell/Desktop/bert-cpp/model/model.proto", ios::in | ios::binary);
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
    EXPECT_NEAR(graph["bert.encoder.layer.10.intermediate.dense.weight"].second[1], 0.0700, 5e-5);
    EXPECT_NEAR(graph["bert.encoder.layer.10.intermediate.dense.weight"].second[2359294], 0.0157, 5e-5);
    EXPECT_NEAR(graph["bert.embeddings.word_embeddings.weight"].second[1], -0.0615, 5e-5);
    EXPECT_NEAR(graph["bert.embeddings.word_embeddings.weight"].second[770], -0.0323, 5e-5);
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
    names.push_back("bert.embeddings.LayerNorm.gamma");
    names.push_back("bert.embeddings.LayerNorm.beta");
    for(int idx;idx<num_layers;idx++){
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.query.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.query.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.key.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.key.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.value.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.value.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.output.dense.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.output.dense.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.output.LayerNorm.gamma");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.output.LayerNorm.beta");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".intermediate.dense.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".intermediate.dense.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".output.dense.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".output.dense.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".output.LayerNorm.gamma");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".output.LayerNorm.beta");
    }
    names.push_back("bert.pooler.dense.weight");
    names.push_back("bert.pooler.dense.bias");

    Bert<float> bert(names, graph, pre_batch_size, pre_seq_len, embedding_size, num_heads, head_hidden_size, intermediate_ratio, num_layers);
    FullTokenizer tokenizer("/home/dell/Desktop/bert-cpp/model/bert-base-uncased-vocab.txt");

    vector<string> input_string = {u8"how are you! i am very happy to see you guys, please give me five ok? thanks", u8"this is some jokes, please tell somebody else that reputation to user privacy protection. There is no central authority or supervisor having overall manipulations over others, which makes Bitcoin favored by many. Unlike lling piles of identity information sheets before opening bank accounts, users of Bitcoin need only a pseudonym, a.k.a an address or a hashed public key, to participate the system."};
    
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
    EXPECT_EQ(input_ids[1], 2129);
    EXPECT_EQ(input_ids[128], 101);
    EXPECT_EQ(input_ids[129], 2023);

    float out[2*128*embedding_size];
    float pool_out[2*embedding_size];
    bert.compute(2, 128, input_ids, position_ids, type_ids, mask, out, pool_out);

    EXPECT_NEAR(out[0], 0.0051, 1e-4);
    EXPECT_NEAR(out[1], 0.2862, 1e-4);
    EXPECT_NEAR(out[2], 0.3087, 1e-4);
    EXPECT_NEAR(out[765], -0.0016, 1e-4);
    EXPECT_NEAR(out[766], 0.3782, 1e-4);
    EXPECT_NEAR(out[767], 0.4665, 1e-4);
    EXPECT_NEAR(out[768], -0.4208, 1e-4);

    EXPECT_NEAR(out[98304], -0.1748, 1e-4);
    EXPECT_NEAR(out[98305], -0.0781, 1e-4);
    EXPECT_NEAR(out[98306], 0.0979, 1e-4);
    EXPECT_NEAR(out[99071], 1.1872, 1e-4);
    EXPECT_NEAR(out[99070], -0.0092, 1e-4);
    EXPECT_NEAR(out[99072], -0.4539, 1e-4);

    EXPECT_NEAR(pool_out[0], -0.8036, 1e-4);
    EXPECT_NEAR(pool_out[1], -0.5174, 1e-4);
    EXPECT_NEAR(pool_out[767], 0.8579, 1e-4);
    EXPECT_NEAR(pool_out[766], -0.7373, 1e-4);
    EXPECT_NEAR(pool_out[768], -0.4499, 1e-4);
    EXPECT_NEAR(pool_out[769], -0.2875, 1e-4);
    EXPECT_NEAR(pool_out[1535], 0.3012, 1e-4);
    EXPECT_NEAR(pool_out[1534], -0.5041, 1e-4);
}