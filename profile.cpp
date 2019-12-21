#include "src/bert.h"
#include "src/tokenizer.h"
#include "src/model.pb.h"
#include <fstream>
#include <iostream>
#include <chrono>

using namespace std;
using namespace lh;

int main()
{
    Model model;
    Graph<float> graph;
    fstream input("/home/dell/Desktop/bert-cpp/model/model.proto", ios::in | ios::binary);
    if (!model.ParseFromIstream(&input))
    {
        throw std::invalid_argument("can not read protofile");
    }
    for (int i = 0; i < model.param_size(); i++)
    {
        Model_Paramter paramter = model.param(i);
        int size = 1;
        vector<size_t> dims(paramter.n_dim());
        for (int j = 0; j < paramter.n_dim(); j++)
        {
            int dim = paramter.dim(j);
            size *= dim;
            dims[j] = dim;
        }
        float *data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = paramter.data(i);
        }
        graph[paramter.name()] = make_pair(dims, data);
    }
    google::protobuf::ShutdownProtobufLibrary();
    cout << "load paramter from protobuf successly!" << endl;

    size_t pre_batch_size = 100;
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
    for (int idx; idx < num_layers; idx++)
    {
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

    cout << "init model from pb file and tokenizer successly!" << endl;

    vector<string> input_string = {u8"how are you! i am very happy to see you guys, please give me five ok? thanks", u8"this is some jokes, please tell somebody else that reputation to user privacy protection. There is no central authority or supervisor having overall manipulations over others, which makes Bitcoin favored by many. Unlike lling piles of identity information sheets before opening bank accounts, users of Bitcoin need only a pseudonym, a.k.a an address or a hashed public key, to participate the system."};

    vector<vector<string>> input_tokens(2);
    for (int i = 0; i < 2; i++)
    {
        tokenizer.tokenize(input_string[i].c_str(), &input_tokens[i], 128);
        input_tokens[i].insert(input_tokens[i].begin(), "[CLS]");
        input_tokens[i].push_back("[SEP]");
    }
    uint64_t mask[2];
    for (int i = 0; i < 2; i++)
    {
        mask[i] = input_tokens[i].size();
        for (int j = input_tokens[i].size(); j < 128; j++)
        {
            input_tokens[i].push_back("[PAD]");
        }
    }
    uint64_t input_ids[256];
    uint64_t position_ids[256];
    uint64_t type_ids[256];
    for (int i = 0; i < 2; i++)
    {
        tokenizer.convert_tokens_to_ids(input_tokens[i], input_ids + i * 128);
        for (int j = 0; j < 128; j++)
        {
            position_ids[i * 128 + j] = j;
            type_ids[i * 128 + j] = 0;
        }
    }

    float out[2 * 128 * embedding_size];
    float pool_out[2 * embedding_size];

    auto begin = chrono::system_clock::now();
    for(int i = 0; i < 10; i++) bert.compute(2, 128, input_ids, position_ids, type_ids, mask, out, pool_out);
    auto end = chrono::system_clock::now();
    
    cout<<"time: "<<chrono::duration_cast<chrono::milliseconds>(end-begin).count() <<endl;
}