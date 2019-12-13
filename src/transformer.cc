#include "transformer.h"
#include "memory.h"

namespace lh{

    template<class T>
    Transformer<T>::Transformer(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len, std::size_t num_heads, std::size_t head_hidden_size, std::size_t intermediate_ratio, std::size_t num_layers){

        std::size_t hidden_size = num_heads * head_hidden_size;
        std::size_t intermediate_size = hidden_size * intermediate_ratio;

        pre_batch_size_ = pre_batch_size;
        pre_seq_len_ = pre_seq_len;
        num_heads_ = num_heads;
        hidden_size_ = hidden_size;
        intermediate_size_ = intermediate_size;
        num_layers_ = num_layers;

        mutiheadselfattn_.reserve(num_layers);
        attention_output_dense_.reserve(num_layers);
        attention_layer_norm_.reserve(num_layers);
        intermediate_dense_.reserve(num_layers);
        intermediate_act_.reserve(num_layers);
        output_dense_.reserve(num_layers);
        output_layer_norm_.reserve(num_layers);

        atten_output_.reserve(num_layers);
        atten_dense_output_.reserve(num_layers);
        intermediate_dense_output_.reserve(num_layers);
        output_dense_output_.reserve(num_layers);

        auto startit = names.begin();
        for(int layer_idx = 0; layer_idx < num_layers; layer_idx++){
            std::vector<std::string> attennames(startit, startit+6);
            mutiheadselfattn_[layer_idx] = new MutiheadselfAttn<T>(attennames, pb_graph, pre_batch_size, pre_seq_len, num_heads, head_hidden_size);
            atten_output_[layer_idx] = new T[pre_batch_size*pre_seq_len*hidden_size];
            startit += 6;

            std::vector<std::string> attendensenames(startit, startit+2);
            attention_output_dense_[layer_idx] = new Dense<T>(attendensenames, pb_graph);
            atten_dense_output_[layer_idx] = new T[pre_batch_size*pre_seq_len*hidden_size];
            startit += 2;

            std::vector<std::string> attennormnames(startit, startit+2);
            attention_layer_norm_[layer_idx] = new Layernorm<T>(attennormnames, pb_graph, pre_batch_size, pre_seq_len);
            startit += 2;

            std::vector<std::string> mediatedensenames(startit, startit+2);
            intermediate_dense_[layer_idx] = new Dense<T>(mediatedensenames, pb_graph);
            intermediate_dense_output_[layer_idx] = new T[pre_batch_size*pre_seq_len*hidden_size];
            startit += 2;

            intermediate_act_[layer_idx] = new Gelu<T>;

            std::vector<std::string> outputdensenames(startit, startit+2);
            output_dense_[layer_idx] = new Dense<T>(outputdensenames, pb_graph);
            output_dense_output_[layer_idx] = new T[pre_batch_size*pre_seq_len*hidden_size];
            startit += 2;

            std::vector<std::string> outputnormnames(startit, startit+2);
            output_layer_norm_[layer_idx] = new Layernorm<T>(outputnormnames, pb_graph, pre_batch_size, pre_seq_len);
            startit += 2;

        } 

    }

    template<class T>
    Transformer<T>::~Transformer(){
        
        for(int layer_idx = 0; layer_idx < num_layers_; layer_idx++){
            delete mutiheadselfattn_[layer_idx];
            delete attention_output_dense_[layer_idx];
            delete attention_layer_norm_[layer_idx];
            delete intermediate_dense_[layer_idx];
            delete intermediate_act_[layer_idx];
            delete output_dense_[layer_idx];
            delete output_layer_norm_[layer_idx];

            delete atten_output_[layer_idx];
            delete atten_dense_output_[layer_idx];
            delete intermediate_dense_output_[layer_idx];
            delete output_dense_output_[layer_idx];
        }
    }

    template<class T>
    void Transformer<T>::compute(std::size_t batch_size, std::size_t seq_len, T* input, int* mask, T* output){

        T* pre_input = input;

        for(int layer_idx = 0; layer_idx < num_layers_; layer_idx++){
            
            mutiheadselfattn_[layer_idx]->compute(batch_size, seq_len, pre_input, mask, atten_output_[layer_idx]);

            attention_output_dense_[layer_idx]->compute(batch_size, seq_len, atten_output_[layer_idx], atten_dense_output_[layer_idx]);

            for(int idx = 0; idx < batch_size * seq_len * hidden_size_; idx++){
                atten_dense_output_[layer_idx][idx] += pre_input[idx];
            }
            
            attention_layer_norm_[layer_idx]->compute(batch_size, seq_len, atten_dense_output_[layer_idx], atten_dense_output_[layer_idx]);
            
            intermediate_dense_[layer_idx]->compute(batch_size, seq_len, atten_dense_output_[layer_idx], intermediate_dense_output_[layer_idx]);
            intermediate_act_[layer_idx]->compute(batch_size*seq_len*intermediate_size_, intermediate_dense_output_[layer_idx]);

            output_dense_[layer_idx]->compute(batch_size, seq_len, intermediate_dense_output_[layer_idx], output_dense_output_[layer_idx]);
            for(int idx = 0; idx < batch_size * seq_len * hidden_size_; idx++){
                output_dense_output_[layer_idx][idx] += atten_dense_output_[layer_idx][idx];
            }

            output_layer_norm_[layer_idx]->compute(batch_size, seq_len, output_dense_output_[layer_idx], output_dense_output_[layer_idx]);

            memcpy(pre_input, output_dense_output_[layer_idx], sizeof(T)*batch_size*seq_len*hidden_size_);
        }

        memcpy(output, pre_input, sizeof(T)*batch_size*seq_len*hidden_size_);
    }

    template class Transformer<float>;
}