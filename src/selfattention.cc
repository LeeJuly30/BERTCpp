#include "selfattention.h"
#include "memory.h"
#include <cmath>

namespace lh{
    template<class T>
    MutiheadselfAttn<T>::MutiheadselfAttn(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len, std::size_t num_heads, std::size_t head_hidden_size){
        
        pre_batch_size_ = pre_batch_size;
        pre_seq_len_ = pre_seq_len;
        num_heads_ = num_heads;
        head_hidden_size_ = head_hidden_size;

        auto startit = names.begin();
        std::vector<std::string> query_names(startit, startit+2);
        query_layer = new Dense<T>(query_names, pb_graph);
        startit += 2;
        std::vector<std::string> key_names(startit, startit+2);
        key_layer = new Dense<T>(key_names, pb_graph);
        startit += 2;
        std::vector<std::string> value_names(startit, startit+2);
        value_layer = new Dense<T>(value_names, pb_graph);
        softmax = new Softmax<T>();

        query_layer_out = new T[pre_batch_size * pre_seq_len * head_hidden_size * num_heads];
        key_layer_out = new T[pre_batch_size * pre_seq_len * head_hidden_size * num_heads];
        value_layer_out = new T[pre_batch_size * pre_seq_len * head_hidden_size * num_heads];
        attention_scores = new T[pre_batch_size * pre_seq_len * pre_seq_len * num_heads];

        q_array = new const T *[pre_batch_size * num_heads];
        k_array = new const T *[pre_batch_size * num_heads];
        pointer_qk_array = new T* [pre_batch_size * num_heads];

        sim_array = new const T *[pre_batch_size * num_heads];
        value_array = new const T *[pre_batch_size * num_heads];
        pointer_sv_array = new T *[pre_batch_size * num_heads];
    
    }

    template<class T>
    MutiheadselfAttn<T>::~MutiheadselfAttn(){

        delete [] query_layer_out;
        delete [] key_layer_out;
        delete [] value_layer_out;
        delete [] attention_scores;

        delete query_layer;
        delete key_layer;
        delete value_layer;
        delete softmax;

        delete q_array;
        delete k_array;
        delete pointer_qk_array;

        delete sim_array;
        delete value_array;
        delete pointer_sv_array;
    }

    template<class T>
    void MutiheadselfAttn<T>::compute(std::size_t batch_size, std::size_t seq_len, T *input, uint64_t* mask, T *output){
        
        query_layer->compute(batch_size, seq_len, input, query_layer_out);
        key_layer->compute(batch_size, seq_len, input, key_layer_out);
        value_layer->compute(batch_size, seq_len, input, value_layer_out);

        attn_qk<T>(batch_size, num_heads_, seq_len, head_hidden_size_, query_layer_out, key_layer_out, attention_scores, q_array, k_array, pointer_qk_array);

        for(std::size_t idx = 0; idx < batch_size; idx++){
            uint64_t len = mask[idx];
            for(std::size_t len_idx = 0; len_idx < len; len_idx++){
                T* start = attention_scores + idx * seq_len * num_heads_ * seq_len + len_idx * num_heads_ * seq_len;
                for(std::size_t head_idx = 0; head_idx < num_heads_; head_idx++){
                    for(std::size_t j = 0; j < len; j++){
                        start[head_idx * seq_len + j] = start[head_idx * seq_len + j] / std::sqrt(head_hidden_size_);
                    }
                    for(std::size_t j = len; j < seq_len; j++){
                        start[head_idx * seq_len + j] = -10000.0f;
                    }
                }
            }

            for(std::size_t len_idx = len; len_idx < seq_len; len_idx++){
                T* start = attention_scores + idx * seq_len * num_heads_ * seq_len + len_idx * num_heads_ * seq_len;
                for(std::size_t sub_idx = 0; sub_idx < num_heads_ * seq_len; sub_idx++){
                    start[sub_idx] = -10000.0f;
                }
            }

        }

        softmax->compute(batch_size*seq_len*num_heads_, seq_len, attention_scores, attention_scores);

        attn_sv<T>(batch_size, num_heads_, seq_len, head_hidden_size_, attention_scores, value_layer_out, output, sim_array, value_array, pointer_sv_array);
    }

    template class MutiheadselfAttn<float>;

}