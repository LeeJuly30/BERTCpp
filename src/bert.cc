#include "bert.h"

namespace lh{

    template<class T>
    Bert<T>::Bert(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len, std::size_t embedding_size, std::size_t num_heads, std::size_t head_hidden_size, std::size_t intermediate_ratio, std::size_t num_layers){

        embedding_size_ = embedding_size;
        hidden_size_ = num_heads * head_hidden_size;

        auto startit = names.begin();
        std::vector<std::string> embednames(startit, startit+5);
        bertembedding_ = new BertEmbedding<T>(embednames, pb_graph, pre_batch_size, pre_seq_len, embedding_size);
        embedding_output_ = new T[pre_batch_size * pre_seq_len * embedding_size];
        startit += 5;

        std::vector<std::string> transnames(startit, startit + 16*num_layers);
        transformer_ = new Transformer<T>(transnames, pb_graph, pre_batch_size, pre_seq_len, num_heads, head_hidden_size, intermediate_ratio, num_layers);
        startit += 16*num_layers;

    }

    template<class T>
    Bert<T>::~Bert(){
        
        delete bertembedding_;
        delete transformer_;

        delete [] embedding_output_;
    }    

    template<class T>
    void Bert<T>::compute(std::size_t batch_size, std::size_t seq_len, int* token_input, int* posit_input, int* type_input, int* mask, T* output){

        bertembedding_->compute(batch_size, seq_len, token_input, posit_input, type_input, embedding_output_);

        transformer_->compute(batch_size, seq_len, embedding_output_, mask, output);
    }

    template class Bert<float>;
}