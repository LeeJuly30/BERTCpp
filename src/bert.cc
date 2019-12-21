#include "bert.h"

#ifdef PRFILE_FUNCTION
    #include <chrono>
    #include <iostream>
#endif

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

        std::vector<std::string> poolnames(startit, startit + 2);
        pooler_ = new Pooler<T>(poolnames, pb_graph, pre_batch_size, hidden_size_);
        startit += 2;

    }

    template<class T>
    Bert<T>::~Bert(){
        
        delete bertembedding_;
        delete transformer_;
        delete pooler_;

        delete [] embedding_output_;
    }    

    template<class T>
    void Bert<T>::compute(std::size_t batch_size, std::size_t seq_len, uint64_t* token_input, uint64_t* posit_input, uint64_t* type_input, uint64_t* mask, T* seq_output, T* pool_output){

        #ifdef PRFILE_FUNCTION
            auto begin = std::chrono::system_clock::now();
        #endif

        bertembedding_->compute(batch_size, seq_len, token_input, posit_input, type_input, embedding_output_);

        #ifdef PRFILE_FUNCTION
            auto end = std::chrono::system_clock::now();
            std::cout<<"bert embedding use: "<< std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << std::endl;
            begin = std::chrono::system_clock::now();
        #endif

        transformer_->compute(batch_size, seq_len, embedding_output_, mask, seq_output);

        #ifdef PRFILE_FUNCTION
            begin = std::chrono::system_clock::now();
        #endif

        pooler_->compute(batch_size, seq_len, seq_output, pool_output);

        #ifdef PRFILE_FUNCTION
            end = std::chrono::system_clock::now();
            std::cout<<"bert pooler use: "<< std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << std::endl;
            begin = std::chrono::system_clock::now();
        #endif
    }

    template class Bert<float>;
}