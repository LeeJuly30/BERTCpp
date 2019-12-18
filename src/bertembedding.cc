#include "bertembedding.h"

namespace lh{

    template<class T>
    BertEmbedding<T>::BertEmbedding(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len, std::size_t embedding_size){

        pre_batch_size_ = pre_batch_size;
        pre_seq_len_ = pre_seq_len;
        embedding_size_ = embedding_size;

        auto startit = names.begin();

        std::vector<std::string> wordembednames(startit, startit+1);
        word_embedding_ = new Embedding<T>(wordembednames, pb_graph);
        word_embedding_output_ = new T[pre_batch_size*pre_seq_len*embedding_size];
        startit += 1;

        std::vector<std::string> posiembednames(startit, startit+1);
        position_embedding_ = new Embedding<T>(posiembednames, pb_graph);
        position_embedding_output_ = new T[pre_batch_size*pre_seq_len*embedding_size];
        startit += 1;

        std::vector<std::string> typeembednames(startit, startit+1);
        token_type_embedding_ = new Embedding<T>(typeembednames, pb_graph);
        token_type_embedding_output_ = new T[pre_batch_size*pre_seq_len*embedding_size];
        startit += 1;

        std::vector<std::string> normnames(startit, startit+2);
        embedding_layer_norm_ = new Layernorm<T>(normnames, pb_graph, pre_batch_size, pre_seq_len);
        startit += 2;

    }

    template<class T>
    BertEmbedding<T>::~BertEmbedding(){

        delete word_embedding_;
        delete position_embedding_;
        delete token_type_embedding_;

        delete [] word_embedding_output_;
        delete [] position_embedding_output_;
        delete [] token_type_embedding_output_;
    }

    template<class T>
    void BertEmbedding<T>::compute(std::size_t batch_size, std::size_t seq_len, uint64_t* token_input, uint64_t* posit_input, uint64_t* type_input, T* output){

        word_embedding_->compute(batch_size, seq_len, token_input, word_embedding_output_);
        position_embedding_->compute(batch_size, seq_len, posit_input, position_embedding_output_);
        token_type_embedding_->compute(batch_size, seq_len, type_input, token_type_embedding_output_);

        for(std::size_t j = 0; j < batch_size * seq_len * embedding_size_; j++){
            word_embedding_output_[j] += position_embedding_output_[j] + token_type_embedding_output_[j];
        }

        embedding_layer_norm_->compute(batch_size, seq_len, word_embedding_output_, output);

    }

    template class BertEmbedding<float>;
}