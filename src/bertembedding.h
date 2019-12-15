#include "util.h"
#include "embedding.h"
#include "layernorm.h"

namespace lh{

    template<class T>
    class BertEmbedding{
        
        public:
            BertEmbedding(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len, std::size_t embedding_size);
            ~BertEmbedding();
            void compute(std::size_t batch_size, std::size_t seq_len, uint64_t* token_input, uint64_t* posit_input, uint64_t* type_input, T* output);

        private:
            std::size_t pre_batch_size_;
            std::size_t pre_seq_len_;
            std::size_t embedding_size_;

            Embedding<T>* word_embedding_;
            Embedding<T>* position_embedding_;
            Embedding<T>* token_type_embedding_;

            Layernorm<T>* embedding_layer_norm_;

            T* word_embedding_output_;
            T* position_embedding_output_;
            T* token_type_embedding_output_;

    };
}