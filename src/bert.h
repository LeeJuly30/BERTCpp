#include "util.h"

#include "bertembedding.h"
#include "transformer.h"
#include "pooler.h"

namespace lh{

    template<class T>
    class Bert{

        public:
            explicit Bert(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len, std::size_t embedding_size, std::size_t num_heads, std::size_t head_hidden_size, std::size_t intermediate_ratio, std::size_t num_layers);
            ~Bert();
            void compute(std::size_t batch_size, std::size_t seq_len, uint64_t* token_input, uint64_t* posit_input, uint64_t* type_input, uint64_t* mask, T* seq_output, T* pool_output);

        private:
            std::size_t embedding_size_;
            std::size_t hidden_size_;

            BertEmbedding<T>* bertembedding_;
            Transformer<T>* transformer_;
            Pooler<T>* pooler_;

            T* embedding_output_;
    };
}