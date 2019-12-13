#include "util.h"

#include "bertembedding.h"
#include "transformer.h"

namespace lh{

    template<class T>
    class Bert{

        public:
            Bert(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len, std::size_t embedding_size, std::size_t num_heads, std::size_t head_hidden_size, std::size_t intermediate_ratio, std::size_t num_layers);
            ~Bert();
            void compute(std::size_t batch_size, std::size_t seq_len, int* token_input, int* posit_input, int* type_input, int* mask, T* output);

        private:
            std::size_t embedding_size_;
            std::size_t hidden_size_;

            BertEmbedding<T>* bertembedding_;
            Transformer<T>* transformer_;

            T* embedding_output_;
    };
}