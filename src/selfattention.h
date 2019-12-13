#include "util.h"

#include "dense.h"
#include "softmax.h"
#include "batchgemm.h"

namespace lh{
    template<class T>
    class MutiheadselfAttn{
        public:
            explicit MutiheadselfAttn(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len, std::size_t num_heads, std::size_t head_hidden_size);
            ~MutiheadselfAttn();
            void compute(std::size_t batch_size, std::size_t seq_len, T *input, int* mask, T *output);

        private:
            Dense<T>* query_layer;
            Dense<T>* key_layer;
            Dense<T>* value_layer;
            Softmax<T>* softmax;

            T* query_layer_out;
            T* key_layer_out;
            T* value_layer_out;
            T* attention_scores;

            std::size_t pre_batch_size_;
            std::size_t pre_seq_len_;
            std::size_t num_heads_;
            std::size_t head_hidden_size_;
    };
}