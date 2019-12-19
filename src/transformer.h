#include "util.h"
#include "dense.h"
#include "layernorm.h"
#include "gelu.h"
#include "selfattention.h"

namespace lh
{
    template<class T>
    class Transformer{
        public:
            explicit Transformer(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len, std::size_t num_heads, std::size_t head_hidden_size, std::size_t intermediate_ratio, std::size_t num_layers);
            ~Transformer();
            void compute(std::size_t batch_size, std::size_t seq_len, T* input, uint64_t* mask, T* output);
        
        private:
            std::size_t num_heads_;
            std::size_t hidden_size_;
            std::size_t intermediate_size_;
            std::size_t num_layers_;

            std::vector<MutiheadselfAttn<T>* > mutiheadselfattn_;
            std::vector<Dense<T>* > attention_output_dense_;
            std::vector<Layernorm<T>* > attention_layer_norm_;

            std::vector<Dense<T>* > intermediate_dense_;
            std::vector<Gelu<T>* > intermediate_act_;

            std::vector<Dense<T>* > output_dense_;
            std::vector<Layernorm<T>* > output_layer_norm_;

            std::vector<T*> atten_output_;
            std::vector<T*> atten_dense_output_;

            std::vector<T*> intermediate_dense_output_;

            std::vector<T* > output_dense_output_;




    };
} // namespace lh
