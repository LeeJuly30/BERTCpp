#include "util.h"
#include "dense.h"

namespace lh
{
    template<class T>
    void tanh_(std::size_t size, T* input);

    template<class T>
    class Pooler{
        public:
            explicit Pooler(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t hidden_size);
            ~Pooler();
            void compute(std::size_t batch_size, std::size_t seq_len, T* input, T* output);
        
        private:
            std::size_t hidden_size_;

            Dense<T>* tranfor_dense_;

            T* tranfor_dense_output_;
    };
} // namespace lh
