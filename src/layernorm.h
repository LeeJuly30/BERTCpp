#pragma once

// #include <cstddef>
// #include <vector>
// #include <unordered_map>
// #include <string>
#include "util.h"

namespace lh
{
    template<class T>
    T var_compute(T input);

    template <class T>
    class Layernorm
    {
        public:
            explicit Layernorm(std::vector<std::string> names, Graph &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len);
            ~Layernorm();
            void compute(std::size_t batch_size, std::size_t seq_len, T *input, T *output);

        private:
            std::size_t norm_size_;
            std::size_t pre_batch_size_;
            std::size_t pre_seq_len_;

            T *gamma; // shape [norm_size_]
            T *beta;  // shape [norm_size_]
            tensor* w;
            tensor* b;
            T* mean; // shape[pre_batch_size_, pre_seq_len_]
            T* var; // shape[pre_batch_size_, pre_seq_len_]
    };
} // namespace lh