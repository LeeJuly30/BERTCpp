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
            // using Param = std::pair<std::vector<std::size_t>, T *>;
            // using Graph = std::unordered_map<std::string, Param>;
            explicit Layernorm(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len);
            ~Layernorm();
            void compute(std::size_t batch_size, std::size_t seq_len, T *input, T *output);

        private:
            std::size_t norm_size_;
            std::size_t pre_batch_size_;
            std::size_t pre_seq_len_;
            T *gamma; // shape [norm_size_]
            T *beta;  // shape [norm_size_]
            T* mean; // shape[pre_batch_size_, pre_seq_len_]
            T* var; // shape[pre_batch_size_, pre_seq_len_]
    };
} // namespace lh