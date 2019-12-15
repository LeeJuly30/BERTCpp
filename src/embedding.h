#pragma once

// #include <cstddef>
// #include <vector>
// #include <unordered_map>
// #include <string>
#include "util.h"

namespace lh
{
    template <class T>
    class Embedding
    {
        public:
            // using Param = std::pair<std::vector<std::size_t>, T *>;
            // using Graph = std::unordered_map<std::string, Param>;
            explicit Embedding(std::vector<std::string> names, Graph<T> &pb_graph);
            ~Embedding();
            void compute(std::size_t batch_size, std::size_t seq_len, uint64_t *input, T *output);

        private:
            std::size_t vocab_size_;
            std::size_t embedding_size_;
            T *weight; // shape [vocab_size_, embedding_size_]
    };
} // namespace lh