#pragma once

// #include <cstddef>
// #include <vector>
// #include <unordered_map>
// #include <string>
#include "util.h"

namespace lh
{
    template <class T>
    class Dense
    {
        public:
            // using Param = std::pair<std::vector<std::size_t>, T *>;
            // using Graph = std::unordered_map<std::string, Param>;
            explicit Dense(std::vector<std::string> names, Graph<T> &pb_graph);
            ~Dense();
            void compute(std::size_t batch_size, std::size_t seq_len, T *input, T *output);

        private:
            std::size_t input_size_;
            std::size_t output_size_;
            T *weight; // shape [input_size_, output_size_]
            T *bias;   // shape [output_size_]
    };
} // namespace lh