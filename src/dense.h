#pragma once

// #include <cstddef>
// #include <vector>
// #include <unordered_map>
// #include <string>
#include "util.h"
#include "quantization.h"

namespace lh
{
    template <class T>
    class Dense
    {
        public:
            explicit Dense(std::vector<std::string> names, Graph &pb_graph);
            void create_weight_ptr();
            void create_bias_ptr();
            ~Dense();
            
            void compute(std::size_t batch_size, std::size_t seq_len, T *input, T *output);
            
            void addobserver(float average_constant);
            void calibration(std::size_t batch_size, std::size_t seq_len, T *input, T *output);

            Observer* weight_observer;

        private:
            void multiplyweight(std::size_t batch_size, std::size_t seq_len, T *input, T *output);
            void addbias(std::size_t batch_size, std::size_t seq_len, T *output);
            
            std::size_t input_size_;
            std::size_t output_size_;
            T *weight; // weight pointer shape [input_size_, output_size_]
            T *bias;   // bias pointer shape [output_size_]
            tensor* w; // tensor pointer, hold weight pointer.
            tensor* b; // tensor pointer, hold bias pointer

    };
} // namespace lh