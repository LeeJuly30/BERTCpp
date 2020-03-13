#pragma once

// #include <cstddef>
// #include <vector>
// #include <unordered_map>
// #include <string>
#include "util.h"

namespace lh
{
    template<class T>
    T exp_(T input);

    template<class T>
    T sum_(T sum);

    template <class T>
    class Softmax
    {
        public:
            explicit Softmax();
            ~Softmax();
            void compute(std::size_t batch_size, std::size_t stride, T *input, T *output);
        private:

    };
} // namespace lh