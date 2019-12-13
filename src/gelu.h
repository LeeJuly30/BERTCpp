#pragma once

// #include <cstddef>
// #include <vector>
// #include <unordered_map>
// #include <string>
#include "util.h"

namespace lh
{
    template <class T>
    class Gelu
    {
        public:
            explicit Gelu();
            ~Gelu();
            void compute(std::size_t size, T *input);

    };
} // namespace lh