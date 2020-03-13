#pragma once

#include "util.h"

namespace lh
{   
    template<class T>
    void gelu_(std::size_t size, T* input);

    template <class T>
    class Gelu
    {
        public:
            explicit Gelu();
            ~Gelu();
            void compute(std::size_t size, T *input);

    };
} // namespace lh