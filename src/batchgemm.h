#pragma once

#include "util.h"

namespace lh{
    template<class T>
    void attn_qk(std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t hidden_size, T* query, T* key, T* output);

    template<class T>
    void attn_sv(std::size_t batch_size, std::size_t num_heads, std::size_t seq_len, std::size_t hidden_size, T* sim, T* value, T* output);
}