#include "embedding.h"
#include <exception>
#include <mkl.h>
#include <iostream>
#include <memory.h>

namespace lh
{

template <class T>
Embedding<T>::Embedding(std::vector<std::string> names, Graph<T> &pb_graph)
{
    if (names.size() > 1)
        throw std::invalid_argument("embedding only need 1 arg!");
    std::string name_w = names[0];
    if (pb_graph.find(name_w) == pb_graph.end())
        throw std::invalid_argument("name " + name_w + " not found in graph!");
    Param<T>& w = pb_graph[name_w];
    std::vector<std::size_t> dims = w.first;
    
    vocab_size_ = dims[0];
    embedding_size_ = dims[1];
    
    weight = new T[vocab_size_ * embedding_size_];
    for (std::size_t i = 0; i < vocab_size_ * embedding_size_; i++)
    {
        weight[i] = w.second[i];
    }
}

template <class T>
Embedding<T>::~Embedding()
{
    delete[] weight;
}

template <class T>
void Embedding<T>::compute(std::size_t batch_size, std::size_t seq_len, uint64_t *input, T *output)
{
    for (std::size_t i = 0; i < batch_size * seq_len; i++)
    {
        T *start = output + i * embedding_size_;
        uint64_t index = input[i];
        if (index >= vocab_size_)
            throw std::invalid_argument("index must small than vocab size");
        T *weight_start = weight + index * embedding_size_;
        memcpy(start, weight_start, embedding_size_ * sizeof(T));
    }
}

template class Embedding<float>;
} // namespace lh
