#include "layernorm.h"
#include <exception>
#include <mkl.h>
#include <iostream>
#include <cmath>
#include <memory.h>

namespace lh
{
    template<>
    float var_compute<float>(float input){
        return 1.f / sqrtf(input + 1e-12f);
    }

    template <class T>
    Layernorm<T>::Layernorm(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t pre_seq_len)
    {
        pre_batch_size_ = pre_batch_size;
        pre_seq_len_ = pre_seq_len;

        std::string name_w = names[0];
        if (pb_graph.find(name_w) == pb_graph.end())
            throw std::invalid_argument("name "+ name_w + " not found in graph!");
        Param<T>& w = pb_graph[name_w];
        std::vector<std::size_t> dims = w.first;
        norm_size_ = dims[0];
        gamma = new T[norm_size_];
        for (int i = 0; i < norm_size_; i++)
        {
            gamma[i] = w.second[i];
        }
        
        std::string name_b = names[1];
        if (pb_graph.find(name_b) == pb_graph.end())
            throw std::invalid_argument("name " + name_b + " not found in graph!");
        Param<T>& b = pb_graph[name_b];
        beta = new T[norm_size_];
        for (int i = 0; i < norm_size_; i++)
        {
            beta[i] = b.second[i];
        }
        
        mean = new T[pre_batch_size*pre_seq_len];
        var = new T[pre_batch_size*pre_seq_len];
    }

    template <class T>
    Layernorm<T>::~Layernorm()
    {
        delete[] gamma;
        delete[] beta;
        delete[] mean;
        delete[] var;
    }

    template <class T>
    void Layernorm<T>::compute(std::size_t batch_size, std::size_t seq_len, T *input, T *output)
    {
        // input [batch_size, seq_len, norm_size]
        // output [batch_size, seq_len, norm_size]

        for(std::size_t idx=0; idx < batch_size*seq_len; idx++){
            mean[idx] = 0;
            var[idx] = 0;
            for(int j=0; j < norm_size_; j++){
                mean[idx] += input[idx*norm_size_+j] / norm_size_;
                var[idx] += input[idx*norm_size_+j] * input[idx*norm_size_+j] /  norm_size_;
            }
            var[idx] -= mean[idx]*mean[idx];
            var[idx] = var_compute(var[idx]);
            for(int j=0; j < norm_size_; j++){
                output[idx*norm_size_+j] = beta[j] + gamma[j] * var[idx] * (input[idx*norm_size_+j] - mean[idx]);
            } 
        }
    }

    template class Layernorm<float>;

} // namespace lh