#include "pooler.h"
#include <cmath>
#include <memory.h>

namespace lh
{
    template<>
    void tanh_<float>(std::size_t size, float* input){
        for(std::size_t i = 0; i < size; i++) input[i] = tanhf(input[i]);
    }

    template<class T>
    Pooler<T>::Pooler(std::vector<std::string> names, Graph<T> &pb_graph, std::size_t pre_batch_size, std::size_t hidden_size){
        hidden_size_ = hidden_size;

        tranfor_dense_ = new Dense<T>(names, pb_graph);

        tranfor_dense_output_ = new T[pre_batch_size * hidden_size];
    }

    template<class T>
    Pooler<T>::~Pooler(){

        delete tranfor_dense_;

        delete [] tranfor_dense_output_;
    }

    template<class T>
    void Pooler<T>::compute(std::size_t batch_size, std::size_t seq_len, T* input, T* output){
        
        for(std::size_t idx = 0; idx < batch_size; idx++){
            memcpy(tranfor_dense_output_ + idx * hidden_size_, input + idx * seq_len * hidden_size_, hidden_size_*sizeof(T));
        }

        tranfor_dense_->compute(batch_size, 1, tranfor_dense_output_, output);

        tanh_(batch_size * hidden_size_, output);

    }

    template class Pooler<float>;
    
} // namespace lh
