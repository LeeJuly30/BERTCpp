#include "softmax.h"
#include <cmath>

namespace lh{
    template<>
    float exp_<float>(float input){
        return expf(input);
    }

    template<>
    float sum_<float>(float sum){
        return sum > 1e-22f ? sum : 1e-22f;
    }

    template<class T>
    Softmax<T>::Softmax(){
        
    }

    template<class T>
    Softmax<T>::~Softmax(){

    }

    template<class T>
    void Softmax<T>::compute(std::size_t batch_size, std::size_t stride, T *input, T *output){
        for(std::size_t idx = 0; idx < batch_size; idx++){
            T sum = 0;
            for(std::size_t i=idx*stride; i<(idx+1)*stride;i++){
                output[i] = exp_(input[i]);
                sum += output[i];
            }
            sum = sum_(sum);
            for(std::size_t i=idx*stride; i<(idx+1)*stride;i++){
                output[i] = output[i] / sum; 
            }
        }
    }

    template class Softmax<float>;
}