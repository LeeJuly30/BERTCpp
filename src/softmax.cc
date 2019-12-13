#include "softmax.h"
#include <cmath>

namespace lh{
    template<class T>
    Softmax<T>::Softmax(){
        
    }

    template<class T>
    Softmax<T>::~Softmax(){

    }

    template<class T>
    void Softmax<T>::compute(std::size_t batch_size, std::size_t stride, T *input, T *output){
        for(int idx = 0; idx < batch_size; idx++){
            T sum = 0;
            for(int i=idx*stride; i<(idx+1)*stride;i++){
                output[i] = expf(input[i]);
                sum += output[i];
            }
            sum = sum > 1e-22f ? sum : 1e-22f;
            for(int i=idx*stride; i<(idx+1)*stride;i++){
                output[i] = output[i] / sum; 
            }
        }
    }

    template class Softmax<float>;
}