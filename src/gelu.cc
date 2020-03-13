#include "gelu.h"
#include <exception>
#include <cmath>

namespace lh
{   
    template<>
    void gelu_<float>(std::size_t size, float* input){
        for(std::size_t i=0; i<size; i++) input[i] = input[i] * 0.5f * (1.0f + erff(input[i] * sqrtf(0.5f)));
    }

    template<class T>
    Gelu<T>::Gelu(){

    }

    template<class T>
    Gelu<T>::~Gelu(){

    }

    template<class T>
    void Gelu<T>::compute(std::size_t size, T *input){
        gelu_(size, input);
    }

    template class Gelu<float>;
}