#include "dense.h"
#include<exception>
#include<mkl.h>
#include<iostream>

namespace lh{

    template<class T>
    Dense<T>::Dense(std::vector<std::string> names, Graph<T>& pb_graph){
        std::string name_w = names[0];
        if(pb_graph.find(name_w) == pb_graph.end()) throw std::invalid_argument("name" + name_w + "not found in graph!");
        Param<T>& w = pb_graph[name_w];
        std::vector<std::size_t> dims = w.first; 
        input_size_ = dims[0];
        output_size_ = dims[1];
        weight = new T[input_size_*output_size_];
        for(int i=0;i<input_size_*output_size_;i++){
            weight[i] = w.second[i];
        }
        
        if(names.size() > 1){
            std::string name_b = names[1];
            if(pb_graph.find(name_b) == pb_graph.end()) throw std::invalid_argument("name" + name_b +  "not found in graph!");
            Param<T>& b = pb_graph[name_b]; 
            bias = new T[output_size_];
            for(int i=0;i<output_size_;i++){
                bias[i] = b.second[i];
            }
        }
        else bias = nullptr;
    }

    template<class T>
    Dense<T>::~Dense(){
        delete [] weight;
        if(bias != nullptr) delete [] bias;
    }

    template<class T>
    void Dense<T>::compute(std::size_t batch_size, std::size_t seq_len, T* input, T* output){
        // input shape [batch_size, input_size_]
        // output shape [batch_size, output_size_]
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size*seq_len, output_size_, input_size_, 1.0, input, input_size_, weight, output_size_, 0.0, output, output_size_);
        if(bias != nullptr){
            for(int idx=0;idx<batch_size*seq_len;idx++){
                T* start = output + idx*output_size_;
                cblas_saxpby(output_size_, 1.0, bias, 1, 1.0, start, 1);
            }
        }
    }

    template class Dense<float>;

}