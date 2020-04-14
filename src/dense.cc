#include "dense.h"
#include <exception>
#include <mkl.h>
#include <memory.h>

namespace lh{

    template<class T>
    Dense<T>::Dense(std::vector<std::string> names, Graph<T>& pb_graph){
        std::string name_w = names[0];
        if(pb_graph.find(name_w) == pb_graph.end()) throw std::invalid_argument("name " + name_w + " not found in graph!");
        Param<T>& w = pb_graph[name_w];
        std::vector<std::size_t> dims = w.first; 
        input_size_ = dims[0];
        output_size_ = dims[1];
        weight = new T[input_size_*output_size_];
        for(std::size_t i=0; i<input_size_*output_size_; i++){
            weight[i] = w.second[i];
        }
        
        if(names.size() > 1){
            std::string name_b = names[1];
            if(pb_graph.find(name_b) == pb_graph.end()) throw std::invalid_argument("name " + name_b +  " not found in graph!");
            Param<T>& b = pb_graph[name_b]; 
            bias = new T[output_size_];
            for(std::size_t i=0; i<output_size_; i++){
                bias[i] = b.second[i];
            }
        }
        else bias = nullptr;

        weight_observer = nullptr;
    }

    template<class T>
    Dense<T>::~Dense(){
        delete [] weight;
        if(bias != nullptr) delete [] bias;
    }

    template<>
    void Dense<float>::multiplyweight(std::size_t batch_size, std::size_t seq_len, float* input, float* output){
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size*seq_len, output_size_, input_size_, 1.0f, input, input_size_, weight, output_size_, 0.0f, output, output_size_);
    }

    template<>
    void Dense<float>::addbias(std::size_t batch_size, std::size_t seq_len, float* output){

        for(std::size_t idx = 0; idx < batch_size * seq_len; idx++){
            for(std::size_t feature_idx = 0; feature_idx < output_size_; feature_idx++){
                output[idx * output_size_ + feature_idx] += bias[feature_idx];
            }
        }
    }

    template<>
    void Dense<float>::compute(std::size_t batch_size, std::size_t seq_len, float* input, float* output){
        // input shape [batch_size, input_size_]
        // output shape [batch_size, output_size_]
        
        multiplyweight(batch_size, seq_len, input, output);
        // add bias vector here
        if(bias != nullptr){
            addbias(batch_size, seq_len, output);
        }
    }

    template<>
    void Dense<float>::addobserver(float average_constant){
        weight_observer = new Observer(average_constant);
    }

    template<>
    void Dense<float>::calibration(std::size_t batch_size, std::size_t seq_len, float* input, float* output){
        // calibration data, record blas output

        if(weight_observer == nullptr) throw std::invalid_argument("the observer is null, please add observer before calibration!");
        
        multiplyweight(batch_size, seq_len, input, output);
        
        // record output
        std::size_t size = batch_size * seq_len * output_size_;
        weight_observer->compute(output, size);

        // add bias vector here
        if(bias != nullptr){
            addbias(batch_size, seq_len, output);
        }

    }

    template class Dense<float>;

}