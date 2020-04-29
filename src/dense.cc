#include "dense.h"
#include <exception>
#include <mkl.h>
#include <memory.h>

namespace lh{

    template<class T>
    Dense<T>::Dense(std::vector<std::string> names, Graph& pb_graph){
        std::string name_w = names[0];
        if(pb_graph.find(name_w) == pb_graph.end()) throw std::invalid_argument("name " + name_w + " not found in graph!");
        w = pb_graph[name_w];
        shape& dims = w->sizes; 
        input_size_ = dims[0];
        output_size_ = dims[1];
        switch(w->type_){
            case float32 : {
                create_weight_ptr();
                break;
            }
            case qint8 :{
                create_weight_ptr();
                break;
            } 
            default: throw std::invalid_argument("weight only accepte float or int8_t data type!");
        }
        
        if(names.size() > 1){
            std::string name_b = names[1];
            if(pb_graph.find(name_b) == pb_graph.end()) throw std::invalid_argument("name " + name_b +  " not found in graph!");
            b = pb_graph[name_b]; 
            switch(b->type_){
                case float32 : {
                    create_bias_ptr();
                    break;
                }
                case qint8 :{
                    create_bias_ptr();
                    break;
                } 
                default: throw std::invalid_argument("bias only accepte float or int8_t data type!");
            }
        }
        else{
            b = nullptr;
            bias = nullptr;
        }
        weight_observer = nullptr;
    }

    template<>
    void Dense<float>::create_weight_ptr(){
        if(w->type_ != float32) throw std::invalid_argument("weight type are not float");
        weight = w->raw_data_.float_ptr;
    }

    template<>
    void Dense<float>::create_bias_ptr(){
        if(b->type_ != float32) throw std::invalid_argument("bias type are not float");
        bias = b->raw_data_.float_ptr;
    }

    template<>
    void Dense<int8_t>::create_weight_ptr(){
        if(w->type_ != qint8) throw std::invalid_argument("weight type are not int8_t");
        weight = w->raw_data_.int8_ptr;
    }

    template<>
    void Dense<int8_t>::create_bias_ptr(){
        if(b->type_ != qint8) throw std::invalid_argument("bias type are not int8_t");
        bias = b->raw_data_.int8_ptr;
    }
 
    template<class T>
    Dense<T>::~Dense(){
        delete w; 
        if(b != nullptr) delete b; 
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