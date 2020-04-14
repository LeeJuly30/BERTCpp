#include "quantization.h"

namespace lh{

    Observer::Observer(float average_constant){
        min_val_ = NAN;
        max_val_ = NAN;

        average_constant_ = average_constant;
    }

    Observer::~Observer(){

    };

    void Observer::find_min_max(float* data, std::size_t size, float& min_input, float& max_input){
        max_input = *std::max_element(data, data + size);
        min_input = *std::min_element(data, data + size);
    }

    void Observer::update_min_max(float min_current, float max_current){
        if(std::isnan(min_val_)) min_val_ = min_current;
        else min_val_ = min_val_ + average_constant_ * (min_current - min_val_);
        if(std::isnan(max_val_)) max_val_ = max_current;
        else max_val_ = max_val_ + average_constant_ * (max_current - max_val_);
    }

    void Observer::compute(float* data, std::size_t size){
        float min_current, max_current;
        find_min_max(data, size, min_current, max_current);
        update_min_max(min_current, max_current);
    }



}