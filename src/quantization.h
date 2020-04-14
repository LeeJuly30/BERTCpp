#include <cstddef>
#include <algorithm>
#include <limits>
#include "math.h"

namespace lh{
    struct qparam{
        float scale;
        float zero_point;
    };

    class Observer{
        public:
            Observer(float average_constant);

            ~Observer();

            void find_min_max(float* data, std::size_t size, float& min_input, float& max_input);

            void update_min_max(float min_current, float max_current);

            void compute(float* data, std::size_t size);

            float min_val_;
            float max_val_;

        private:

            float average_constant_;
    };
    namespace quantization{
        template<class T>
        T quantizate(float data, float scale, float zero_point){
            float output = floorf( data / scale - zero_point);
            output = output > std::numeric_limits<T>::min() ? output: std::numeric_limits<T>::min();
            output = output < std::numeric_limits<T>::max() ? output: std::numeric_limits<T>::max();
            return output;
        }

        template<class T>
        float dequantizate(T data, float scale, float zero_point){
            float output = (float(data) + zero_point) * scale;
            return output;
        }

        template<class T>
        qparam ChooseQuantizationParams(float min_input, float max_input, T int_min, T int_max){

            max_input = std::max(0.0f, max_input);
            min_input = std::min(0.0f, min_input);
            qparam result;
            result.scale = (max_input - min_input) / 256;
            float zero_point = min_input / result.scale - int_min;
            if(zero_point < (float) int_min) result.zero_point = (float) int_min;
            else if(zero_point > (float) int_max) result.zero_point = (float) int_max;
            else result.zero_point = zero_point;

            return result;

        }
    }
}