#include "util.h"

namespace lh{
    void buildgraphfrompb(std::string pb_path, Graph& graph){
        Model model;
        std::fstream input(pb_path, std::ios::in | std::ios::binary);
        if (!model.ParseFromIstream(&input)) {
            throw std::invalid_argument("can not read protofile");
        }
        for(int i=0;i<model.param_size();i++){
            Model_Paramter paramter = model.param(i);
            int size = 1;
            std::vector<std::size_t> dims(paramter.n_dim());
            for(int j=0;j<paramter.n_dim();j++){
                int dim = paramter.dim(j);
                size *= dim;
                dims[j] = dim;
            }
            dtype type = static_cast<dtype>(paramter.dtype());
            switch(type){
                case float32 : 
                {
                    float* data = new float[size];
                    for(int i=0;i<size;i++){
                        data[i] = paramter.data(i);
                    }
                    qparam q;
                    tensor* param = new tensor(static_cast<void*>(data), q, float32, dims);
                    graph[paramter.name()] = param; 
                    break;
                }
                case quint8 : 
                {
                    uint8_t* data = new uint8_t[size];
                    for(int i=0;i<size;i++){
                        data[i] = paramter.data(i);
                    }
                    qparam q(paramter.scale(), paramter.zero_point());
                    tensor* param = new tensor(static_cast<void*>(data), q, quint8, dims);
                    graph[paramter.name()] = param; 
                    break;
                }
                case qint8 : {
                    int8_t* data = new int8_t[size];
                    for(int i=0;i<size;i++){
                        data[i] = paramter.data(i);
                    }
                    qparam q(paramter.scale(), paramter.zero_point());
                    tensor* param = new tensor(static_cast<void*>(data), q, qint8, dims);
                    graph[paramter.name()] = param; 
                    break;
                }
                case qint32 : {
                    int32_t* data = new int32_t[size];
                    for(int i=0;i<size;i++){
                        data[i] = paramter.data(i);
                    }
                    qparam q(paramter.scale(), paramter.zero_point());
                    tensor* param = new tensor(static_cast<void*>(data), q, qint32, dims);
                    graph[paramter.name()] = param; 
                    break;
                }
            }
        }
        google::protobuf::ShutdownProtobufLibrary();
    }
}