# BERTCpp
A lightly BERT inference project using intel MKL and Protobuf in c++ (working in progress)
## Dependency
### Protobuf
bertcpp using protobuf to convert pytorch pretrained model in pb file and load it in c++
### MKL
bertcpp using MKL to implement blas operator
### utf8proc
bertcpp using utf8proc to process string
## Build
```bash
mkdir build & cd build
cmake ..
make -j4
```
To run unitest
```bash
./bert_test
```
## Thanks
tokenizer part comes from [cuBERT](https://github.com/zhihu/cuBERT)