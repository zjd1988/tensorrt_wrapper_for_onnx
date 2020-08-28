#ifndef __ONNX_TENSORRT_WRAPPER_HPP__
#define __ONNX_TENSORRT_WRAPPER_HPP__

#include <iostream>
#include <string>
#include <map>
using namespace std;

namespace tensorrtInference
{
    class onnxTensorrtWrapper
    {
    public:
        onnxTensorrtWrapper(std::string engineFile, int deviceID = 0);
        ~onnxTensorrtWrapper();
        void prepareData(std::map<std::string, void*> dataMap);
        std::map<std::string, void*> getInferenceResult();
        void doInference(bool syncFlag);
    private:
        void *inferenceEngine = nullptr;
    };
}

#endif