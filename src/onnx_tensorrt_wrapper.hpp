#pragma once
#include <iostream>
#include <string>
#include <map>
using namespace std;

namespace TENSORRT_WRAPPER
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

} // namespace TENSORRT_WRAPPER