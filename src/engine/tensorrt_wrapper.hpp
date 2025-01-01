/********************************************
 * Filename: tensorrt_wrapper.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <iostream>
#include <string>
#include <map>
using namespace std;

namespace TENSORRT_WRAPPER
{

    class TensorrtWrapper
    {
    public:
        TensorrtWrapper(std::string engineFile, int deviceID = 0);
        ~TensorrtWrapper();
        void prepareData(std::map<std::string, void*> dataMap);
        std::map<std::string, void*> getInferenceResult();
        void doInference(bool syncFlag);

    private:
        void *inferenceEngine = nullptr;
    };

} // namespace TENSORRT_WRAPPER