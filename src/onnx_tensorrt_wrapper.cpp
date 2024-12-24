#include "onnx_tensorrt_wrapper.hpp"
#include "tensorrt_engine.hpp"

namespace TENSORRT_WRAPPER
{
    onnxTensorrtWrapper::onnxTensorrtWrapper(std::string engineFile, int deviceID)
    {
        auto ptr = new tensorrtEngine(engineFile, deviceID);
        inferenceEngine = ptr;
    }
    onnxTensorrtWrapper::~onnxTensorrtWrapper()
    {
        if(inferenceEngine != nullptr)
        {
            auto ptr = (tensorrtEngine*)inferenceEngine;
            delete ptr;
            inferenceEngine = nullptr;
        }
    }
    void onnxTensorrtWrapper::prepareData(std::map<std::string, void*> dataMap)
    {
        auto ptr = (tensorrtEngine*)inferenceEngine;
        ptr->prepareData(dataMap);
    }

    void onnxTensorrtWrapper::doInference(bool syncFlag)
    {
        auto ptr = (tensorrtEngine*)inferenceEngine;
        ptr->doInference(syncFlag);
    }
    std::map<std::string, void*> onnxTensorrtWrapper::getInferenceResult()
    {
        auto ptr = (tensorrtEngine*)inferenceEngine;
        return ptr->getInferenceResult();
    }

}