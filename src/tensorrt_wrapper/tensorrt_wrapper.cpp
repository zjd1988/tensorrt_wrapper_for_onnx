#include "tensorrt_wrapper.hpp"
#include "tensorrt_engine.hpp"

namespace TENSORRT_WRAPPER
{

    TensorrtWrapper::TensorrtWrapper(std::string engineFile, int deviceID)
    {
        auto ptr = new TensorrtEngine(engineFile, deviceID);
        inferenceEngine = ptr;
    }

    TensorrtWrapper::~TensorrtWrapper()
    {
        if(inferenceEngine != nullptr)
        {
            auto ptr = (TensorrtEngine*)inferenceEngine;
            delete ptr;
            inferenceEngine = nullptr;
        }
    }

    void TensorrtWrapper::prepareData(std::map<std::string, void*> dataMap)
    {
        auto ptr = (TensorrtEngine*)inferenceEngine;
        ptr->prepareData(dataMap);
    }

    void TensorrtWrapper::doInference(bool syncFlag)
    {
        auto ptr = (TensorrtEngine*)inferenceEngine;
        ptr->doInference(syncFlag);
    }

    std::map<std::string, void*> TensorrtWrapper::getInferenceResult()
    {
        auto ptr = (TensorrtEngine*)inferenceEngine;
        return ptr->getInferenceResult();
    }

} // namespace TENSORRT_WRAPPER