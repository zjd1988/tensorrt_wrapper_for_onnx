#include "tensorrt_engine.hpp"
#include "create_node.hpp"
#include <fstream>
#include <vector>

using namespace std;

namespace tensorrtInference 
{
    tensorrtEngine::tensorrtEngine(std::string jsonFile, std::string weightsFile, bool fp16Flag)
    {
        weightsAndGraph.reset(new weightsAndGraphParse(jsonFile, weightsFile, fp16Flag));
        CHECK_ASSERT((weightsAndGraph.get()->getInitFlag() != false), "init jsonFile and weightsFile fail!!\n");
    }
    
    tensorrtEngine::tensorrtEngine(std::string engineFile, int gpuId)
    {
        cudaRuntime.reset(new CUDARuntime(gpuId));
        executionInfo.reset(new executionParse(cudaRuntime.get(), engineFile));
        CHECK_ASSERT((executionInfo.get()->getInitFlag() != false), "init engineFile fail!!\n");
    }

    tensorrtEngine::~tensorrtEngine()
    {
    }
    
    bool tensorrtEngine::saveEnginePlanFile(std::string saveFile)
    {
        return weightsAndGraph->saveEnginePlanFile(saveFile);
    }

    void tensorrtEngine::prepareData(std::map<std::string, void*> dataMap)
    {
        auto allTensors = executionInfo->getTensorsInfo();
        for(auto inputData : dataMap)
        {
            if(allTensors.count(inputData.first) != 0)
            {
                allTensors[inputData.first]->setHost(dataMap[inputData.first]);
            }
        }
    }

    void tensorrtEngine::doInference(bool syncFlag)
    {
        cudaRuntime->activate();
        executionInfo->runInference();
        if(syncFlag)
            cudaRuntime->onWaitFinish();
    }
    std::map<std::string, void*> tensorrtEngine::getInferenceResult()
    {
        return executionInfo->getInferenceResult();
    }

}