/********************************************
 * Filename: tensorrt_engine.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include <fstream>
#include <vector>
#include "common/utils.hpp"
#include "engine/tensorrt_engine.hpp"
using namespace std;

namespace TENSORRT_WRAPPER
{

    TensorrtEngine::TensorrtEngine(const std::string json_file, const std::string weight_file)
    {
        m_graph_parser.reset(new GraphParser(json_file, weight_file));
        CHECK_ASSERT(m_graph_parser->getInitFlag(), "init from json_file: {} and weight_file: {} fail!", json_file, weight_file);
        return;
    }

    TensorrtEngine::TensorrtEngine(const std::string engine_file, int gpu_id)
    {
        m_cuda_runtime.reset(new CUDARuntime(gpu_id));
        CHECK_ASSERT(m_cuda_runtime.get()->getInitFlag(), "init from engine_file: {} fail!", engine_file);
        m_execution_parser.reset(new ExecutionParser(m_cuda_runtime.get(), engine_file));
        CHECK_ASSERT(m_execution_parser.get()->getInitFlag(), "init from engine_file: {} fail!", engine_file);
        return;
    }

    bool TensorrtEngine::saveEnginePlanFile(std::string save_file)
    {
        return m_graph_parser->saveEnginePlanFile(save_file);
    }

    void TensorrtEngine::prepareData(std::map<std::string, void*> dataMap)
    {
        auto allTensors = m_execution_parser->getTensorsInfo();
        for(auto inputData : dataMap)
        {
            if(allTensors.count(inputData.first) != 0)
            {
                allTensors[inputData.first]->setHost(dataMap[inputData.first]);
            }
        }
        return;
    }

    void TensorrtEngine::doInference(bool sync_flag)
    {
        m_cuda_runtime->activate();
        m_execution_parser->runInference();
        if(sync_flag)
            m_cuda_runtime->onWaitFinish();
    }
    std::map<std::string, void*> TensorrtEngine::getInferenceResult()
    {
        return m_execution_parser->getInferenceResult();
    }

} // namespace TENSORRT_WRAPPER