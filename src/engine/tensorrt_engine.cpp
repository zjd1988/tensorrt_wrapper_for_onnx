/********************************************
 * Filename: tensorrt_engine.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include <fstream>
#include <vector>
#include "infer_engine/tensorrt_engine.hpp"
#include "node_create/create_node.hpp"
using namespace std;

namespace TENSORRT_WRAPPER
{

    TensorrtEngine::TensorrtEngine(const std::string json_file, const std::string weightsFile, bool fp16_flag)
    {
        m_weight_graph_parser.reset(new WeightGraphParser(json_file, weightsFile, fp16_flag));
        CHECK_ASSERT(m_weight_graph_parser.get()->getInitFlag(), "init from json_file and weight_file fail!\n");
        return;
    }

    TensorrtEngine::TensorrtEngine(const std::string engine_file, int gpu_id)
    {
        m_cuda_runtime.reset(new CUDARuntime(gpu_id));
        m_execution_parser.reset(new ExecutionParser(m_cuda_runtime.get(), engine_file));
        CHECK_ASSERT(m_execution_parser.get()->getInitFlag(), "init from engine_file fail!\n");
        return;
    }

    bool TensorrtEngine::saveEnginePlanFile(std::string save_file)
    {
        return m_weight_graph_parser->saveEnginePlanFile(save_file);
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