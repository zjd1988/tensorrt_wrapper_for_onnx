/********************************************
 * Filename: tensorrt_engine.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "common/cuda_runtime.hpp"
#include "parser/graph_parser.hpp"
#include "parser/execution_parser.hpp"
using namespace nvinfer1;
using namespace std;

namespace TENSORRT_WRAPPER
{

    class TensorrtEngine
    {
    public:
        TensorrtEngine(const std::string json_file, const std::string weight_file, bool fp16_flag = false);
        TensorrtEngine(const std::string engine_file, int gpu_id = 0);
        ~TensorrtEngine() = default;
        bool saveEnginePlanFile(std::string save_file);
        void prepareData(std::map<std::string, void*> dataMap);
        void doInference(bool sync_flag);
        std::map<std::string, void*> getInferenceResult();

    private:
        std::shared_ptr<GraphParser>             m_graph_parser;
        std::shared_ptr<ExecutionParser>         m_execution_parser;
        // cuda runtime
        std::shared_ptr<CUDARuntime>             m_cuda_runtime;
    };

} // namespace TENSORRT_WRAPPER
