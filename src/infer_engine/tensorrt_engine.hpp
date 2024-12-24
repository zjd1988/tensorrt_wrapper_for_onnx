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
#include "common/utils.hpp"
#include "common/cuda_runtime.hpp"
#include "infer_engine/weights_graph_parse.hpp"
#include "infer_engine/execution_parse.hpp"

using namespace nvinfer1;
using namespace std;

namespace TENSORRT_WRAPPER
{

    class tensorrtEngine
    {
    public:
        tensorrtEngine(std::string jsonFile, std::string weightsFile, bool fp16Flag = false);
        tensorrtEngine(std::string engineFile, int gpuId = 0);
        ~tensorrtEngine();
        bool saveEnginePlanFile(std::string saveFile);
        // void createEngine(unsigned int maxBatchSize, bool fp16Flag);
        void prepareData(std::map<std::string, void*> dataMap);
        void doInference(bool syncFlag);
        std::map<std::string, void*> getInferenceResult();

    private:
        std::shared_ptr<weightsAndGraphParse> weightsAndGraph;
        std::shared_ptr<executionParse> executionInfo;
        //gpu runtime
        std::shared_ptr<CUDARuntime> cudaRuntime;
    };

} // namespace TENSORRT_WRAPPER
