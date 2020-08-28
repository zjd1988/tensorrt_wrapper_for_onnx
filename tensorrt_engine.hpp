#ifndef __TENSORRT_ENGINE_HPP__
#define __TENSORRT_ENGINE_HPP__

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "utils.hpp"
#include "weights_graph_parse.hpp"
#include "execution_parse.hpp"
#include "cuda_runtime.hpp"

using namespace nvinfer1;
using namespace std;

namespace tensorrtInference
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
        std::shared_ptr<tensorrtInference::weightsAndGraphParse> weightsAndGraph;
        std::shared_ptr<tensorrtInference::executionParse> executionInfo;
        //gpu runtime
        std::shared_ptr<CUDARuntime> cudaRuntime;
    };
}

#endif