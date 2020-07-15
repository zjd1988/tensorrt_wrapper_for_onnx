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
        tensorrtEngine(std::string engineFile);
        ~tensorrtEngine();
        bool saveEnginePlanFile(std::string saveFile);
        void doInference(bool syncFlag);
        void createEngine(unsigned int maxBatchSize, bool fp16Flag);
        void prepareData(std::map<std::string, void*> dataMap);
        std::map<std::string, void*> getInferenceResult();

    private:
        void initConstTensors(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        void setNetInput(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        void createNetBackbone(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        Logger mLogger;
        std::shared_ptr<tensorrtInference::weightsAndGraphParse> weightsAndGraph;
        std::shared_ptr<tensorrtInference::executionParse> executionInfo;
        // nvinfer1::IRuntime* runtime;
        nvinfer1::IBuilder* builder;
        nvinfer1::ICudaEngine* cudaEngine;
        // nvinfer1::IExecutionContext* context;
        //gpu runtime
        std::shared_ptr<CUDARuntime> cudaRuntime;
    };
}

#endif