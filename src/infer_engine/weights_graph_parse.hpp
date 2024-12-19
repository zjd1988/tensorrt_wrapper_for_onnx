/********************************************
 * Filename: weights_graph_parse.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "utils.hpp"
#include "json/json.h"
#include "node_info.hpp"

namespace tensorrtInference
{
    struct weightInfo {
        weightInfo() {
            byteCount = 0;
            dataType = 0;
            data = nullptr;
            shape.clear();
        }
        nvinfer1::Weights getTensorrtWeights()
        {
            nvinfer1::Weights w{};
            w.values = data;
            auto type = tensorrtInference::getTensorrtDataType((tensorrtInference::OnnxDataType)dataType);
            CHECK_ASSERT(type != -1, "not supported type !!\n");
            w.type = (nvinfer1::DataType)type;
            w.count = byteCount / onnxDataTypeEleCount[dataType];
            return w;
        }
        nvinfer1::Dims getTensorrtDims()
        {
            nvinfer1::Dims dims;
            dims.nbDims = shape.size();
            CHECK_ASSERT(shape.size() < nvinfer1::Dims::MAX_DIMS, "max dims must less equal than 8\n");
            for(int i = 0; i < shape.size(); i++) {
                dims.d[i] = shape[i];
            }
            return dims;
        }
        int byteCount;
        int dataType;
        std::vector<int> shape;
        char* data;
    };

    class weightsAndGraphParse {
    public:
        weightsAndGraphParse(std::string &jsonFile, std::string &weightsFile, bool fp16Flag);
        ~weightsAndGraphParse();
        bool getInitFlag() {return initFlag;}
        bool saveEnginePlanFile(std::string saveFile);
    private:
        std::vector<std::string> getConstWeightTensorNames();
        void initConstTensors(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        void setNetInput(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        void createNetBackbone(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);        
        void createEngine(unsigned int maxBatchSize, bool fp16Flag);
        bool extractNodeInfo(Json::Value &root);
        std::map<std::string, char*> weightsData;
        std::vector<std::string> topoNodeOrder;
        std::map<std::string, std::shared_ptr<nodeInfo>> nodeInfoMap;
        std::map<std::string, weightInfo> netWeightsInfo;
        std::vector<std::string> inputTensorNames;
        std::vector<std::string> outputTensorNames;
        bool initFlag = false;
        
        // create/save engine 
        Logger mLogger;
        nvinfer1::IBuilder* builder;
        nvinfer1::ICudaEngine* cudaEngine;
    };
} //tensorrtInference