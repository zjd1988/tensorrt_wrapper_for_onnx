/********************************************
 * Filename: weight_graph_parser.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "json/json.h"
#include "common/utils.hpp"
#include "parser/node_parser.hpp"

namespace TENSORRT_WRAPPER
{

    struct WeightInfo
    {
        WeightInfo()
        {
            byteCount = 0;
            dataType = 0;
            data = nullptr;
            shape.clear();
        }

        nvinfer1::Weights getTensorrtWeights()
        {
            nvinfer1::Weights w{};
            w.values = data;
            auto type = getTensorrtDataType((OnnxDataType)dataType);
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

    class WeightGraphParser
    {
    public:
        WeightGraphParser(const std::string &json_file, std::string &weight_file, bool fp16_flag);
        ~WeightGraphParser();
        bool getInitFlag() { return m_init_flag; }
        bool saveEnginePlanFile(const std::string save_file);

    private:
        std::vector<std::string> getConstWeightTensorNames();
        void initConstTensors(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        void setNetInput(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        void createNetBackbone(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);        
        void createEngine(unsigned int maxBatchSize, bool fp16Flag);
        bool extractNodeInfo(Json::Value &root);

    private:
        std::map<std::string, char*>                              m_weights_data;
        std::vector<std::string>                                  m_topo_node_order;
        std::map<std::string, std::shared_ptr<NodeInfo>>          m_node_info_map;
        std::map<std::string, WeightInfo>                         m_net_weights_info;
        std::vector<std::string>                                  m_input_tensor_names;
        std::vector<std::string>                                  m_output_tensor_names;
        bool                                                      m_init_flag = false;

        // create/save engine 
        Logger                                                    m_logger;
        nvinfer1::IBuilder*                                       m_builder;
        nvinfer1::ICudaEngine*                                    m_cuda_engine;
    };

} // namespace TENSORRT_WRAPPER