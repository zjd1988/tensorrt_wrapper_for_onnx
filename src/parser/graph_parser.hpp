/********************************************
 * Filename: graph_parser.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include "NvInfer.h"
#include "json/json.h"
#include "common/non_copyable.hpp"
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    struct WeightInfo
    {
        WeightInfo()
        {
            byte_count = 0;
            data_type = 0;
            data = nullptr;
            shape.clear();
        }

        nvinfer1::Weights getTensorrtWeights()
        {
            nvinfer1::Weights w;
            w.values = data;
            auto type = getTensorrtDataType((OnnxDataType)data_type);
            CHECK_ASSERT(type != -1, "not supported type !!");
            w.type = (nvinfer1::DataType)type;
            w.count = byte_count / onnxDataTypeEleCount[data_type];
            return w;
        }

        nvinfer1::Dims getTensorrtDims()
        {
            nvinfer1::Dims dims;
            dims.nbDims = shape.size();
            CHECK_ASSERT(shape.size() < nvinfer1::Dims::MAX_DIMS, "max dims must less equal than 8\n");
            for(int i = 0; i < shape.size(); i++)
            {
                dims.d[i] = shape[i];
            }
            return dims;
        }

        int                           byte_count;
        int                           data_type;
        std::vector<int>              shape;
        char*                         data;
    };

    class GraphParser : public NonCopyable
    {
    public:
        GraphParser(const std::string& json_file, const std::string& weight_file);
        ~GraphParser();
        bool getInitFlag() { return m_init_flag; }
        bool saveEngineFile(const std::string save_file);

    private:
        std::vector<std::string> getConstWeightTensorNames();
        void initConstTensors(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        void setNetInput(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        void createNetBackbone(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);        
        bool createEngine(unsigned int maxBatchSize, bool fp16Flag);
        bool parseTopoOrder(const Json::Value& root);
        bool parseWeightsInfo(const Json::Value& root, const std::string& weight_file);
        bool parseNodesInfo(const Json::Value& root);
        bool parseEngineInfo(const Json::Value& root);

    private:
        std::map<std::string, char*>                              m_weights_map;
        std::vector<std::string>                                  m_node_topo_order;
        std::map<std::string, std::shared_ptr<NodeInfo>>          m_node_info_map;
        std::map<std::string, WeightInfo>                         m_net_weights_info;
        std::vector<std::string>                                  m_input_names;
        std::vector<std::string>                                  m_output_names;
        bool                                                      m_init_flag = false;

        // create/save engine 
        Logger                                                    m_logger;
        nvinfer1::IBuilder*                                       m_builder = nullptr;
        nvinfer1::ICudaEngine*                                    m_cuda_engine = nullptr;
    };

} // namespace TENSORRT_WRAPPER