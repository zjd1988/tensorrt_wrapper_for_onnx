/********************************************
 * Filename: create_unary_node.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "NvInfer.h"
#include "node_info/node_info.hpp"
#include "parser/weight_graph_parser.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createUnaryNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info);

} // namespace TENSORRT_WRAPPER