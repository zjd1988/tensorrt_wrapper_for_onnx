/********************************************
 * Filename: create_nonzero_node.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "NvInfer.h"
#include "node_info/node_info.hpp"
#include "infer_engine/weights_graph_parse.hpp"

//plugin currently not support DataType::kBOOL
namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createNonZeroNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info);

} // namespace TENSORRT_WRAPPER