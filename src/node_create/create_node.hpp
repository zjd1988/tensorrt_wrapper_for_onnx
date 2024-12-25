/********************************************
 * Filename: create_node.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "node_info.hpp"
#include "weight_graph_parser.hpp"
#include "utils.hpp"
#include "common.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
using namespace std;

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors, 
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info);

} // namespace TENSORRT_WRAPPER