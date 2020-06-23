#ifndef __CREATE_NODE_HPP__
#define __CREATE_NODE_HPP__
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "node_info.hpp"
#include "weights_graph_parse.hpp"
#include "utils.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
using namespace std;

namespace tensorrtInference
{
    extern nvinfer1::ILayer* createNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors, 
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo);
}

#endif