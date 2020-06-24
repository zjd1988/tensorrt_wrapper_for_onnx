#ifndef __COMMON_HPP__
#define __COMMON_HPP__
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
    extern bool broadcastTensors(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor*& tensor1, nvinfer1::ITensor*& tensor2);
}


#endif //__COMMON_HPP__