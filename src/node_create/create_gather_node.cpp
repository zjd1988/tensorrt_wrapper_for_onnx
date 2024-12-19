#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_gather_node.hpp"
#include "gather_node_info.hpp"

namespace tensorrtInference
{
    nvinfer1::ILayer* createGatherNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto gatherNodeInfo = (GatherNodeInfo*)nodeConfInfo;
        auto inputs = gatherNodeInfo->getInputs();
        int axis    = gatherNodeInfo->getAxis();
        nvinfer1::ITensor* data        = (tensors.count(inputs[0]) != 0) ? tensors[inputs[0]] : nullptr;
        nvinfer1::ITensor* indices     = (tensors.count(inputs[1]) != 0) ? tensors[inputs[1]] : nullptr;
        nvinfer1::Dims dims            = data->getDimensions();
        CHECK_ASSERT(data != nullptr && indices != nullptr, "get gather input tensor topo order error\n");
        if(axis < 0)
        {
            axis = dims.nbDims + axis;
            CHECK_ASSERT(axis >= 0, "axis value wrong\n");
        }        
        nvinfer1::IGatherLayer* gather = network->addGather(*data, *indices, axis);
        CHECK_ASSERT(gather, "create gather node fail\n");
        return gather;
    }
}