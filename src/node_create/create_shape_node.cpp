#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_shape_node.hpp"

namespace tensorrtInference
{
    nvinfer1::ILayer* createShapeNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto inputs = nodeConfInfo->getInputs();
        nvinfer1::ITensor* inputTensor = nullptr;
        inputTensor = (tensors.count(inputs[0]) != 0) ? tensors[inputs[0]] : nullptr;
        CHECK_ASSERT(inputTensor != nullptr, "topo order error\n");
        nvinfer1::IShapeLayer* shape = network->addShape(*inputTensor);
        CHECK_ASSERT(shape, "create shape node fail\n");
        return shape;
    }
}