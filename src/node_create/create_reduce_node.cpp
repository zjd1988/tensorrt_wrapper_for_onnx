#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_reduce_node.hpp"
#include "reduce_node_info.hpp"
namespace TENSORRT_WRAPPER
{
    nvinfer1::ILayer* createReduceNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto reduceNodeInfo = (ReduceNodeInfo*)node_info;
        auto subType = reduceNodeInfo->getNodeSubType();
        nvinfer1::ReduceOperation operation;
        nvinfer1::IReduceLayer* reduce = nullptr;
        //ReduceSum
        if(subType.compare("ReduceSum") == 0) {
            operation = nvinfer1::ReduceOperation::kSUM;
        }
        else if(subType.compare("GlobalAveragePool") == 0) {
            operation = nvinfer1::ReduceOperation::kAVG;
        }
        else {
            LOG("Current not support unary operation(%s) \n", subType);
            return nullptr;
        }
        auto inputs = reduceNodeInfo->getInputs();
        nvinfer1::ITensor* inputTensors = tensors[inputs[0]];
        auto axesNodeConfig = reduceNodeInfo->getAxes();
        unsigned int axes = 0;
        for(int i = 0; i < axesNodeConfig.size(); i++)
        {
            axes |= (1 << axesNodeConfig[i]);
        }
        bool keepdims = reduceNodeInfo->getKeepdims();
        if(subType.compare("GlobalAveragePool") == 0){
            keepdims = true;
            nvinfer1::Dims dims = inputTensors->getDimensions();
            // Generate a bitmask of all 1s except the last 2 bits (N and C axes)
            axes = ((1 << dims.nbDims) - 1) & ~0b11;
            reduce = network->addReduce(*inputTensors, operation, axes, keepdims);
        }
        else
            reduce = network->addReduce(*inputTensors, operation, axes, keepdims);
        CHECK_ASSERT(reduce, "create reduce node fail\n");
        return reduce;
    }
}