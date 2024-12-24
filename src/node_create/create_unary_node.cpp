#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_unary_node.hpp"

namespace TENSORRT_WRAPPER
{
    nvinfer1::ILayer* createUnaryNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto subType = node_info->getNodeSubType();
        nvinfer1::UnaryOperation operation;
        //Sqrt Reciprocal Abs
        if(subType.compare("Sqrt") == 0) {
            operation = nvinfer1::UnaryOperation::kSQRT;
        }
        else if(subType.compare("Reciprocal") == 0) {
            operation = nvinfer1::UnaryOperation::kRECIP;
        }
        else if(subType.compare("Abs") == 0) {
            operation = nvinfer1::UnaryOperation::kABS;
        }
        else if(subType.compare("Exp") == 0) {
            operation = nvinfer1::UnaryOperation::kEXP;
        }        
        else {
            LOG("Current not support unary operation(%s) \n", subType);
            return nullptr;
        }
        auto inputs = node_info->getInputs();
        nvinfer1::ITensor* inputTensors = tensors[inputs[0]];
        nvinfer1::IUnaryLayer* unary = network->addUnary(*inputTensors, operation);
        CHECK_ASSERT(unary, "create unary node fail\n");
        return unary;
    }
}