#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "activation_node_info.hpp"
#include "create_activation_node.hpp"

namespace tensorrtInference
{
    nvinfer1::ILayer* createActivationNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        ActivationNodeInfo* activationNodeConfInfo = (ActivationNodeInfo*)nodeConfInfo;
        auto subType = activationNodeConfInfo->getSubNodeType();
        nvinfer1::ActivationType type;
        auto inputs = activationNodeConfInfo->getInputs();
        nvinfer1::IActivationLayer* activation = nullptr;
        nvinfer1::ITensor* inputTensors = tensors[inputs[0]];
        //Clip kRELU
        if(subType.compare("Clip") == 0) {
            type = nvinfer1::ActivationType::kCLIP;
            int size = inputs.size();
            CHECK_ASSERT(size == 3, "Clip must have 3 inputs!\n");
            auto alpha = parseFloatArrayValue(nodeWeightsInfo[inputs[1]].dataType, nodeWeightsInfo[inputs[1]].data, 
                            nodeWeightsInfo[inputs[1]].byteCount, nodeWeightsInfo[inputs[1]].shape);
            auto beta = parseFloatArrayValue(nodeWeightsInfo[inputs[2]].dataType, nodeWeightsInfo[inputs[2]].data, 
                            nodeWeightsInfo[inputs[2]].byteCount, nodeWeightsInfo[inputs[2]].shape);
            
            activation = network->addActivation(*inputTensors, type);
            CHECK_ASSERT(activation, "create activation node fail, activation type is %s\n", subType.c_str());
            activation->setAlpha(alpha[0]);
            activation->setBeta(beta[0]);
        }
        else if(subType.compare("Relu") == 0) {
            type = nvinfer1::ActivationType::kRELU;
            activation = network->addActivation(*inputTensors, type);
            CHECK_ASSERT(activation, "create activation node fail, activation type is %s\n", subType.c_str());
        }
        else if(subType.compare("LeakyRelu") == 0)
        {
            type = nvinfer1::ActivationType::kLEAKY_RELU;
            auto alpha = activationNodeConfInfo->getAlpha();
            activation = network->addActivation(*inputTensors, type);
            CHECK_ASSERT(activation, "create activation node fail, activation type is %s\n", subType.c_str());
            activation->setAlpha(alpha);
        }
        else {
            LOG("Current not support activation type(%s) \n", subType);
            return nullptr;
        }
        
        return activation;
    }
}