/********************************************
 * Filename: create_activation_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_create/create_node.hpp"
#include "node_create/create_activation_node.hpp"
#include "node_info/activation_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createActivationNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto act_node_info = (ActivationNodeInfo*)node_info;
        auto sub_type = act_node_info->getNodeSubType();
        nvinfer1::ActivationType act_type;
        auto inputs = act_node_info->getInputs();
        nvinfer1::IActivationLayer* activation = nullptr;
        nvinfer1::ITensor* input_tensors = tensors[inputs[0]];
        //Clip kRELU
        if(0 == sub_type.compare("Clip"))
        {
            act_type = nvinfer1::ActivationType::kCLIP;
            int size = inputs.size();
            CHECK_ASSERT(size == 3, "Clip must have 3 inputs!\n");
            auto alpha = parseFloatArrayValue(node_weight_info[inputs[1]].dataType, node_weight_info[inputs[1]].data, 
                node_weight_info[inputs[1]].byteCount, node_weight_info[inputs[1]].shape);
            auto beta = parseFloatArrayValue(node_weight_info[inputs[2]].dataType, node_weight_info[inputs[2]].data, 
                node_weight_info[inputs[2]].byteCount, node_weight_info[inputs[2]].shape);
            activation = network->addActivation(*input_tensors, act_type);
            CHECK_ASSERT(activation, "create activation node fail, activation type is %s\n", sub_type.c_str());
            activation->setAlpha(alpha[0]);
            activation->setBeta(beta[0]);
        }
        else if(0 == sub_type.compare("Relu"))
        {
            act_type = nvinfer1::ActivationType::kRELU;
            activation = network->addActivation(*input_tensors, act_type);
            CHECK_ASSERT(activation, "create activation node fail, activation type is %s\n", sub_type.c_str());
        }
        else if(0 == sub_type.compare("LeakyRelu"))
        {
            act_type = nvinfer1::ActivationType::kLEAKY_RELU;
            auto alpha = act_node_info->getAlpha();
            activation = network->addActivation(*input_tensors, act_type);
            CHECK_ASSERT(activation, "create activation node fail, activation type is %s\n", sub_type.c_str());
            activation->setAlpha(alpha);
        }
        else if(0 == sub_type.compare("Sigmoid"))
        {
            act_type = nvinfer1::ActivationType::kSIGMOID;
            activation = network->addActivation(*input_tensors, act_type);
            CHECK_ASSERT(activation, "create activation node fail, activation type is %s\n", sub_type.c_str());
        }
        else if(0 == sub_type.compare("Softplus"))
        {
            float alpha = 1.0f;
            float beta = 1.0f;
            act_type = nvinfer1::ActivationType::kSOFTPLUS;
            activation = network->addActivation(*input_tensors, act_type);
            CHECK_ASSERT(activation, "create activation node fail, activation type is %s\n", sub_type.c_str());
            activation->setAlpha(alpha);
            activation->setBeta(beta);
        }
        else if(0 == sub_type.compare("Tanh"))
        {
            act_type = nvinfer1::ActivationType::kTANH;
            activation = network->addActivation(*input_tensors, act_type);
            CHECK_ASSERT(activation, "create activation node fail, activation type is %s\n", sub_type.c_str());
        }
        else
        {
            LOG("Current not support activation type(%s) \n", sub_type);
            return nullptr;
        }

        return activation;
    }

} // namespace TENSORRT_WRAPPER