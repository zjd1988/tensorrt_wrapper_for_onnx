/********************************************
 * Filename: activation_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "NvInfer.h"
#include "parser/graph_parser.hpp"
#include "node/node_creator.hpp"
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
        if("Clip" == sub_type)
        {
            act_type = nvinfer1::ActivationType::kCLIP;
            int size = inputs.size();
            CHECK_ASSERT(size == 3, "Clip expect 3 inputs!");
            auto alpha = parseFloatArrayValue(node_weight_info[inputs[1]].dataType, node_weight_info[inputs[1]].data, 
                node_weight_info[inputs[1]].byteCount, node_weight_info[inputs[1]].shape);
            auto beta = parseFloatArrayValue(node_weight_info[inputs[2]].dataType, node_weight_info[inputs[2]].data, 
                node_weight_info[inputs[2]].byteCount, node_weight_info[inputs[2]].shape);
            activation = network->addActivation(*input_tensors, act_type);
            CHECK_ASSERT(nullptr != activation, "create activation node fail, activation type is {}", sub_type);
            activation->setAlpha(alpha[0]);
            activation->setBeta(beta[0]);
        }
        else if("Relu" == sub_type)
        {
            act_type = nvinfer1::ActivationType::kRELU;
            activation = network->addActivation(*input_tensors, act_type);
            CHECK_ASSERT(nullptr != activation, "create activation node fail, activation type is {}", sub_type);
        }
        else if("LeakyRelu" == sub_type)
        {
            act_type = nvinfer1::ActivationType::kLEAKY_RELU;
            auto alpha = act_node_info->getAlpha();
            activation = network->addActivation(*input_tensors, act_type);
            CHECK_ASSERT(nullptr != activation, "create activation node fail, activation type is {}", sub_type);
            activation->setAlpha(alpha);
        }
        else if("Sigmoid" == sub_type)
        {
            act_type = nvinfer1::ActivationType::kSIGMOID;
            activation = network->addActivation(*input_tensors, act_type);
            CHECK_ASSERT(nullptr != activation, "create activation node fail, activation type is {}", sub_type);
        }
        else if("Softplus" == sub_type)
        {
            float alpha = 1.0f;
            float beta = 1.0f;
            act_type = nvinfer1::ActivationType::kSOFTPLUS;
            activation = network->addActivation(*input_tensors, act_type);
            CHECK_ASSERT(nullptr != activation, "create activation node fail, activation type is {}", sub_type);
            activation->setAlpha(alpha);
            activation->setBeta(beta);
        }
        else if("Tanh" == sub_type)
        {
            act_type = nvinfer1::ActivationType::kTANH;
            activation = network->addActivation(*input_tensors, act_type);
            CHECK_ASSERT(nullptr != activation, "create activation node fail, activation type is {}", sub_type);
        }
        else
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "Current not support activation type: {}", sub_type);
            return nullptr;
        }

        return activation;
    }

    class ActivationNodeCreator : public NodeCreator
    {
    public:
        virtual nvinfer1::ILayer* onCreate(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
            NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info) const override 
        {
            return createActivationNode(network, tensors, node_info, node_weight_info);
        }
    };

    void registerActivationNodeCreator()
    {
        insertNodeCreator("Activation", new ActivationNodeCreator);
    }

} // namespace TENSORRT_WRAPPER