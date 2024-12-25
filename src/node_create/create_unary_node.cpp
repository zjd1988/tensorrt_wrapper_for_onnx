/********************************************
 * Filename: create_unary_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_create/create_node.hpp"
#include "node_create/create_unary_node.hpp"
#include "node_info/unary_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createUnaryNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto sub_type = node_info->getNodeSubType();
        nvinfer1::UnaryOperation operation;
        //Sqrt Reciprocal Abs
        if(0 == sub_type.compare("Sqrt"))
        {
            operation = nvinfer1::UnaryOperation::kSQRT;
        }
        else if(0 == sub_type.compare("Reciprocal"))
        {
            operation = nvinfer1::UnaryOperation::kRECIP;
        }
        else if(0 == sub_type.compare("Abs"))
        {
            operation = nvinfer1::UnaryOperation::kABS;
        }
        else if(0 == sub_type.compare("Exp"))
        {
            operation = nvinfer1::UnaryOperation::kEXP;
        }        
        else
        {
            LOG("Current not support unary operation(%s) \n", sub_type);
            return nullptr;
        }
        auto inputs = node_info->getInputs();
        nvinfer1::ITensor* inputTensors = tensors[inputs[0]];
        nvinfer1::IUnaryLayer* unary = network->addUnary(*inputTensors, operation);
        CHECK_ASSERT(unary, "create unary node fail\n");
        return unary;
    }

} // namespace TENSORRT_WRAPPER