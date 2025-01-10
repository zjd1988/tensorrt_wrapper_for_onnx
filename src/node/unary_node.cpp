/********************************************
 * Filename: unary_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "NvInfer.h"
#include "parser/graph_parser.hpp"
#include "node/node_creator.hpp"
#include "node_info/unary_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createUnaryNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& weight_info)
    {
        auto sub_type = node_info->getNodeSubType();
        nvinfer1::UnaryOperation operation;
        // Sqrt Reciprocal Abs
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
        nvinfer1::ITensor* input_tensors = tensors[inputs[0]];
        nvinfer1::IUnaryLayer* unary = network->addUnary(*input_tensors, operation);
        CHECK_ASSERT(unary, "create unary node fail\n");
        return unary;
    }

    class UnaryNodeCreator : public NodeCreator
    {
    public:
        virtual nvinfer1::ILayer* onCreate(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
            NodeInfo* node_info, std::map<std::string, WeightInfo>& weight_info) const override 
        {
            return createUnaryNode(network, tensors, node_info, weight_info);
        }
    };

    void registerUnaryNodeCreator()
    {
        insertNodeCreator("Unary", new UnaryNodeCreator);
    }

} // namespace TENSORRT_WRAPPER