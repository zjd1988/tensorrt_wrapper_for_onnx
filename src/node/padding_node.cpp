/********************************************
 * Filename: padding_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "NvInfer.h"
#include "parser/graph_parser.hpp"
#include "node/node_creator.hpp"
#include "node/padding_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createPaddingNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto padding_node_info = (PaddingNodeInfo*)node_info;
        auto inputs = padding_node_info->getInputs();
        CHECK_ASSERT(inputs.size() >= 1, "Padding node must have 1 inputs\n");
        nvinfer1::ITensor* input_tensor = nullptr;
        nvinfer1::IPaddingLayer* padding = nullptr;
        if(inputs.size() > 1)
        {
            input_tensor = tensors[inputs[0]];
            auto dims = input_tensor->getDimensions();
            auto shape = node_weight_info[inputs[1]].shape;
            CHECK_ASSERT(shape.size() == 1 && shape[0] == 2*dims.nbDims, "Pads value must be twice tensor dims\n");
            CHECK_ASSERT(4 == dims.nbDims, "Only 2D padding is currently supported.\n");
            auto pads = parseIntArrayValue(node_weight_info[inputs[1]].dataType, node_weight_info[inputs[1]].data,
                node_weight_info[inputs[1]].byteCount, shape);
            padding = network->addPadding(*input_tensor, nvinfer1::DimsHW{pads[2], pads[3]}, nvinfer1::DimsHW{pads[6], pads[7]});
        }
        else
        {
            input_tensor = tensors[inputs[0]];
            auto dims = input_tensor->getDimensions();
            auto pads = padding_node_info->getPads();
            CHECK_ASSERT(pads.size() == 2*dims.nbDims, "Pads value must be twice tensor dims\n");
            CHECK_ASSERT(4 == dims.nbDims, "Only 2D padding is currently supported.\n");
            auto mode = padding_node_info->getMode();
            auto floatValue = padding_node_info->getFloatValue();//to do
            auto intValue = padding_node_info->getIntValue();
            CHECK_ASSERT(mode.compare("constant") == 0, "Pads only support constant and value must be 0\n");
            padding = network->addPadding(*input_tensor, nvinfer1::DimsHW{pads[2], pads[3]}, nvinfer1::DimsHW{pads[6], pads[7]});
        }

        CHECK_ASSERT(padding != nullptr, "create Padding node fail\n");
        return padding;
    }

    class PaddingNodeCreator : public NodeCreator
    {
    public:
        virtual nvinfer1::ILayer* onCreate(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
            NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info) const override 
        {
            return createPaddingNode(network, tensors, node_info, node_weight_info);
        }
    };

    void registerPaddingNodeCreator()
    {
        insertNodeCreator("Padding", new PaddingNodeCreator);
    }

} // namespace TENSORRT_WRAPPER