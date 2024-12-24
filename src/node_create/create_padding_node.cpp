#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_padding_node.hpp"
#include "padding_node_info.hpp"

namespace TENSORRT_WRAPPER
{
    nvinfer1::ILayer* createPaddingNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        PaddingNodeInfo* paddingNodeInfo = (PaddingNodeInfo*)node_info;
        auto inputs = paddingNodeInfo->getInputs();
        CHECK_ASSERT(inputs.size() >= 1, "Padding node must have 1 inputs\n");
        nvinfer1::ITensor* inputTensor = nullptr;
        nvinfer1::IPaddingLayer* padding = nullptr;
        if(inputs.size() > 1)
        {
            inputTensor = tensors[inputs[0]];
            auto dims = inputTensor->getDimensions();
            auto shape = node_weight_info[inputs[1]].shape;
            CHECK_ASSERT(shape.size() == 1 && shape[0] == 2*dims.nbDims, "Pads value must be twice tensor dims\n");
            CHECK_ASSERT(4 == dims.nbDims, "Only 2D padding is currently supported.\n");
            auto pads = parseIntArrayValue(node_weight_info[inputs[1]].dataType, node_weight_info[inputs[1]].data,
                            node_weight_info[inputs[1]].byteCount, shape);
            padding = network->addPadding(*inputTensor, nvinfer1::DimsHW{pads[2], pads[3]}, nvinfer1::DimsHW{pads[6], pads[7]});
        }
        else
        {
            inputTensor = tensors[inputs[0]];
            auto dims = inputTensor->getDimensions();
            auto pads = paddingNodeInfo->getPads();
            CHECK_ASSERT(pads.size() == 2*dims.nbDims, "Pads value must be twice tensor dims\n");
            CHECK_ASSERT(4 == dims.nbDims, "Only 2D padding is currently supported.\n");
            auto mode = paddingNodeInfo->getMode();
            auto floatValue = paddingNodeInfo->getFloatValue();//to do
            auto intValue = paddingNodeInfo->getIntValue();
            CHECK_ASSERT(mode.compare("constant") == 0, "Pads only support constant and value must be 0\n");
            padding = network->addPadding(*inputTensor, nvinfer1::DimsHW{pads[2], pads[3]}, nvinfer1::DimsHW{pads[6], pads[7]});
        }

        CHECK_ASSERT(padding != nullptr, "create Padding node fail\n");
        return padding;
    }
}