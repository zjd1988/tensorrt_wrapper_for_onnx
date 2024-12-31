/********************************************
 * Filename: create_conv2d_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_create/create_node.hpp"
#include "node_create/create_conv2d_node.hpp"
#include "node_info/conv2d_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createConv2dNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto conv2d_node_info = (Conv2dNodeInfo *)node_info;
        auto inputs = conv2d_node_info->getInputs();
        CHECK_ASSERT(inputs.size() >= 2, "conv2d inputs must greater than 2\n");
        auto kernelShape = conv2d_node_info->getKernelShape();
        CHECK_ASSERT(kernelShape.size() == 2, "conv2d kernel shape must be 2\n");

        nvinfer1::IConvolutionLayer* conv2d = nullptr;
        nvinfer1::ITensor* input_tensor = tensors[inputs[0]];
        nvinfer1::DataType dataType = (node_weight_info[inputs[1]].dataType == OnnxDataType::FLOAT) ? 
            nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
        int weightEleCount = onnxDataTypeEleCount[node_weight_info[inputs[1]].dataType];
        CHECK_ASSERT(node_weight_info[inputs[1]].byteCount % weightEleCount == 0,
            "weights byte count shoud be mulptile of element byte count\n");
        nvinfer1::Weights wt{dataType, nullptr, 0};
        wt.type   = dataType;
        wt.values = node_weight_info[inputs[1]].data;
        wt.count  = node_weight_info[inputs[1]].byteCount / weightEleCount;
        int nbOutputMaps = node_weight_info[inputs[1]].shape[0];
        if(inputs.size() > 2)
        {
            int biasEleCount = onnxDataTypeEleCount[node_weight_info[inputs[2]].dataType];
            CHECK_ASSERT(node_weight_info[inputs[2]].byteCount % biasEleCount == 0,
                "bias byte count shoud be mulptile of element byte count\n");
            nvinfer1::Weights bias{dataType, nullptr, 0};
            bias.type = dataType;
            bias.values = node_weight_info[inputs[2]].data;
            bias.count = node_weight_info[inputs[2]].byteCount / biasEleCount;
            conv2d = network->addConvolution(*input_tensor, nbOutputMaps, nvinfer1::DimsHW{kernelShape[0], kernelShape[1]}, wt, bias);
        }
        else
        {
            nvinfer1::Weights bias{dataType, nullptr, 0};
            conv2d = network->addConvolution(*input_tensor, nbOutputMaps, nvinfer1::DimsHW{kernelShape[0], kernelShape[1]}, wt, bias);
        }
        CHECK_ASSERT(conv2d, "create conv2d node fail\n");
        auto group = conv2d_node_info->getGroup();
        auto strides = conv2d_node_info->getStrides();
        auto pads = conv2d_node_info->getPads();
        auto dilation = conv2d_node_info->getDilation();
        if(group > 1)
            conv2d->setNbGroups(group);
        if(strides.size())
            conv2d->setStride(nvinfer1::DimsHW{strides[0], strides[1]});
        if(pads.size() && pads.size() == 4)
        {
            CHECK_ASSERT(pads[0] == pads[1], "conv2d only support symmetric padding %d %d %d %d\n", pads[0], pads[1], pads[2], pads[3]);
            CHECK_ASSERT(pads[2] == pads[3], "conv2d only support symmetric padding %d %d %d %d\n", pads[0], pads[1], pads[2], pads[3]);
            CHECK_ASSERT(pads[0] == pads[2], "conv2d only support symmetric padding %d %d %d %d\n", pads[0], pads[1], pads[2], pads[3]);
            conv2d->setPadding(nvinfer1::DimsHW{pads[0], pads[1]});
        }

        if(dilation.size())
            conv2d->setDilation(nvinfer1::DimsHW{dilation[0], dilation[1]});
        
        return conv2d;
    }

} // namespace TENSORRT_WRAPPER