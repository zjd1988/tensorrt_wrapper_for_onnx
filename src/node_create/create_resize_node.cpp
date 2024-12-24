#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_resize_node.hpp"
#include "resize_node_info.hpp"

namespace TENSORRT_WRAPPER
{
    nvinfer1::ILayer* createResizeNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto resizeNodeInfo = (ResizeNodeInfo*)node_info;
        auto inputs = resizeNodeInfo->getInputs();
        std::string mode = resizeNodeInfo->getMode();
        nvinfer1::ResizeMode resizeMode;
        if(mode.compare("nearest") == 0)
            resizeMode = nvinfer1::ResizeMode::kNEAREST;
        else if(mode.compare("linear") == 0)
            resizeMode = nvinfer1::ResizeMode::kLINEAR;
        else
            CHECK_ASSERT(0, "current only support nearest/linear resize mode\n");
        nvinfer1::ITensor* inputTensor = tensors[inputs[0]];
        nvinfer1::IResizeLayer* resize = network->addResize(*inputTensor);
        CHECK_ASSERT(resize, "create resize node fail\n");

        auto scaleWeights = node_weight_info[inputs[1]];
        auto scales = parseFloatArrayValue(scaleWeights.dataType, scaleWeights.data, scaleWeights.byteCount, scaleWeights.shape);
        resize->setScales(scales.data(), scales.size());
        resize->setResizeMode(resizeMode);
        return resize;
    }
}