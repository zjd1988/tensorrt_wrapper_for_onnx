#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_nonzero_node.hpp"

namespace TENSORRT_WRAPPER
{
    nvinfer1::ILayer* createNonZeroNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto inputs = node_info->getInputs();
        nvinfer1::ITensor* inputTensor = tensors[inputs[0]];
        auto creator = getPluginRegistry()->getPluginCreator("NonZero_TRT", "1");
        auto pfc = creator->getFieldNames();
        nvinfer1::IPluginV2 *pluginObj = creator->createPlugin("nonzero_plugin", pfc);
        auto nonzero = network->addPluginV2(&inputTensor, 1, *pluginObj);
        CHECK_ASSERT(nonzero, "create nonzero node fail\n");
        return nonzero;
    }
}