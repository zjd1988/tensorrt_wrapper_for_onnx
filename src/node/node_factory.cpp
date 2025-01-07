/********************************************
// Filename: node_factory.cpp
// Created by zjd1988 on 2024/12/27
// Description:

********************************************/
#include "node/node_factory.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* NodeFactory::create(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto node_type = node_info->getNodeType();
        auto creator = getNodeCreator(node_type);
        if (nullptr == creator)
        {
            logRegisteredNodeCreator();
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "have no creator for type: {}", node_type);
            return nullptr;
        }
        auto node = creator->onCreate(network, tensors, node_info, node_weight_info);
        if (nullptr == node)
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "create {} node failed, creator return nullptr", node_type);
        }
        return node;
    }

} // namespace LM_INFER_ENGINE