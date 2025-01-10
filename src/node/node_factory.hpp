/********************************************
// Filename: node_factory.h
// Created by zjd1988 on 2024/12/27
// Description:

********************************************/
#pragma once
#include "node/node_creator.hpp"

namespace TENSORRT_WRAPPER
{

    /** node factory */
    class NodeFactory
    {
    public:
        static nvinfer1::ILayer* create(const std::string type, nvinfer1::INetworkDefinition* network, 
            std::map<std::string, nvinfer1::ITensor*>& tensors, NodeInfo* node_info, 
            std::map<std::string, WeightInfo>& weight_info);
    };

} // namespace TENSORRT_WRAPPER