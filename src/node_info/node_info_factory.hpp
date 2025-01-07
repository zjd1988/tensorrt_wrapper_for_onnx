/********************************************
// Filename: node_info_factory.hpp
// Created by zjd1988 on 2024/12/27
// Description:

********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    /** node factory */
    class NodeInfoFactory
    {
    public:
        static NodeInfo* create(const Json::Value& root);
    };

} // namespace TENSORRT_WRAPPER