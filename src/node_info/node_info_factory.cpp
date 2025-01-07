/********************************************
// Filename: node_info_factory.cpp
// Created by zjd1988 on 2024/12/27
// Description:

********************************************/
#include "node_info/node_info_factory.hpp"

namespace TENSORRT_WRAPPER
{

    NodeInfo* NodeInfoFactory::create(const Json::Value& root)
    {
        std::string type;
        std::string sub_type
        // get type item
        if (!root.isMember("type"))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "cannot find type item in {}", root.toStyledString());
            return;
        }
        if (!root["type"].isString())
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "type item is not string in {}", root.toStyledString());
            return;
        }
        type = root["type"].asString();

        // get sub type item
        if (!root.isMember("sub_type"))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "cannot find sub_type item in {}", root.toStyledString());
            return;
        }
        if (!root["sub_type"].isString())
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "sub_type item is not string in {}", root.toStyledString());
            return;
        }
        sub_type = root["sub_type"].asString();

        // get node info creator
        auto creator = getNodeInfoCreator(type);
        if (nullptr == creator)
        {
            logRegisteredNodeCreator();
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "have no info creator for type: {}", type);
            return nullptr;
        }

        // create node inf
        auto node_info = creator->onCreate(sub_type, root);
        if (nullptr == node_info)
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node info parsed fail from {} ", type, root.toStyledString());
        }
        return node_info;
    }

} // namespace LM_INFER_ENGINE