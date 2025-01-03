/********************************************
 * Filename: unsqueeze_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/unsqueeze_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Unsqueeze Node
    UnsqueezeNodeInfo::UnsqueezeNodeInfo() : NodeInfo("Unsqueeze")
    {
        m_axes.clear();
    }

    bool UnsqueezeNodeInfo::parseNodeAttributesFromJson(const Json::Value& root)
    {
        // check contain attributes
        if (!value.isMember("attributes"))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} not contain attributes", m_type, m_name);
            return false;
        }

        // parse node attributes
        auto attr = root["attributes"];
        if (false == getValue<std::vector<int>>(attr, "axes", m_axes, {}))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} parse attributes fail", m_type, m_name);
            return false;
        }
        return true;
    }

    void UnsqueezeNodeInfo::printNodeAttributeInfo()
    {
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "node attribute is as follows:");
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "axes is: {}", spdlog::fmt::join(m_axes, ","));
        return;
    }

} // namespace TENSORRT_WRAPPER