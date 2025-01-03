/********************************************
 * Filename: softmax_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/softmax_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Softmax Node
    SoftmaxNodeInfo::SoftmaxNodeInfo() : NodeInfo("Softmax")
    {
        m_axis = 0;
    }

    bool SoftmaxNodeInfo::parseNodeAttributesFromJson(std::string type, Json::Value &root)
    {
        // check contain attributes
        if (!value.isMember("attributes"))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} not contain attributes", m_type, m_name);
            return false;
        }

        // parse node attributes
        auto attr = root["attributes"];
        if (false == getValue<int>(attr, "axis", m_axis, 0))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} parse attributes fail", m_type, m_name);
            return false;
        }
        return true;
    }

    void SoftmaxNodeInfo::printNodeAttributeInfo()
    {
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "node attribute is as follows:");
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "axis is: {}", m_axis);
        return;
    }

} // namespace TENSORRT_WRAPPER