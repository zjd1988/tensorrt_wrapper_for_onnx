/********************************************
 * Filename: conv2d_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/conv2d_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Conv2d Node
    Conv2dNodeInfo::Conv2dNodeInfo() : NodeInfo("Conv2d")
    {
        m_group = 0;
        m_kernel_shape.clear();
        m_pads.clear();
        m_strides.clear();
        m_dilation.clear();
    }

    bool Conv2dNodeInfo::parseNodeAttributesFromJson(std::string type, Json::Value &root)
    {
        // check contain attributes
        if (!value.isMember("attributes"))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} not contain attributes", m_type, m_name);
            return false;
        }

        // parse node attributes
        auto attr = root["attributes"];
        if (false == getValue<int>(attr, "group", m_group, true, 0) || 
            false == getValue<std::vector<int>>(attr, "kernel_shape", m_kernel_shape, true, {}) || 
            false == getValue<std::vector<int>>(attr, "pads", m_pads, true, {}) || 
            false == getValue<std::vector<int>>(attr, "strides", m_strides, true, {}) || 
            false == getValue<std::vector<int>>(attr, "dilations", m_dilations, true, {}))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} parse attributes fail", m_type, m_name);
            return false;
        }
        return true;
    }

    bool Conv2dNodeInfo::verifyParsedNodeInfo()
    {
        // verify node inputs size
        auto input_size = m_inputs.size();
        if (!(2 <= input_size))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} get {} inputs, expect [2, ) inputs", 
                m_type, m_name, input_size);
            return false;
        }

        // verify node outputs size
        auto output_size = m_outputs.size();
        if (1 != output_size)
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} get {} outputs, expect 1 outputs",
                m_type, m_name, output_size);
            return false;
        }
        return true;
    }

    void Conv2dNodeInfo::printNodeAttributeInfo()
    {
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "node attribute is as follows:");
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "group is: {}", m_group);
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "kernel_shape is: {}", spdlog::fmt::join(m_kernel_shape, ","));
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "pads is: {}", spdlog::fmt::join(m_pads, ","));
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "strides is: {}", spdlog::fmt::join(m_strides, ","));
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "dilations is: {}", spdlog::fmt::join(m_dilations, ","));
        return;
    }

} // namespace TENSORRT_WRAPPER