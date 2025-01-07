/********************************************
 * Filename: node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    bool NodeInfo::parseNodeBaseInfoFromJson(const Json::Value& root)
    {
        if (false == getValue<std::string>(root, "name", m_name, false) || 
            false == getValue<std::vector<std::string>>(root, "inputs", m_inputs, false) || 
            false == getValue<std::vector<std::string>>(root, "outputs", m_outputs, false))
        {
            return false;
        }
        return true;
    }

    bool NodeInfo::parseNodeAttributesFromJson(const Json::Value& root)
    {
        return true;
    }

    bool NodeInfo::verifyParsedNodeInfo()
    {
        // verify node inputs size
        auto input_size = m_inputs.size();
        if (1 != input_size)
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} get {} inputs, expect 1 inputs", 
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

    bool NodeInfo::parseNodeInfoFromJson(const std::string type, const Json::Value& root)
    {
        // set node sub type
        m_sub_type = type;

        // parse node base info(name/inputs/outputs)
        // parse node attributes
        // verify parsed node info
        if (false == parseNodeBaseInfoFromJson(root) || 
            false == parseNodeAttributesFromJson(root) || 
            false == verifyParsedNodeInfo())
        {
            return false;
        }
        printNodeInfo();
        return true;
    }

    void NodeInfo::printNodeAttributeInfo()
    {
        return;
    }

    void NodeInfo::printNodeInfo()
    {
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "Node type: {} sub type: {}", m_type, m_sub_type);
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "Input tensor size: {} ", m_inputs.size());
        for(size_t i = 0; i < m_inputs.size(); i++)
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "index: {} tensor: {}", i, m_inputs[i]);
        }
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "Output tensor size: {}", m_outputs.size());
        for(size_t i = 0; i < m_outputs.size(); i++)
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "index: {} tensor: {}", i, m_outputs[i]);
        }
        printNodeAttributeInfo();
        return;
    }

} // namespace TENSORRT_WRAPPER