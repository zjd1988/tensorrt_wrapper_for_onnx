/********************************************
 * Filename: activation_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/activation_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Activation Node
    ActivationNodeInfo::ActivationNodeInfo() : NodeInfo("Activation")
    {
        m_alpha = 1.0f;
        m_beta = 0.0f;
    }

    bool ActivationNodeInfo::parseNodeAttributesFromJson(const std::string type, const Json::Value& root)
    {
        // check contain attributes
        if (!value.isMember("attributes"))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} not contain attributes", m_type, m_name);
            return false;
        }

        // parse node attributes
        auto attr = root["attributes"];
        if (false == getValue<float>(attr, "alpha", m_alpha, true, 1.0f) || 
            false == getValue<float>(attr, "beta", m_beta, true, 0.0f))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} parse attributes fail", m_type, m_name);
            return false;
        }
        return true;
    }

    bool ActivationNodeInfo::verifyParsedNodeInfo()
    {
        // verify node inputs size
        auto input_size = m_inputs.size();
        if (!(1 <= input_size && 3 >= input_size))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} get {} inputs, expect [1, 3] inputs", 
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
        
    void ActivationNodeInfo::printNodeAttributeInfo()
    {
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "node attribute as follows:");
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "alpha: {}", m_alpha);
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "beta: {}", m_beta);
        return;
    }

} // namespace TENSORRT_WRAPPER