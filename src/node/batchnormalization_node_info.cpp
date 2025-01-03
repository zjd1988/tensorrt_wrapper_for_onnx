/********************************************
 * Filename: batchnormalization_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/batchnormalization_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // BatchNormalization Node
    BatchNormalizationNodeInfo::BatchNormalizationNodeInfo() : NodeInfo("BatchNormalization")
    {
        m_epsilon = 1e-05f;
        m_momentum = 0.9f;
    }

    bool BatchNormalizationNodeInfo::parseNodeAttributesFromJson(const std::string type, const Json::Value& root)
    {
        // check contain attributes
        if (!value.isMember("attributes"))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} not contain attributes", m_type, m_name);
            return false;
        }

        // parse node attributes
        auto attr = root["attributes"];
        if (false == getValue<float>(attr, "epsilon", m_epsilon, true, 1e-05f) || 
            false == getValue<float>(attr, "momentum", m_momentum, true, 0.9f))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} parse attributes fail", m_type, m_name);
            return false;
        }
        return true;
    }

    bool BatchNormalizationNodeInfo::verifyParsedNodeInfo()
    {
        // verify node inputs size
        auto input_size = m_inputs.size();
        if (5 != input_size)
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} get {} inputs, expect 5 inputs", 
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

    void BatchNormalizationNodeInfo::printNodeAttributeInfo()
    {
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "node attribute as follows:");
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "epsilon is: {}", m_epsilon);
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "momentum is: {}", m_momentum);
        return;
    }

} // namespace TENSORRT_WRAPPER