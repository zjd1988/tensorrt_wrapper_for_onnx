/********************************************
 * Filename: node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    bool NodeInfo::parseNodeInfoFromJson(const std::string type, const Json::Value& root)
    {
        return true;
    }

    void NodeInfo::printNodeInfo()
    {
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "################### NODE INFO ###################");
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "Node type: {} sub type: {}", m_type, m_sub_type);
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "Input tensor size: {} ", m_inputs.size());
        for(size_t i = 0; i < m_inputs.size(); i++)
        {
            LOG("input index: {} tensor: {}", i, m_inputs[i]);
        }
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "Output tensor size: {}", m_outputs.size());
        for(size_t i = 0; i < m_outputs.size(); i++)
        {
            LOG("output index: {} tensor: {}", i, m_outputs[i]);
        }
    }

} // namespace TENSORRT_WRAPPER