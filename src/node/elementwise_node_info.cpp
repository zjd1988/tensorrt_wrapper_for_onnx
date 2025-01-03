/********************************************
 * Filename: elementwise_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/elementwise_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // ElementWise Node
    ElementWiseNodeInfo::ElementWiseNodeInfo() : NodeInfo("ElementWise")
    {
    }

    bool ElementWiseNodeInfo::verifyParsedNodeInfo()
    {
        // verify node inputs size
        auto input_size = m_inputs.size();
        if (2 != input_size)
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} get {} inputs, expect 2 inputs", 
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

} // namespace TENSORRT_WRAPPER