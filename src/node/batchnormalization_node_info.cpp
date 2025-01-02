/********************************************
 * Filename: batchnormalization_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/batchnormalization_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // BatchNormalization Node
    BatchNormalizationNodeInfo::BatchNormalizationNodeInfo()
    {
        m_epsilon = 1e-05f;
        m_momentum = 0.9f;
        setNodeType("BatchNormalization");
        setNodeSubType("");
    }

    bool BatchNormalizationNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);

        // parse node inputs and check inputs size
        auto input_size = root["inputs"].size();
        if (5 != input_size)
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_ERROR, "BatchNormalization node get {} inputs, expect 5 inputs", input_size);
            return false;
        }
        for(int i = 0; i < input_size; i++)
        {
            addInput(root["inputs"][i].asString());
        }

        // parse node outputs and check outputs size
        auto output_size = root["outputs"].size();
        if (1 != output_size)
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_ERROR, "BatchNormalization node get {} outputs, expect 1 outputs", output_size);
            return false;
        }
        for(int i = 0; i < output_size; i++)
        {
            addOutput(root["outputs"][i].asString());
        }

        // parse node attributes
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("epsilon") == 0)
            {
                auto size = attr[elem].size();
                if (1 != size)
                {
                    TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_ERROR, "BatchNormalization node's epsilon get {} elements, expect 1 element", size);
                    return false;
                }
                m_epsilon = attr[elem][0].asFloat();
            }
            else if(elem.compare("momentum") == 0)
            {
                auto size = attr[elem].size();
                if (1 != size)
                {
                    TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_ERROR, "BatchNormalization node's momentum get {} elements, expect 1 element", size);
                    return false;
                }
                m_momentum = attr[elem][0].asFloat();
            }            
            else
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_WARN, "current BatchNormalization node not support {}", elem);
            }
        }
        return true;
    }

    void BatchNormalizationNodeInfo::printNodeInfo()
    {
        NodeInfo::printNodeInfo();
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_INFO, "node attribute as follows:\n");
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_INFO, "epsilon is : %f \n", m_epsilon);
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_INFO, "momentum is : %f \n", m_momentum);
        return;
    }

} // namespace TENSORRT_WRAPPER