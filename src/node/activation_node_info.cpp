/********************************************
 * Filename: activation_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/activation_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Activation Node
    ActivationNodeInfo::ActivationNodeInfo()
    {
        m_alpha = 1.0f;
        m_beta = 0.0f;
        setNodeType("Activation");
        setNodeSubType("");
    }

    bool ActivationNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        // set node sub type
        setNodeSubType(type);

        // parse node inputs and check inputs size
        auto input_size = root["inputs"].size();
        if (3 <= input_size)
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_ERROR, "Activation node get {} inputs, expect less than 3 inputs", input_size);
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
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_ERROR, "Activation node get {} outputs, expect 1 outputs", output_size);
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
            if(elem.compare("alpha") == 0)
            {
                auto size = attr[elem].size();
                if (1 != size)
                {
                    TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_ERROR, "Activation node's alpha get {} elements, expect 1 element", size);
                    return false;
                }
                m_alpha = attr[elem][0].asFloat();
            }
            else if(elem.compare("beta") == 0)
            {
                auto size = attr[elem].size();
                if (1 != size)
                {
                    TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_ERROR, "Activation node's beta get {} elements, expect 1 element", size);
                    return false;
                }
                m_beta = attr[elem][0].asFloat();
            }
            else
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_WARN, "current Activation node not support {}", elem);
            }
        }
        return true;
    }

    void ActivationNodeInfo::printNodeInfo()
    {
        NodeInfo::printNodeInfo();
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_INFO, "node attribute as follows:");
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_INFO, "alpha: {}", m_alpha);
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_INFO, "beta: {}", m_beta);
        return;
    }

} // namespace TENSORRT_WRAPPER