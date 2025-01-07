/********************************************
 * Filename: concatenation_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_info/concatenation_node_info.hpp"
#include "node_info/node_info_creator.hpp"

namespace TENSORRT_WRAPPER
{

    // Concatenation Node
    ConcatenationNodeInfo::ConcatenationNodeInfo() : NodeInfo("Concatenation")
    {
        m_axis = 0;
    }

    bool ConcatenationNodeInfo::parseNodeAttributesFromJson(const Json::Value& root)
    {
        // check contain attributes
        if (!value.isMember("attributes"))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} not contain attributes", m_type, m_name);
            return false;
        }

        // parse node attributes
        auto attr = root["attributes"];
        if (false == getValue<int>(attr, "axis", m_axis, true, 0))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} parse attributes fail", m_type, m_name);
            return false;
        }
        return true;
    }

    bool ConcatenationNodeInfo::verifyParsedNodeInfo()
    {
        // verify node inputs size
        auto input_size = m_inputs.size();
        if (!(1 <= input_size))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} get {} inputs, expect [1, ) inputs", 
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

    void ConcatenationNodeInfo::printNodeAttributeInfo()
    {
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "Attribute as follows:");
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "axes is: {}", m_axis);
        return;
    }

    class ConcatenationNodeInfoCreator : public NodeInfoCreator
    {
    public:
        virtual NodeInfo* onCreate(const std::string sub_type, const Json::Value& root) const override 
        {
            std::unique_ptr<NodeInfo> node_info(new ConcatenationNodeInfo());
            if (nullptr == node_info.get())
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "malloc node info fail for {}", root.toStyledString());
                return nullptr;
            }
            return node_info->parseNodeInfoFromJson(sub_type, root) ? node_info.release() : nullptr;
        }
    };

    void registerConcatenationNodeInfoCreator()
    {
        insertNodeInfoCreator("Concatenation", new ConcatenationNodeInfoCreator);
        return;
    }

} // namespace TENSORRT_WRAPPER