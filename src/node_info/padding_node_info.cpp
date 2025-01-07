/********************************************
 * Filename: padding_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_info/padding_node_info.hpp"
#include "node_info/node_info_creator.hpp"

namespace TENSORRT_WRAPPER
{

    // Padding Node
    PaddingNodeInfo::PaddingNodeInfo() : NodeInfo("Padding")
    {
        m_mode = "constant";
        m_pads.clear();
        m_value = 0.0f;
    }

    bool PaddingNodeInfo::parseNodeAttributesFromJson(const Json::Value& root)
    {
        // check contain attributes
        if (!value.isMember("attributes"))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} not contain attributes", m_type, m_name);
            return false;
        }

        // parse node attributes
        auto attr = root["attributes"];
        if (false == getValue<std::string>(attr, "mode", m_mode, true, "constant") || 
            false == getValue<std::vector<int>>(attr, "pads", m_pads, true, {}) || 
            false == getValue<float>(attr, "value", m_value, true, 0.0f))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} parse attributes fail", m_type, m_name);
            return false;
        }
        return true;
    }

    bool PaddingNodeInfo::verifyParsedNodeInfo()
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

    void PaddingNodeInfo::printNodeAttributeInfo()
    {
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "Attribute as follows:");
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "mode is: {}", m_mode);
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "pads is: {}", spdlog::fmt::join(m_pads, ","));
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "value is: {}", m_value);
        return;
    }

    class PaddingNodeInfoCreator : public NodeInfoCreator
    {
    public:
        virtual NodeInfo* onCreate(const std::string sub_type, const Json::Value& root) const override 
        {
            std::unique_ptr<NodeInfo> node_info(new PaddingNodeInfo());
            if (nullptr == node_info.get())
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "malloc node info fail for {}", root.toStyledString());
                return nullptr;
            }
            return node_info->parseNodeInfoFromJson(sub_type, root) ? node_info.release() : nullptr;
        }
    };

    void registerPaddingNodeInfoCreator()
    {
        insertNodeInfoCreator("Padding", new PaddingNodeInfoCreator);
        return;
    }

} // namespace TENSORRT_WRAPPER