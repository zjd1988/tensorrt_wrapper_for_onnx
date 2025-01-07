/********************************************
 * Filename: reduce_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_info/reduce_node_info.hpp"
#include "node_info/node_info_creator.hpp"

namespace TENSORRT_WRAPPER
{

    // Reduce Node
    ReduceNodeInfo::ReduceNodeInfo() : NodeInfo("Reduce")
    {
        m_axes.clear();
        m_keepdims = 0;
    }

    bool ReduceNodeInfo::parseNodeAttributesFromJson(const Json::Value& root)
    {
        // check contain attributes
        if (!value.isMember("attributes"))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} not contain attributes", m_type, m_name);
            return false;
        }

        // parse node attributes
        auto attr = root["attributes"];
        if (false == getValue<std::vector<int>>(attr, "axes", m_axes, true, {}) || 
            false == getValue<int>(attr, "keepdims", m_keepdims, true, 0))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} parse attributes fail", m_type, m_name);
            return false;
        }
        return true;
    }

    void ReduceNodeInfo::printNodeAttributeInfo()
    {
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "Attribute as follows:");
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "axes is: {}", spdlog::fmt::join(m_axes, ","));
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "keepdims is: {}", m_keepdims);
        return;
    }

    class ReduceNodeInfoCreator : public NodeInfoCreator
    {
    public:
        virtual NodeInfo* onCreate(const std::string sub_type, const Json::Value& root) const override 
        {
            std::unique_ptr<NodeInfo> node_info(new ReduceNodeInfo());
            if (nullptr == node_info.get())
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "malloc node info fail for {}", root.toStyledString());
                return nullptr;
            }
            return node_info->parseNodeInfoFromJson(sub_type, root) ? node_info.release() : nullptr;
        }
    };

    void registerReduceNodeInfoCreator()
    {
        insertNodeInfoCreator("Reduce", new ReduceNodeInfoCreator);
        return;
    }

} // namespace TENSORRT_WRAPPER