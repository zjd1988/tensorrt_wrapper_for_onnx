/********************************************
 * Filename: pooling_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_info/pooling_node_info.hpp"
#include "node_info/node_info_creator.hpp"

namespace TENSORRT_WRAPPER
{

    // Pooling Node
    PoolingNodeInfo::PoolingNodeInfo() : NodeInfo("Pooling")
    {
        m_kernel_shape.clear();
        m_pads.clear();
        m_strides.clear();
        m_ceil_mode = 0;
        m_count_include_pad = 0;
        m_auto_pad = "NOTSET";
    }

    bool PoolingNodeInfo::parseNodeAttributesFromJson(const Json::Value& root)
    {
        // check contain attributes
        if (!value.isMember("attributes"))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} not contain attributes", m_type, m_name);
            return false;
        }

        // parse node attributes
        auto attr = root["attributes"];
        if (false == getValue<std::vector<int>>(attr, "kernel_shape", m_kernel_shape, true, {}) || 
            false == getValue<std::vector<int>>(attr, "pads", m_pads, true, {}) || 
            false == getValue<std::vector<int>>(attr, "strides", m_strides, true, {}) || 
            false == getValue<int>(attr, "ceil_mode", m_ceil_mode, true, 0) || 
            false == getValue<int>(attr, "count_include_pad", m_count_include_pad, 0) || 
            false == getValue<std::string>(attr, "auto_pad", m_auto_pad, "NOTSET"))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} parse attributes fail", m_type, m_name);
            return false;
        }
        return true;
    }

    void PoolingNodeInfo::printNodeAttributeInfo()
    {
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "Attribute as follows:");
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "kernel_shape is: {}", spdlog::fmt::join(m_kernel_shape, ","));
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "pads is: {}", spdlog::fmt::join(m_pads, ","));
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "strides is: {}", spdlog::fmt::join(m_strides, ","));
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "ceil_mode is: {}", m_ceil_mode);
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "count_include_pad is: {}", m_count_include_pad);
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "auto_pad is: {}", m_auto_pad);
        return;
    }

    class PoolingNodeInfoCreator : public NodeInfoCreator
    {
    public:
        virtual NodeInfo* onCreate(const std::string sub_type, const Json::Value& root) const override 
        {
            std::unique_ptr<NodeInfo> node_info(new PoolingNodeInfo());
            if (nullptr == node_info.get())
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "malloc node info fail for {}", root.toStyledString());
                return nullptr;
            }
            return node_info->parseNodeInfoFromJson(sub_type, root) ? node_info.release() : nullptr;
        }
    };

    void registerPoolingNodeInfoCreator()
    {
        insertNodeInfoCreator("Pooling", new PoolingNodeInfoCreator);
        return;
    }

} // namespace TENSORRT_WRAPPER