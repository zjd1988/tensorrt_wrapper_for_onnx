/********************************************
 * Filename: gemm_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_info/gemm_node_info.hpp"
#include "node_info/node_info_creator.hpp"

namespace TENSORRT_WRAPPER
{

    // Gemm Node
    GemmNodeInfo::GemmNodeInfo() : NodeInfo("Gemm")
    {
        m_alpha = 1.0f;
        m_beta = 1.0f;
        m_transA = 0;
        m_transB = 0;
    }

    bool GemmNodeInfo::parseNodeAttributesFromJson(const Json::Value& root)
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
            false == getValue<float>(attr, "beta", m_beta, true, 1.0f) || 
            false == getValue<int>(attr, "transA", m_transA, true, 0) || 
            false == getValue<int>(attr, "transB", m_transB, true, 0))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} parse attributes fail", m_type, m_name);
            return false;
        }
        return true;
    }

    bool GemmNodeInfo::verifyParsedNodeInfo()
    {
        // verify node inputs size
        auto input_size = m_inputs.size();
        if (!(2 <= input_size && 3 >= input_size))
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "{} node:{} get {} inputs, expect [2, 3] inputs", 
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

    void GemmNodeInfo::printNodeAttributeInfo()
    {
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "Attribute as follows:");
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "alpha is: {}", m_alpha);
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "beta is: {}", m_beta);
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "transA is: {}", m_transA);
        TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_INFO, "transB is: {}", m_transB);
        return;
    }

    class GemmNodeInfoCreator : public NodeInfoCreator
    {
    public:
        virtual NodeInfo* onCreate(const std::string sub_type, const Json::Value& root) const override 
        {
            std::unique_ptr<NodeInfo> node_info(new GemmNodeInfo());
            if (nullptr == node_info.get())
            {
                TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "malloc node info fail for {}", root.toStyledString());
                return nullptr;
            }
            return node_info->parseNodeInfoFromJson(sub_type, root) ? node_info.release() : nullptr;
        }
    };

    void registerGemmNodeInfoCreator()
    {
        insertNodeInfoCreator("Gemm", new GemmNodeInfoCreator);
        return;
    }

} // namespace TENSORRT_WRAPPER