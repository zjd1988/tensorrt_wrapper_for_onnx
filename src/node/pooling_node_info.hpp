/********************************************
 * Filename: pooling_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class PoolingNodeInfo : public NodeInfo
    {
    public:
        PoolingNodeInfo();
        ~PoolingNodeInfo() = default;
        std::vector<int> getKernelShape() { return m_kernel_shape; }
        std::vector<int> getPads() { return m_pads; }
        std::vector<int> getStrides() { return m_strides; }
        std::string getAutoPad() { return m_auto_pad; }
        bool getCeilMode() { return (1 == m_ceil_mode); }
        int getCountIncludePad() { return m_count_include_pad; }

    protected:
        virtual bool parseNodeAttributesFromJson(const Json::Value& root) override;
        virtual void printNodeAttributeInfo() override;

    private:
        int                            m_ceil_mode;
        int                            m_count_include_pad;
        std::string                    m_auto_pad;
        std::vector<int>               m_kernel_shape;
        std::vector<int>               m_pads;
        std::vector<int>               m_strides;
    };

} // namespace TENSORRT_WRAPPER