/********************************************
 * Filename: conv2d_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class Conv2dNodeInfo : public NodeInfo
    {
    public:
        Conv2dNodeInfo();
        ~Conv2dNodeInfo() = default;
        int getGroup() { return m_group; }
        std::vector<int> getKernelShape() { return m_kernel_shape; }
        std::vector<int> getPads() { return m_pads; }
        std::vector<int> getStrides() { return m_strides; }
        std::vector<int> getDilation() { return m_dilations; }

    protected:
        virtual bool parseNodeAttributesFromJson(const Json::Value& root) override;
        virtual bool verifyParsedNodeInfo() override;
        virtual void printNodeAttributeInfo() override;

    private:
        int                            m_group;
        std::vector<int>               m_kernel_shape;
        std::vector<int>               m_pads;
        std::vector<int>               m_strides;
        std::vector<int>               m_dilations;
    };

} // namespace TENSORRT_WRAPPER