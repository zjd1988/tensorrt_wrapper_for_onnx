/********************************************
 * Filename: softmax_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class SoftmaxNodeInfo : public NodeInfo
    {
    public:
        SoftmaxNodeInfo();
        ~SoftmaxNodeInfo() = default;
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        int getAxis() { return m_axis; }

    private:
        int                            m_axis;
    };

} // namespace TENSORRT_WRAPPER