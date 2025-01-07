/********************************************
 * Filename: concatenation_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class ConcatenationNodeInfo : public NodeInfo
    {
    public:
        ConcatenationNodeInfo();
        ~ConcatenationNodeInfo() = default;
        int getAxis() { return m_axis; }

    protected:
        virtual bool parseNodeAttributesFromJson(const Json::Value& root) override;
        virtual bool verifyParsedNodeInfo() override;
        virtual void printNodeAttributeInfo() override;

    private:
        int                            m_axis;
    };

} // namespace TENSORRT_WRAPPER