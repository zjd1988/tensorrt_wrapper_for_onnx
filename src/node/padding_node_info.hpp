/********************************************
 * Filename: padding_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class PaddingNodeInfo : public NodeInfo
    {
    public:
        PaddingNodeInfo();
        ~PaddingNodeInfo() = default;
        std::string getMode() { return m_mode; }
        std::vector<int> getPads() { return m_pads; }
        float getFloatValue() { return m_value; }
        int getIntValue() { return (int)m_value; }

    protected:
        virtual bool parseNodeAttributesFromJson(const Json::Value& root) override;
        virtual bool verifyParsedNodeInfo() override;
        virtual void printNodeAttributeInfo() override;

    private:
        std::string                    m_mode;
        std::vector<int>               m_pads;
        float                          m_value;
    };

} // namespace TENSORRT_WRAPPER