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
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        std::string getMode() { return m_mode; }
        std::vector<int> getPads() { return m_pads; }
        float getFloatValue() { return m_float_value; }
        int getIntValue() { return m_int_value; }

    private:
        std::string                    m_mode;
        std::vector<int>               m_pads;
        float                          m_float_value;
        int                            m_int_value;
    };

} // namespace TENSORRT_WRAPPER