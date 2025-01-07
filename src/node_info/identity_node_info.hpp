/********************************************
 * Filename: identity_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class IdentityNodeInfo : public NodeInfo
    {
    public:
        IdentityNodeInfo();
        ~IdentityNodeInfo() = default;
        int getDataType() { return m_data_type; }

    protected:
        virtual bool parseNodeAttributesFromJson(const Json::Value& root) override;
        virtual void printNodeAttributeInfo() override;

    private:
        int                            m_data_type;
    };

} // namespace TENSORRT_WRAPPER