/********************************************
 * Filename: identity_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class IdentityNodeInfo : public NodeInfo
    {
    public:
        IdentityNodeInfo();
        ~IdentityNodeInfo() = default;
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        int getDataType() { return m_data_type; }

    private:
        int                            m_data_type;
    };

} // namespace TENSORRT_WRAPPER