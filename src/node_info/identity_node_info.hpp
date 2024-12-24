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
        ~IdentityNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        int getDataType() { return dataType; }

    private:
        int dataType;
    };

} // namespace TENSORRT_WRAPPER