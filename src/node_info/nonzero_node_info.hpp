
/********************************************
 * Filename: nonzero_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class NonZeroNodeInfo : public NodeInfo
    {
    public:
        NonZeroNodeInfo();
        ~NonZeroNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
    };

} // namespace TENSORRT_WRAPPER