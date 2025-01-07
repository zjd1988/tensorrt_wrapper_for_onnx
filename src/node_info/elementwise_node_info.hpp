/********************************************
 * Filename: elementwise_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class ElementWiseNodeInfo : public NodeInfo
    {
    public:
        ElementWiseNodeInfo();
        ~ElementWiseNodeInfo() = default;

    protected:
        virtual bool verifyParsedNodeInfo() override;
    };

} // namespace TENSORRT_WRAPPER