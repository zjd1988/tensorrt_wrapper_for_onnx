/********************************************
 * Filename: shape_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class ShapeNodeInfo : public NodeInfo
    {
    public:
        ShapeNodeInfo();
        ~ShapeNodeInfo() = default;
    };

} // namespace TENSORRT_WRAPPER