/********************************************
 * Filename: gather_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class GatherNodeInfo : public NodeInfo
    {
    public:
        GatherNodeInfo();
        ~GatherNodeInfo() = default;
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        int getAxis() { return m_axis; }

    private:
        int                            m_axis;
    };

} // namespace TENSORRT_WRAPPER