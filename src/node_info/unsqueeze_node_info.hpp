/********************************************
 * Filename: unsqueeze_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class UnsqueezeNodeInfo : public NodeInfo
    {
    public:
        UnsqueezeNodeInfo();
        ~UnsqueezeNodeInfo() = default;
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        std::vector<int> getAxes() { return m_axes; }

    private:
        std::vector<int>               m_axes;
    };

} // namespace TENSORRT_WRAPPER