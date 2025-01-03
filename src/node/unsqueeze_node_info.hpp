/********************************************
 * Filename: unsqueeze_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class UnsqueezeNodeInfo : public NodeInfo
    {
    public:
        UnsqueezeNodeInfo();
        ~UnsqueezeNodeInfo() = default;
        std::vector<int> getAxes() { return m_axes; }

    protected:
        virtual bool parseNodeAttributesFromJson(const Json::Value& root) override;
        virtual void printNodeAttributeInfo() override;

    private:
        std::vector<int>               m_axes;
    };

} // namespace TENSORRT_WRAPPER