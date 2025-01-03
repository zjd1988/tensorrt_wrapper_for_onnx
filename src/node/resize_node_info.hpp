/********************************************
 * Filename: resize_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class ResizeNodeInfo : public NodeInfo
    {
    public:
        ResizeNodeInfo();
        ~ResizeNodeInfo() = default;
        std::string getMode() { return m_mode; }

    protected:
        virtual bool parseNodeAttributesFromJson(const Json::Value& root) override;
        virtual bool verifyParsedNodeInfo() override;
        virtual void printNodeAttributeInfo() override;

    private:
        std::string                    m_mode;
    };

} // namespace TENSORRT_WRAPPER