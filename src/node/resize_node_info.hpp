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
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        std::string getMode() { return m_mode; }
    private:
        std::string                    m_mode;
    };

} // namespace TENSORRT_WRAPPER