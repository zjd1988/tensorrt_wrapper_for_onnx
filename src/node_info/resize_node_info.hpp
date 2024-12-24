/********************************************
 * Filename: resize_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{
    class ResizeNodeInfo : public NodeInfo
    {
    public:
        ResizeNodeInfo();
        ~ResizeNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        std::string getMode(){return mode;}
    private:
        std::string mode;
    };
} // namespace TENSORRT_WRAPPER
#endif //__RESIZE_NODE_INFO_HPP__