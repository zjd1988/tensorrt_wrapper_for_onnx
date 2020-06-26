
#ifndef __RESIZE_NODE_INFO_HPP__
#define __RESIZE_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class ResizeNodeInfo : public nodeInfo
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
} // tensorrtInference
#endif //__RESIZE_NODE_INFO_HPP__