
#ifndef __UNSQUEEZE_NODE_INFO_HPP__
#define __UNSQUEEZE_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class UnsqueezeNodeInfo : public nodeInfo
    {
    public:
        UnsqueezeNodeInfo();
        ~UnsqueezeNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        std::vector<int> getAxes(){return axes;}
    private:
        std::vector<int> axes;
    };
} // tensorrtInference
#endif //__UNSQUEEZE_NODE_INFO_HPP__