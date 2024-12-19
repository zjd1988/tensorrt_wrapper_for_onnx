
#ifndef __GATHER_NODE_INFO_HPP__
#define __GATHER_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class GatherNodeInfo : public nodeInfo
    {
    public:
        GatherNodeInfo();
        ~GatherNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        int getAxis() { return axis;}
    private:
        int axis;
    };
} // tensorrtInference
#endif //__GATHER_NODE_INFO_HPP__