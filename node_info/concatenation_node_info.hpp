
#ifndef __CONCATENATION_NODE_INFO_HPP__
#define __CONCATENATION_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class ConcatenationNodeInfo : public nodeInfo
    {
    public:
        ConcatenationNodeInfo();
        ~ConcatenationNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        int getAxes(){return axis;}
    private:
        int axis;
    };
} // tensorrtInference
#endif //__CONCATENATION_NODE_INFO_HPP__