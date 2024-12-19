
#ifndef __SHAPE_NODE_INFO_HPP__
#define __SHAPE_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class ShapeNodeInfo : public nodeInfo
    {
    public:
        ShapeNodeInfo();
        ~ShapeNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
    private:
        
    };
} // tensorrtInference
#endif //__SHAPE_NODE_INFO_HPP__