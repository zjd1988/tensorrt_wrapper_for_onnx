
#ifndef __PADDING_NODE_INFO_HPP__
#define __PADDING_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class PaddingNodeInfo : public nodeInfo
    {
    public:
        PaddingNodeInfo();
        ~PaddingNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        std::string getMode() {return mode;}
        std::vector<int> getPads() {return pads;}
        float getFloatValue() {return floatValue;}
        int getIntValue() {return intValue;}
    private:
        std::string mode;
        std::vector<int> pads;
        float floatValue;
        int intValue;
    };
} //tensorrtInference
#endif // __PADDING_NODE_INFO_HPP__