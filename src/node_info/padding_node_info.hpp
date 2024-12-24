/********************************************
 * Filename: padding_node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node_info/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    class PaddingNodeInfo : public NodeInfo
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

} // namespace TENSORRT_WRAPPER