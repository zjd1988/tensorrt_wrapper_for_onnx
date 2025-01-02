/********************************************
 * Filename: resize_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/resize_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Resize Node
    ResizeNodeInfo::ResizeNodeInfo()
    {
        m_mode = "nearest";
        setNodeType("Resize");
        setNodeSubType("");
    }

    bool ResizeNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto input_size = root["inputs"].size();
        CHECK_ASSERT(input_size > 1, "Resize node must larger than 1 inputs\n");
        for(int i = 0; i < input_size; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto output_size = root["outputs"].size();
        CHECK_ASSERT(output_size == 1, "Resize node must have 1 output\n");
        for(int i = 0; i < output_size; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("mode") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Resize node's mode must have 1 element\n");
                m_mode = attr[elem][0].asString();
            }
            else
            {
                LOG("currnet Resize node not support %s \n", elem.c_str());
            }
        }
        return true;
    }

    void ResizeNodeInfo::printNodeInfo()
    {
        NodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----mode is : %s \n", mode.c_str());
    }

} // namespace TENSORRT_WRAPPER