#include "resize_node_info.hpp"
#include "utils.hpp"

namespace tensorrtInference
{
    // Resize Node
    ResizeNodeInfo::ResizeNodeInfo()
    {
        mode = "nearest";
        setNodeType("Resize");
        setSubNodeType("");
    }
    ResizeNodeInfo::~ResizeNodeInfo()
    {
        mode = "";
    }
    bool ResizeNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setSubNodeType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize > 1, "Resize node must larger than 1 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Resize node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
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
                mode = attr[elem][0].asString();
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
        nodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----mode is : %s \n", mode.c_str());
    }
}