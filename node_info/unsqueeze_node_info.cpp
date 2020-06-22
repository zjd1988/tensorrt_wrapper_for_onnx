#include "unsqueeze_node_info.hpp"
#include "utils.hpp"

namespace tensorrtInference
{
    // Unsqueeze Node
    UnsqueezeNodeInfo::UnsqueezeNodeInfo()
    {
        axes.clear();
        setNodeType("Unsqueeze");
        setSubNodeType("");
    }
    UnsqueezeNodeInfo::~UnsqueezeNodeInfo()
    {
        axes.clear();
    }
    bool UnsqueezeNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setSubNodeType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize == 1, "Unsqueeze node must have 1 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Unsqueeze node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("axes") == 0)
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    axes.push_back(attr[elem][i].asInt());
                }
            }
            else
            {
                LOG("currnet Unsqueeze node not support %s \n", elem.c_str());
            }
        }
        return true;
    }
    void UnsqueezeNodeInfo::printNodeInfo()
    {
        nodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----axes is :  ");
        for(int i = 0; i < axes.size(); i++) {
            LOG("%d ", axes[i]);  
        }
        LOG("\n");
    }
}