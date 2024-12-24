/********************************************
 * Filename: unsqueeze_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/utils.hpp"
#include "node_info/unsqueeze_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Unsqueeze Node
    UnsqueezeNodeInfo::UnsqueezeNodeInfo()
    {
        axes.clear();
        setNodeType("Unsqueeze");
        setNodeSubType("");
    }

    UnsqueezeNodeInfo::~UnsqueezeNodeInfo()
    {
    }

    bool UnsqueezeNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
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
        NodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----axes is :  ");
        for(int i = 0; i < axes.size(); i++) {
            LOG("%d ", axes[i]);  
        }
        LOG("\n");
    }

} // namespace TENSORRT_WRAPPER