/********************************************
 * Filename: unsqueeze_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/unsqueeze_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Unsqueeze Node
    UnsqueezeNodeInfo::UnsqueezeNodeInfo()
    {
        axes.clear();
        setNodeType("Unsqueeze");
        setNodeSubType("");
    }

    bool UnsqueezeNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto input_size = root["inputs"].size();
        CHECK_ASSERT(input_size == 1, "Unsqueeze node must have 1 inputs\n");
        for(int i = 0; i < input_size; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto output_size = root["outputs"].size();
        CHECK_ASSERT(output_size == 1, "Unsqueeze node must have 1 output\n");
        for(int i = 0; i < output_size; i++)
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
                    m_axes.push_back(attr[elem][i].asInt());
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
        for(int i = 0; i < m_axes.size(); i++)
        {
            LOG("%d ", m_axes[i]);
        }
        LOG("\n");
    }

} // namespace TENSORRT_WRAPPER