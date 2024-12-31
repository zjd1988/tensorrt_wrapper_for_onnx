/********************************************
 * Filename: gather_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/utils.hpp"
#include "node_info/gather_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Gather Node
    GatherNodeInfo::GatherNodeInfo()
    {
        setNodeType("Gather");
        setNodeSubType("");
    }

    bool GatherNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto input_size = root["inputs"].size();
        CHECK_ASSERT(input_size == 2, "Gather node must have 2 inputs\n");
        for(int i = 0; i < input_size; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto output_size = root["outputs"].size();
        CHECK_ASSERT(output_size == 1, "Gather node must have 1 output\n");
        for(int i = 0; i < output_size; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("axis") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Gather node's axis must have 1 element\n");
                m_axis = attr[elem][0].asInt();
            }
            else
            {
                LOG("current Gather node not support %s \n", elem.c_str());
            }
        }
        return true;
    }

} // namespace TENSORRT_WRAPPER