/********************************************
 * Filename: identity_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/utils.hpp"
#include "node_info/identity_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Identity Node
    IdentityNodeInfo::IdentityNodeInfo()
    {
        setNodeType("Identity");
        setNodeSubType("");
        m_data_type = 0;
    }

    bool IdentityNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto input_size = root["inputs"].size();
        CHECK_ASSERT(input_size == 1, "Identity node must have 1 inputs\n");
        for(int i = 0; i < input_size; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto output_size = root["outputs"].size();
        CHECK_ASSERT(output_size == 1, "Identity node must have 1 output\n");
        for(int i = 0; i < output_size; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("to") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Identity node's to must have 1 element\n");
                m_data_type = attr[elem][0].asInt();
            }
            else
            {
                LOG("currnet Identity node not support %s \n", elem.c_str());
            }
        }
        return true;
    }

    void IdentityNodeInfo::printNodeInfo()
    {
        NodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----dataType is : %d \n", m_data_type);
    }

} // namespace TENSORRT_WRAPPER