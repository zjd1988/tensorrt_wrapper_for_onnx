/********************************************
 * Filename: concatenation_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/concatenation_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Concatenation Node
    ConcatenationNodeInfo::ConcatenationNodeInfo()
    {
        m_axis = 0;
        setNodeType("Concatenation");
        setNodeSubType("");
    }

    bool ConcatenationNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto input_size = root["inputs"].size();
        CHECK_ASSERT(input_size >= 1, "Concatenation node must have larger than 1 inputs\n");
        for(int i = 0; i < input_size; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto output_size = root["outputs"].size();
        CHECK_ASSERT(output_size == 1, "Concatenation node must have 1 output\n");
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
                CHECK_ASSERT(size == 1, "Concatenation node's axis must have 1 element\n");
                m_axis = attr[elem][0].asInt();
            }
            else
            {
                LOG("currnet Concatenation node not support %s \n", elem.c_str());
            }
        }
        return true;
    }

    void ConcatenationNodeInfo::printNodeInfo()
    {
        NodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----axes is : %d\n", m_axis);
    }

} // namespace TENSORRT_WRAPPER