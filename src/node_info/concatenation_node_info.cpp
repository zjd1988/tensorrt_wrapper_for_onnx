/********************************************
 * Filename: concatenation_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/utils.hpp"
#include "node_info/concatenation_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Concatenation Node
    ConcatenationNodeInfo::ConcatenationNodeInfo()
    {
        m_axis = 0;
        setNodeType("Concatenation");
        setNodeSubType("");
    }

    ConcatenationNodeInfo::~ConcatenationNodeInfo()
    {
    }

    bool ConcatenationNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize >= 1, "Concatenation node must have larger than 1 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Concatenation node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
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
                axis = attr[elem][0].asInt();
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