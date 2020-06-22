#include "concat_node_info.hpp"
#include "utils.hpp"

namespace tensorrtInference
{
    // Concat Node
    ConcatenationNodeInfo::ConcatenationNodeInfo()
    {
        axis = 0;
        setNodeType("Concat");
        setSubNodeType("");
    }
    ConcatenationNodeInfo::~ConcatenationNodeInfo()
    {
        axis = 0;
    }
    bool ConcatenationNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setSubNodeType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize == 1, "Concatenation node must have 1 inputs\n");
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
        nodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----axes is : %d\n", axis);
    }
} //tensorrtInference