#include "gather_node_info.hpp"
#include "utils.hpp"

namespace tensorrtInference
{
    // Gather Node
    GatherNodeInfo::GatherNodeInfo()
    {
        setNodeType("Gather");
        setSubNodeType("");
    }
    GatherNodeInfo::~GatherNodeInfo()
    {  
        
    }
    bool GatherNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setSubNodeType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize == 2, "Gather node must have 2 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Gather node must have 1 output\n");
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
                CHECK_ASSERT(size == 1, "Gather node's axis must have 1 element\n");
                axis = attr[elem][0].asInt();
            }
            else
            {
                LOG("current Gather node not support %s \n", elem.c_str());
            }
        }
        return true;
    }
} //tensorrtInference