#include "padding_node_info.hpp"
#include "utils.hpp"

namespace tensorrtInference
{
    // Padding Node
    PaddingNodeInfo::PaddingNodeInfo()
    {
        mode = "constant";
        pads.clear();
        floatValue = 0.0f;
        intValue = 0;
        setNodeType("Padding");
        setSubNodeType("");
    }
    PaddingNodeInfo::~PaddingNodeInfo()
    {  
        mode = "";
        pads.clear();
        floatValue = 0.0f;
        intValue = 0;
    }
    bool PaddingNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setSubNodeType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize >= 1, "Padding node must have 2 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Padding node must have 1 output\n");
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
                CHECK_ASSERT(size == 1, "Padding node's mode must have 1 element\n");
                mode = attr[elem][0].asString();
            }
            else if(elem.compare("pads") == 0)
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    pads.push_back(attr[elem][i].asInt());
                }
            }
            else if(elem.compare("value") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Padding node's value must have 1 element\n");
                floatValue = attr[elem][0].asFloat();
            }            
            else
            {
                LOG("currnet Softmax node not support %s \n", elem.c_str());
            }
        }        
        return true;
    }
}