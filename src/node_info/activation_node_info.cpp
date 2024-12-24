/********************************************
 * Filename: activation_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/utils.hpp"
#include "node_info/activation_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Activation Node
    ActivationNodeInfo::ActivationNodeInfo()
    {
        setNodeType("Activation");
        setNodeSubType("");
    }

    ActivationNodeInfo::~ActivationNodeInfo()
    {
    }

    bool ActivationNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize <= 3, "Activation node must less than 3 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Activation node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("alpha") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Activation node's alpha must have 1 element\n");
                m_alpha = attr[elem][0].asFloat();
            }
            else if(elem.compare("beta") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Activation node's beta must have 1 element\n");
                m_alpha = attr[elem][0].asFloat();
            }
            else
            {
                LOG("currnet Activation node not support %s \n", elem.c_str());
            }
        }        
        return true;
    }

} // namespace TENSORRT_WRAPPER