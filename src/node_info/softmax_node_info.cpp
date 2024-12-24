/********************************************
 * Filename: softmax_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/utils.hpp"
#include "node_info/softmax_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Softmax Node
    SoftmaxNodeInfo::SoftmaxNodeInfo()
    {
        axis = 0;
        setNodeType("Softmax");
        setNodeSubType("");
    }

    SoftmaxNodeInfo::~SoftmaxNodeInfo()
    {
    }

    bool SoftmaxNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize == 1, "Softmax node must have 1 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Softmax node must have 1 output\n");
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
                CHECK_ASSERT(size == 1, "Softmax node's axis must have 1 element\n");
                axis = attr[elem][0].asInt();
            }
            else
            {
                LOG("currnet Softmax node not support %s \n", elem.c_str());
            }
        }
        return true;
    }

    void SoftmaxNodeInfo::printNodeInfo()
    {
        NodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----axis is : %d \n", axis);
    }

} // namespace TENSORRT_WRAPPER