/********************************************
 * Filename: unary_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/utils.hpp"
#include "node_info/unary_node_info.hpp"

namespace TENSORRT_WRAPPER
{
    // Unary Node
    UnaryNodeInfo::UnaryNodeInfo()
    {
        setNodeType("Unary");
        setNodeSubType("");
    }

    UnaryNodeInfo::~UnaryNodeInfo()
    {
    }

    bool UnaryNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize == 1, "Unary node must have 1 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Unary node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        return true;
    }

} // namespace TENSORRT_WRAPPER