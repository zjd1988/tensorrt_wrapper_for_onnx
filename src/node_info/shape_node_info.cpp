#include "shape_node_info.hpp"
#include "utils.hpp"

namespace tensorrtInference
{
    // Shape Node
    ShapeNodeInfo::ShapeNodeInfo()
    {
        setNodeType("Shape");
        setSubNodeType("");
    }
    ShapeNodeInfo::~ShapeNodeInfo()
    {  
        
    }
    bool ShapeNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setSubNodeType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize == 1, "Shape node must have 1 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Shape node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        return true;
    }
}