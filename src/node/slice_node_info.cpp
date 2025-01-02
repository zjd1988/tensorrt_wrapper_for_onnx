/********************************************
 * Filename: slice_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/slice_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Slice Node
    SliceNodeInfo::SliceNodeInfo()
    {
        setNodeType("Slice");
        setNodeSubType("");
    }

    bool SliceNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto input_size = root["inputs"].size();
        CHECK_ASSERT(input_size >= 3, "slice node must greate equal than 3 inputs\n");
        for(int i = 0; i < input_size; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto output_size = root["outputs"].size();
        CHECK_ASSERT(output_size == 1, "slice node must have 1 output\n");
        for(int i = 0; i < output_size; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        return true;
    }

} // namespace TENSORRT_WRAPPER