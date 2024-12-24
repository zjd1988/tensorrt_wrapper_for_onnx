/********************************************
 * Filename: reduce_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/utils.hpp"
#include "node_info/reduce_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Reduce Node
    ReduceNodeInfo::ReduceNodeInfo()
    {
        axes.clear();
        keepdims = 0;
        setNodeType("Reduce");
        setNodeSubType("");        
    }

    ReduceNodeInfo::~ReduceNodeInfo()
    {
        axes.clear();
        keepdims = 0;
    }

    bool ReduceNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize == 1, "Reduce node must have 1 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Reduce node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("axes") == 0)
            {
                auto size = attr[elem].size();
                // CHECK_ASSERT(size == 1, "Reduce node's axes must have 1 element\n");
                for(int i = 0; i < size; i++)
                    axes.push_back(attr[elem][0].asInt());
            }
            else if(elem.compare("keepdims") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Reduce node's keepdims must have 1 element\n");
                keepdims = attr[elem][0].asInt();
            }            
            else
            {
                LOG("currnet Reduce node not support %s \n", elem.c_str());
            }
        }
        return true;
    }

    void ReduceNodeInfo::printNodeInfo()
    {
        NodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----axes is : %d \n", axes);
        LOG("----keepdims is : %d \n", keepdims);
    }

} // namespace TENSORRT_WRAPPER