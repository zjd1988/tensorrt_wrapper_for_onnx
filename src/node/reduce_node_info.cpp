/********************************************
 * Filename: reduce_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/reduce_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Reduce Node
    ReduceNodeInfo::ReduceNodeInfo()
    {
        m_axes.clear();
        m_keepdims = 0;
        setNodeType("Reduce");
        setNodeSubType("");        
    }

    bool ReduceNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto input_size = root["inputs"].size();
        CHECK_ASSERT(input_size == 1, "Reduce node must have 1 inputs\n");
        for(int i = 0; i < input_size; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto output_size = root["outputs"].size();
        CHECK_ASSERT(output_size == 1, "Reduce node must have 1 output\n");
        for(int i = 0; i < output_size; i++)
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
                    m_axes.push_back(attr[elem][0].asInt());
            }
            else if(elem.compare("keepdims") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Reduce node's keepdims must have 1 element\n");
                m_keepdims = attr[elem][0].asInt();
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