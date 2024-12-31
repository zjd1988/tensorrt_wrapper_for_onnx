/********************************************
 * Filename: batchnormalization_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/utils.hpp"
#include "node_info/batchnormalization_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // BatchNormalization Node
    BatchNormalizationNodeInfo::BatchNormalizationNodeInfo()
    {
        m_epsilon = 1e-05f;
        m_momentum = 0.9f;
        setNodeType("BatchNormalization");
        setNodeSubType("");
    }

    bool BatchNormalizationNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto input_size = root["inputs"].size();
        CHECK_ASSERT(input_size == 5, "BatchNormalization node must have 5 inputs\n");
        for(int i = 0; i < input_size; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto output_size = root["outputs"].size();
        CHECK_ASSERT(output_size == 1, "BatchNormalization node must have 1 output\n");
        for(int i = 0; i < output_size; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("m_epsilon") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "BatchNormalization node's m_epsilon must have 1 element\n");
                m_epsilon = attr[elem][0].asFloat();
            }
            else if(elem.compare("m_momentum") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "BatchNormalization node's m_momentum must have 1 element\n");
                m_momentum = attr[elem][0].asFloat();
            }            
            else
            {
                LOG("currnet BatchNormalization node not support %s \n", elem.c_str());
            }
        }
        return true;
    }

    void BatchNormalizationNodeInfo::printNodeInfo()
    {
        NodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----m_epsilon is : %f \n", m_epsilon);
        LOG("----m_momentum is : %f \n", m_momentum);
    }

} // namespace TENSORRT_WRAPPER