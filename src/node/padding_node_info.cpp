/********************************************
 * Filename: padding_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/padding_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Padding Node
    PaddingNodeInfo::PaddingNodeInfo()
    {
        m_mode = "constant";
        m_pads.clear();
        m_float_value = 0.0f;
        m_int_value = 0;
        setNodeType("Padding");
        setNodeSubType("");
    }

    bool PaddingNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto input_size = root["inputs"].size();
        CHECK_ASSERT(input_size >= 1, "Padding node must have 2 inputs\n");
        for(int i = 0; i < input_size; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto output_size = root["outputs"].size();
        CHECK_ASSERT(output_size == 1, "Padding node must have 1 output\n");
        for(int i = 0; i < output_size; i++)
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
                m_mode = attr[elem][0].asString();
            }
            else if(elem.compare("pads") == 0)
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    m_pads.push_back(attr[elem][i].asInt());
                }
            }
            else if(elem.compare("value") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Padding node's value must have 1 element\n");
                m_float_value = attr[elem][0].asFloat();
            }            
            else
            {
                LOG("currnet Softmax node not support %s \n", elem.c_str());
            }
        }        
        return true;
    }

} // namespace TENSORRT_WRAPPER