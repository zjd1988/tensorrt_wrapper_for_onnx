/********************************************
 * Filename: conv2d_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node/conv2d_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Conv2d Node
    Conv2dNodeInfo::Conv2dNodeInfo()
    {
        m_group = 0;
        m_kernel_shape.clear();
        m_pads.clear();
        m_strides.clear();
        m_dilation.clear();
        setNodeType("Conv2d");
        setNodeSubType("");
    }

    bool Conv2dNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto input_size = root["inputs"].size();
        CHECK_ASSERT(input_size >= 2, "conv2d node inputs size must larger than 2\n");
        for(int i = 0; i < input_size; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto output_size = root["outputs"].size();
        CHECK_ASSERT(output_size == 1, "conv2d node must have 1 output\n");
        for(int i = 0; i < output_size; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("kernel_shape") == 0)
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    m_kernel_shape.push_back(attr[elem][i].asInt());
                }
            }
            else if(elem.compare("dilations") == 0 )
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    m_dilation.push_back(attr[elem][i].asInt());
                }
            }
            else if(elem.compare("strides") == 0)
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    m_strides.push_back(attr[elem][i].asInt());
                }
            }
            else if(elem.compare("group") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size <= 1, "conv2d node's group must less than 1 element\n");
                if(size)
                    m_group = attr[elem][0].asInt();
            }
            else if(elem.compare("pads") == 0)
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    m_pads.push_back(attr[elem][i].asInt());
                }                
            }
            else
            {
                LOG("currnet conv2d node not support %s \n", elem.c_str());
            }
        }
        return true;
    }

    void Conv2dNodeInfo::printNodeInfo()
    {
        NodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----group is %d \n", group);
        LOG("----kernel_shape is : ");
        for(int i = 0; i < m_kernel_shape.size(); i++)
        {
            LOG("%d ", m_kernel_shape[i]);  
        }
        LOG("\n----pads is : ");
        for(int i = 0; i < m_pads.size(); i++)
        {
            LOG("%d ", m_pads[i]);  
        }
        LOG("\n----stride is : ");
        for(int i = 0; i < m_strides.size(); i++)
        {
            LOG("%d ", m_strides[i]);  
        }
        LOG("\n----dilation is : ");
        for(int i = 0; i < m_dilation.size(); i++)
        {
            LOG("%d ", m_dilation[i]);
        }
        LOG("\n");
    }

} // namespace TENSORRT_WRAPPER