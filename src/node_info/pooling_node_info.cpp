/********************************************
 * Filename: pooling_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/utils.hpp"
#include "node_info/pooling_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Pooling Node
    PoolingNodeInfo::PoolingNodeInfo()
    {
        m_kernel_shape.clear();
        m_pads.clear();
        m_strides.clear();
        m_ceil_mode = 0;
        m_count_include_pad = 0;
        m_auto_pad = "NOTSET";
        setNodeType("Pooling");
        setNodeSubType("");
    }

    bool PoolingNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto input_size = root["inputs"].size();
        CHECK_ASSERT(input_size == 1, "Pooling node must have 1 inputs\n");
        for(int i = 0; i < input_size; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto output_size = root["outputs"].size();
        CHECK_ASSERT(output_size == 1, "Pooling node must have 1 output\n");
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
            else if(elem.compare("pads") == 0)
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    m_pads.push_back(attr[elem][i].asInt());
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
            else if(elem.compare("ceil_mode") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Pooling node's ceil_mode must have 1 element\n");
                m_ceil_mode = attr[elem][0].asInt();
            }
            else if(elem.compare("count_include_pad") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Pooling node's count_include_pad must have 1 element\n");
                m_count_include_pad = attr[elem][0].asInt();
            }
            else if(elem.compare("auto_pad") == 0)
            {
                m_auto_pad = attr[elem].asString();
            }
            else
            {
                LOG("currnet Pooling node not support %s attribute\n", elem.c_str());
            }
        }
        return true;
    }

    void PoolingNodeInfo::printNodeInfo()
    {
        NodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----kernelShape is : ");
        for(int i = 0; i < m_kernel_shape.size(); i++)
        {
            LOG("%d ", m_kernel_shape[i]);
        }
        LOG("\n----pads is : ");
        for(int i = 0; i < m_pads.size(); i++)
        {
            LOG("%d ", m_pads[i]);  
        }
        LOG("\n----strides is : ");
        for(int i = 0; i < m_strides.size(); i++)
        {
            LOG("%d ", m_strides[i]);  
        }
        LOG("\n----ceil_mode is : %d", m_ceil_mode);
        LOG("\n----count_include_pad is : %d", m_count_include_pad);
        LOG("\n----auto_pad is : %s", m_auto_pad.c_str());
        LOG("\n");
    }

} // namespace TENSORRT_WRAPPER