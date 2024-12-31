/********************************************
 * Filename: gemm_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/utils.hpp"
#include "node_info/gemm_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Gemm Node
    GemmNodeInfo::GemmNodeInfo()
    {
        m_alpha = 1.0f;
        m_beta = 1.0f;
        m_transA = 0;
        m_transB = 0;
        setNodeType("Gemm");
        setNodeSubType("");
    }

    bool GemmNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto input_size = root["inputs"].size();
        CHECK_ASSERT(input_size >= 2 && input_size <= 3, "Gemm node inputs must less than 3 and biger than 2\n");
        for(int i = 0; i < input_size; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto output_size = root["outputs"].size();
        CHECK_ASSERT(output_size == 1, "Gemm node must have 1 output\n");
        for(int i = 0; i < output_size; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("alpha") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Gemm node's alpha must have 1 element\n");
                m_alpha = attr[elem][0].asFloat();
            }
            else if(elem.compare("beta") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Gemm node's beta must have 1 element\n");
                m_beta = attr[elem][0].asFloat();
            }
            else if(elem.compare("transA") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Gemm node's transA must have 1 element\n");
                m_transA = attr[elem][0].asFloat();
            }
            else if(elem.compare("transB") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Gemm node's transB must have 1 element\n");
                m_transB = attr[elem][0].asFloat();
            }                        
            else
            {
                LOG("currnet Gemm node not support %s \n", elem.c_str());
            }
        }
        return true;
    }

    void GemmNodeInfo::printNodeInfo()
    {
        NodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----alpha is :  %f\n", m_alpha);
        LOG("----beta is :  %f\n", m_beta);
        LOG("----transA is :  %d\n", m_transA);
        LOG("----transB is :  %d\n", m_transB);
    }

} // namespace TENSORRT_WRAPPER