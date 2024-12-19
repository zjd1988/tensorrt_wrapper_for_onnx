#include "gemm_node_info.hpp"
#include "utils.hpp"

namespace tensorrtInference
{
    // Gemm Node
    GemmNodeInfo::GemmNodeInfo()
    {
        alpha = 1.0f;
        beta = 1.0f;
        transA = 0;
        transB = 0;
        setNodeType("Gemm");
        setSubNodeType("");
    }
    GemmNodeInfo::~GemmNodeInfo()
    {
        alpha = 1.0f;
        beta = 1.0f;
        transA = 0;
        transB = 0;        
    }
    bool GemmNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setSubNodeType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize >= 2 && inputSize <= 3, "Gemm node inputs must less than 3 and biger than 2\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Gemm node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
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
                alpha = attr[elem][0].asFloat();
            }
            else if(elem.compare("beta") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Gemm node's beta must have 1 element\n");
                beta = attr[elem][0].asFloat();
            }
            else if(elem.compare("transA") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Gemm node's transA must have 1 element\n");
                transA = attr[elem][0].asFloat();
            }
            else if(elem.compare("transB") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Gemm node's transB must have 1 element\n");
                transB = attr[elem][0].asFloat();
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
        nodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----alpha is :  %f\n", alpha);
        LOG("----beta is :  %f\n", beta);
        LOG("----transA is :  %d\n", transA);
        LOG("----transB is :  %d\n", transB);
    }
}