#include "batchnormalization_node_info.hpp"
#include "utils.hpp"

namespace TENSORRT_WRAPPER
{
    // BatchNormalization Node
    BatchNormalizationNodeInfo::BatchNormalizationNodeInfo()
    {
        epsilon = 1e-05f;
        momentum = 0.9f;
        setNodeType("BatchNormalization");
        setNodeSubType("");
    }
    BatchNormalizationNodeInfo::~BatchNormalizationNodeInfo()
    {
        epsilon = 1e-05f;
        momentum = 0.9f;
    }
    bool BatchNormalizationNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize == 5, "BatchNormalization node must have 5 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "BatchNormalization node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("epsilon") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "BatchNormalization node's epsilon must have 1 element\n");
                epsilon = attr[elem][0].asFloat();
            }
            else if(elem.compare("momentum") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "BatchNormalization node's momentum must have 1 element\n");
                momentum = attr[elem][0].asFloat();
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
        LOG("----epsilon is : %f \n", epsilon);
        LOG("----momentum is : %f \n", momentum);
    }
}