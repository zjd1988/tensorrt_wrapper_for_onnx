/********************************************
 * Filename: shuffle_node_info.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/utils.hpp"
#include "node_info/shuffle_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    // Shuffle Node
    ShuffleNodeInfo::ShuffleNodeInfo()
    {
        setNodeType("Shuffle");
        setNodeSubType("");
        axis = 1;
        perm.clear();
    }

    ShuffleNodeInfo::~ShuffleNodeInfo()
    {
        axis = 1;
        perm.clear();
    }

    bool ShuffleNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setNodeSubType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize <= 2, "Shuffle node must less than 2 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Shuffle node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("perm") == 0)
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    perm.push_back(attr[elem][i].asInt());
                }
            }
            else if(elem.compare("axis") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Shuffle(Flatten) node's axis must have 1 element\n");
                axis = attr[elem][0].asInt();
            }            
            else
            {
                LOG("current Shuffle node not support %s \n", elem.c_str());
            }
        }
        return true;
    }

    void ShuffleNodeInfo::printNodeInfo()
    {
        NodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----perm is : ");
        for(int i = 0; i < perm.size(); i++) {
            LOG("%d ", perm[i]);
        }
        LOG("\n");
    }

} // namespace TENSORRT_WRAPPER