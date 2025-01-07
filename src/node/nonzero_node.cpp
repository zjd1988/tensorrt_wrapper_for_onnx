/********************************************
 * Filename: nonzero_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "NvInfer.h"
#include "parser/graph_parser.hpp"
#include "node/node_creator.hpp"
#include "node/nonzero_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createNonZeroNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto inputs = node_info->getInputs();
        nvinfer1::ITensor* input_tensor = tensors[inputs[0]];
        auto creator = getPluginRegistry()->getPluginCreator("NonZero_TRT", "1");
        auto pfc = creator->getFieldNames();
        nvinfer1::IPluginV2 *plugin_obj = creator->createPlugin("nonzero_plugin", pfc);
        auto nonzero = network->addPluginV2(&input_tensor, 1, *plugin_obj);
        CHECK_ASSERT(nonzero, "create nonzero node fail\n");
        return nonzero;
    }

    class NonZeroNodeCreator : public NodeCreator
    {
    public:
        virtual nvinfer1::ILayer* onCreate(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
            NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info) const override 
        {
            return createNonZeroNode(network, tensors, node_info, node_weight_info);
        }
    };

    void registerNonZeroNodeCreator()
    {
        insertNodeCreator("NonZero", new NonZeroNodeCreator);
    }

} // namespace TENSORRT_WRAPPER