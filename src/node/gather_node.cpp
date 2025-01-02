/********************************************
 * Filename: gather_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "NvInfer.h"
#include "parser/graph_parser.hpp"
#include "node/node_creator.hpp"
#include "node/gather_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createGatherNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto gather_node_info = (GatherNodeInfo*)node_info;
        auto inputs = gather_node_info->getInputs();
        int axis    = gather_node_info->getAxis();
        nvinfer1::ITensor* data = (tensors.count(inputs[0]) != 0) ? tensors[inputs[0]] : nullptr;
        nvinfer1::ITensor* indices = (tensors.count(inputs[1]) != 0) ? tensors[inputs[1]] : nullptr;
        nvinfer1::Dims dims = data->getDimensions();
        CHECK_ASSERT(data != nullptr && indices != nullptr, "get gather input tensor topo order error\n");
        if(axis < 0)
        {
            axis = dims.nbDims + axis;
            CHECK_ASSERT(axis >= 0, "axis value wrong\n");
        }
        nvinfer1::IGatherLayer* gather = network->addGather(*data, *indices, axis);
        CHECK_ASSERT(gather, "create gather node fail\n");
        return gather;
    }

    class GatherNodeCreator : public NodeCreator
    {
    public:
        virtual nvinfer1::ILayer* onCreate(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
            NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info) const override 
        {
            return createGatherNode(network, tensors, node_info, node_weight_info);
        }
    };

    void registerGatherNodeCreator()
    {
        insertNodeCreator("Gather", new GatherNodeCreator);
    }

} // namespace TENSORRT_WRAPPER