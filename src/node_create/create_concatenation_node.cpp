/********************************************
 * Filename: create_concatenation_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_create/create_node.hpp"
#include "node_create/create_concatenation_node.hpp"
#include "node_info/concatenation_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createConcatenationNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto concat_node_info = (ConcatenationNodeInfo*)node_info;
        auto inputs = concat_node_info->getInputs();
        auto axis   = concat_node_info->getAxis();
        std::vector<nvinfer1::ITensor*> input_tensors;
        for(int i = 0; i < inputs.size(); i++)
        {
            nvinfer1::ITensor* tensor = (tensors.count(inputs[i]) != 0) ? tensors[inputs[i]] : nullptr;
            CHECK_ASSERT(tensor != nullptr, "get concatenation input %d tensor fail, topo order error\n", i);
            input_tensors.push_back(tensor);
        }
        auto dims = input_tensors[0]->getDimensions();
        if(0 > axis)
        {
            axis = dims.nbDims + axis;
            CHECK_ASSERT(axis >= 0, "axis value wrong: %d\n", axis);
        }
        nvinfer1::IConcatenationLayer* concat = network->addConcatenation(input_tensors.data(), input_tensors.size());
        CHECK_ASSERT(concat, "create concatenation node fail\n");
        concat->setAxis(axis);
        return concat;
    }

} // namespace TENSORRT_WRAPPER