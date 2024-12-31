/********************************************
 * Filename: create_reduce_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_create/create_node.hpp"
#include "node_create/create_reduce_node.hpp"
#include "node_info/reduce_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createReduceNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto reduce_node_info = (ReduceNodeInfo*)node_info;
        auto subType = reduce_node_info->getNodeSubType();
        nvinfer1::ReduceOperation operation;
        nvinfer1::IReduceLayer* reduce = nullptr;
        //ReduceSum
        if(subType.compare("ReduceSum") == 0)
        {
            operation = nvinfer1::ReduceOperation::kSUM;
        }
        else if(subType.compare("GlobalAveragePool") == 0)
        {
            operation = nvinfer1::ReduceOperation::kAVG;
        }
        else
        {
            LOG("Current not support unary operation(%s) \n", subType);
            return nullptr;
        }
        auto inputs = reduce_node_info->getInputs();
        nvinfer1::ITensor* input_tensor = tensors[inputs[0]];
        auto axesNodeConfig = reduce_node_info->getAxes();
        unsigned int axes = 0;
        for(int i = 0; i < axesNodeConfig.size(); i++)
        {
            axes |= (1 << axesNodeConfig[i]);
        }
        bool keep_dims = reduce_node_info->getKeepdims();
        if(subType.compare("GlobalAveragePool") == 0)
        {
            keep_dims = true;
            nvinfer1::Dims dims = input_tensor->getDimensions();
            // Generate a bitmask of all 1s except the last 2 bits (N and C axes)
            axes = ((1 << dims.nbDims) - 1) & ~0b11;
            reduce = network->addReduce(*input_tensor, operation, axes, keep_dims);
        }
        else
            reduce = network->addReduce(*input_tensor, operation, axes, keep_dims);
        CHECK_ASSERT(reduce, "create reduce node fail\n");
        return reduce;
    }

} // namespace TENSORRT_WRAPPER