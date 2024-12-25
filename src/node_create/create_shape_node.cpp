/********************************************
 * Filename: create_shape_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_create/create_node.hpp"
#include "node_create/create_shape_node.hpp"
#include "node_info/shape_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createShapeNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto inputs = node_info->getInputs();
        nvinfer1::ITensor* inputTensor = nullptr;
        inputTensor = (tensors.count(inputs[0]) != 0) ? tensors[inputs[0]] : nullptr;
        CHECK_ASSERT(inputTensor != nullptr, "topo order error\n");
        nvinfer1::IShapeLayer* shape = network->addShape(*inputTensor);
        CHECK_ASSERT(shape, "create shape node fail\n");
        return shape;
    }

} // namespace TENSORRT_WRAPPER