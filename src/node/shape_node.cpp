/********************************************
 * Filename: shape_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "NvInfer.h"
#include "parser/graph_parser.hpp"
#include "node/node_creator.hpp"
#include "node/shape_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createShapeNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto inputs = node_info->getInputs();
        nvinfer1::ITensor* input_tensor = nullptr;
        input_tensor = (tensors.count(inputs[0]) != 0) ? tensors[inputs[0]] : nullptr;
        CHECK_ASSERT(input_tensor != nullptr, "topo order error\n");
        nvinfer1::IShapeLayer* shape = network->addShape(*input_tensor);
        CHECK_ASSERT(shape, "create shape node fail\n");
        return shape;
    }

    class ShapeNodeCreator : public NodeCreator
    {
    public:
        virtual nvinfer1::ILayer* onCreate(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
            NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info) const override 
        {
            return createShapeNode(network, tensors, node_info, node_weight_info);
        }
    };

    void registerShapeNodeCreator()
    {
        insertNodeCreator("Shape", new ShapeNodeCreator);
    }

} // namespace TENSORRT_WRAPPER