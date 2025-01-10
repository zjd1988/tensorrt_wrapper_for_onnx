/********************************************
 * Filename: softmax_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "NvInfer.h"
#include "parser/graph_parser.hpp"
#include "node/node_creator.hpp"
#include "node_info/softmax_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createSoftmaxNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& weight_info)
    {
        auto softmax_node_info = (SoftmaxNodeInfo*)node_info;
        auto inputs = softmax_node_info->getInputs();
        int axes = softmax_node_info->getAxis();
        // CHECK_ASSERT(axes >= 0, "axes only support positive\n");
        nvinfer1::ITensor* input_tensor = tensors[inputs[0]];
        nvinfer1::Dims dims = input_tensor->getDimensions();
        nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*input_tensor);
        CHECK_ASSERT(softmax, "create softmax node fail\n");
        if(axes < 0)
        {
            axes = dims.nbDims + axes;
            CHECK_ASSERT(axes >= 0, "axes value wrong\n");
        }
        softmax->setAxes(1 << axes);
        return softmax;
    }

    class SoftmaxNodeCreator : public NodeCreator
    {
    public:
        virtual nvinfer1::ILayer* onCreate(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
            NodeInfo* node_info, std::map<std::string, WeightInfo>& weight_info) const override 
        {
            return createSoftmaxNode(network, tensors, node_info, weight_info);
        }
    };

    void registerSoftmaxNodeCreator()
    {
        insertNodeCreator("Softmax", new SoftmaxNodeCreator);
    }

} // namespace TENSORRT_WRAPPER