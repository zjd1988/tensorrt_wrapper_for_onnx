/********************************************
 * Filename: unsqueeze_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "NvInfer.h"
#include "parser/graph_parser.hpp"
#include "node/node_creator.hpp"
#include "node/unsqueeze_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createUnsqueezeNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto unsqueeze_node_info = (UnsqueezeNodeInfo*)node_info;
        auto inputs = unsqueeze_node_info->getInputs();
        auto axes = unsqueeze_node_info->getAxes();
        nvinfer1::ITensor* input_tensor = (tensors.count(inputs[0]) != 0) ? tensors[inputs[0]] : nullptr;
        CHECK_ASSERT(input_tensor != nullptr, "get gather input tensor topo order error\n");
        nvinfer1::Dims dims = input_tensor->getDimensions();
        CHECK_ASSERT(dims.nbDims + axes.size() <= nvinfer1::Dims::MAX_DIMS, "after unsqueeze tensor's dim size is larger than MAX_DIMS\n");
        for(int i = 0; i < axes.size(); i++)
        {
            if(axes[i] < 0)
            {
                axes[i] = dims.nbDims + axes[i];
                CHECK_ASSERT(axes[i] >= 0, "axes[%d] value wrong: %d\n", i, axes[i]);
            }
        }
        std::vector<int> input_shape = dimsToVector(dims);
        input_shape.push_back(1);
        std::vector<int> subscripts(dims.nbDims);
        std::iota(subscripts.begin(), subscripts.end(), 0);
        for (int i = 0; i < axes.size(); i++)
        {
            subscripts.insert(subscripts.begin() + axes[i], dims.nbDims);
        }
        std::vector<int> newShape(subscripts.size());
        std::transform(subscripts.begin(), subscripts.end(), newShape.begin(), [&](int64_t i) {
            CHECK_ASSERT(0 <= i, "index should not less than 0\n");
            CHECK_ASSERT(static_cast<size_t>(i) < input_shape.size(), "index should not larger than %d\n", input_shape.size());
            return input_shape[i];
        });

        nvinfer1::Dims new_dims = vectorToDims(newShape);
        nvinfer1::IShuffleLayer* unsqueeze = network->addShuffle(*input_tensor);
        CHECK_ASSERT(unsqueeze, "create unsqueeze node fail\n");
        unsqueeze->setReshapeDimensions(new_dims);
        return unsqueeze;
    }

    class UnsqueezeNodeCreator : public NodeCreator
    {
    public:
        virtual nvinfer1::ILayer* onCreate(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
            NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info) const override 
        {
            return createUnsqueezeNode(network, tensors, node_info, node_weight_info);
        }
    };

    void registerUnsqueezeNodeCreator()
    {
        insertNodeCreator("Unsqueeze", new UnsqueezeNodeCreator);
    }

} // namespace TENSORRT_WRAPPER