#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_unsqueeze_node.hpp"
#include "unsqueeze_node_info.hpp"

namespace tensorrtInference
{
    nvinfer1::ILayer* createUnsqueezeNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto unsqueezeNodeInfo = (UnsqueezeNodeInfo*)nodeConfInfo;
        auto inputs = unsqueezeNodeInfo->getInputs();
        auto axes    = unsqueezeNodeInfo->getAxes();
        nvinfer1::ITensor* inputTensor = (tensors.count(inputs[0]) != 0) ? tensors[inputs[0]] : nullptr;
        CHECK_ASSERT(inputTensor != nullptr, "get gather input tensor topo order error\n");
        nvinfer1::Dims dims            = inputTensor->getDimensions();
        CHECK_ASSERT(dims.nbDims + axes.size() <= nvinfer1::Dims::MAX_DIMS, "after unsqueeze tensor's dim size is larger than MAX_DIMS\n");
        for(int i = 0; i < axes.size(); i++)
        {
            if(axes[i] < 0)
            {
                axes[i] = dims.nbDims + axes[i];
                CHECK_ASSERT(axes[i] >= 0, "axes[%d] value wrong: %d\n", i, axes[i]);
            }
        }
        std::vector<int> inputShape = dimsToVector(dims);
        inputShape.push_back(1);
        std::vector<int> subscripts(dims.nbDims);
        std::iota(subscripts.begin(), subscripts.end(), 0);
        for (int i = 0; i < axes.size(); i++)
        {
            subscripts.insert(subscripts.begin() + axes[i], dims.nbDims);
        }
        std::vector<int> newShape(subscripts.size());
        std::transform(subscripts.begin(), subscripts.end(), newShape.begin(), [&](int64_t i) {
            CHECK_ASSERT(0 <= i, "index should not less than 0\n");
            CHECK_ASSERT(static_cast<size_t>(i) < inputShape.size(), "index should not larger than %d\n", inputShape.size());
            return inputShape[i];
        });

        nvinfer1::Dims newDims = vectorToDims(newShape);
        nvinfer1::IShuffleLayer* unsqueeze = network->addShuffle(*inputTensor);
        CHECK_ASSERT(unsqueeze, "create unsqueeze node fail\n");
        unsqueeze->setReshapeDimensions(newDims);
        return unsqueeze;
    }
}