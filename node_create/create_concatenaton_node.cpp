#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_concat_node.hpp"
#include "concatenation_node_info.hpp"

namespace tensorrtInference
{
    nvinfer1::ILayer* createConcatenationNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto concatenationNodeInfo = (ConcatenationNodeInfo*)nodeConfInfo;
        auto inputs = concatenationNodeInfo->getInputs();
        auto axis    = concatenationNodeInfo->getAxis();
        std::vector<nvinfer1::ITensor*> inputTensors;
        for(int i = 0; i < inputs.size(); i++) {
            nvinfer1::ITensor* inputTensor = (tensors.count(inputs[0]) != 0) ? tensors[inputs[0]] : nullptr;
            CHECK_ASSERT(inputTensor != nullptr, "get concatenation input %d tensor fail, topo order error\n", i);
            inputTensors.push_back(inputTensor);
        }
        
        for(int i = 0; i < axes.size(); i++)
        {
            if(axes[i] < 0)
            {
                axes[i] = dims.nbDims + axes[i];
                CHECK_ASSERT(axes[i] >= 0, "axes[%d] value wrong: %d\n", i, axes[i]);
            }
        }
        nvinfer1::IConcatenationLayer* concat = network->addConcatenation(inputTensors.data(), inputTensors.size());
        CHECK_ASSERT(concat, "create concatenation node fail\n");
        return concat;
    }
} //tensorrtInference