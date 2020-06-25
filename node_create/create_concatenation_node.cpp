#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_concatenation_node.hpp"
#include "concatenation_node_info.hpp"

namespace tensorrtInference
{
    nvinfer1::ILayer* createConcatenationNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto concatenationNodeInfo = (ConcatenationNodeInfo*)nodeConfInfo;
        auto inputs = concatenationNodeInfo->getInputs();
        auto axis   = concatenationNodeInfo->getAxis();
        std::vector<nvinfer1::ITensor*> inputTensors;
        for(int i = 0; i < inputs.size(); i++) {
            nvinfer1::ITensor* inputTensor = (tensors.count(inputs[i]) != 0) ? tensors[inputs[i]] : nullptr;
            CHECK_ASSERT(inputTensor != nullptr, "get concatenation input %d tensor fail, topo order error\n", i);
            inputTensors.push_back(inputTensor);
        }
        auto dims = inputTensors[0]->getDimensions();
        if(axis < 0)
        {
            axis = dims.nbDims + axis;
            CHECK_ASSERT(axis >= 0, "axis value wrong: %d\n", axis);
        }
        nvinfer1::IConcatenationLayer* concat = network->addConcatenation(inputTensors.data(), inputTensors.size());
        CHECK_ASSERT(concat, "create concatenation node fail\n");
        return concat;
    }
} //tensorrtInference