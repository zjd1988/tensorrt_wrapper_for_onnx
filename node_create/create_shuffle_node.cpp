#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "shuffle_node_info.hpp"
namespace tensorrtInference
{
    nvinfer1::ILayer* createShuffleNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto shuffleNodeInfo = (ShuffleNodeInfo*)nodeConfInfo;
        auto inputs = shuffleNodeInfo->getInputs();
        nvinfer1::IShuffleLayer* shuffle = nullptr;
        nvinfer1::ITensor* inputTensor = tensors[inputs[0]];
        auto subType = shuffleNodeInfo->getSubNodeType();
        if(inputs.size() == 1 && subType.compare("Transpose") == 0)
        {
            auto perm = shuffleNodeInfo->getPerm();
            shuffle = network->addShuffle(*inputTensor);
            CHECK_ASSERT(shuffle, "create shuffle node fail\n");
            CHECK_ASSERT(perm.size() == 4, "perm dims must equal to 4\n");
            nvinfer1::Dims dims = inputTensor->getDimensions();
            shuffle->setFirstTranspose(nvinfer1::Permutation{perm[0], perm[1], perm[2], perm[3]});
            if(dims.nbDims == 4)
                shuffle->setReshapeDimensions(nvinfer1::Dims4(dims.d[perm[0]], dims.d[perm[1]], dims.d[perm[2]], dims.d[perm[3]]));
            else
                CHECK_ASSERT(0, "current only support 4 dims in transpose\n");
        }
        else if(inputs.size() == 2 && subType.compare("Reshape") == 0)
        {
            auto dims = parseIntArrayValue(nodeWeightsInfo[inputs[1]].dataType, nodeWeightsInfo[inputs[1]].data, 
                            nodeWeightsInfo[inputs[1]].byteCount, nodeWeightsInfo[inputs[1]].shape);
            shuffle = network->addShuffle(*inputTensor);
            CHECK_ASSERT(shuffle, "create shuffle node fail\n");

            nvinfer1::Dims newDims;
            newDims.nbDims = dims.size();
            for(int i = 0; i < dims.size(); i++)
            {
                newDims.d[i] = dims[i];
            }
            shuffle->setReshapeDimensions(newDims);
        }
        else if(subType.compare("Flatten") == 0)
        {
            auto tensorDims = inputTensor->getDimensions();
            int axis = shuffleNodeInfo->getAxis();
            int newAxis = 0;
            newAxis = (axis < 0) ? (axis += tensorDims.nbDims) : axis;
            CHECK_ASSERT(newAxis >= 0 && newAxis < tensorDims.nbDims, "axis(%d) must be 0 < axis < %d\n", newAxis, tensorDims.nbDims);
            int d0 = 1;
            int d1 = 1;
            std::vector<int> shape;
            d0 = std::accumulate(&(tensorDims.d[0]), &(tensorDims.d[newAxis]), 1, std::multiplies<int>());
            d1 = std::accumulate(&(tensorDims.d[newAxis]), &(tensorDims.d[tensorDims.nbDims]), 1, std::multiplies<int>());
            shape.push_back(d0);
            shape.push_back(d1);
            nvinfer1::Dims newDims = vectorToDims(shape);
            shuffle = network->addShuffle(*inputTensor);
            CHECK_ASSERT(shuffle, "create shuffle(flatten) node fail\n");
            shuffle->setReshapeDimensions(newDims);
        }
        else
            LOG("unspported shuffle sub type: %s", subType.c_str());
        return shuffle;
    }
}