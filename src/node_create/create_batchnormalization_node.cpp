#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_batchnormalization_node.hpp"
#include "batchnormalization_node_info.hpp"

namespace tensorrtInference
{
    nvinfer1::ILayer* createBatchNormalizationNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto batchNormalNodeInfo = (BatchNormalizationNodeInfo*)nodeConfInfo;
        auto inputs = batchNormalNodeInfo->getInputs();
        float epsilon = batchNormalNodeInfo->getEpsilon();
        nvinfer1::ITensor* inputTensor = tensors[inputs[0]];
        auto scaleWeight = nodeWeightsInfo[inputs[1]];
        auto biasWeight = nodeWeightsInfo[inputs[2]];
        auto meanWeight = nodeWeightsInfo[inputs[3]];
        auto varWeight = nodeWeightsInfo[inputs[4]];
        auto scaleType = scaleWeight.dataType;
        auto biasType = biasWeight.dataType;
        auto meanType = meanWeight.dataType;
        auto varType = varWeight.dataType;
        CHECK_ASSERT(scaleType == biasType && meanType == varType && biasType == varType, "scale bias mean var must have same data type!\n");
        CHECK_ASSERT((OnnxDataType)scaleType == OnnxDataType::FLOAT, "scale bias mean var must be float!\n");
        nvinfer1::Dims dims = inputTensor->getDimensions();
        CHECK_ASSERT(dims.nbDims == 4 || dims.nbDims == 5, "input tensor dims must be 4 or 5!\n");

        weightInfo combinedScale;
        weightInfo combinedBias;
        combinedScale.byteCount = scaleWeight.byteCount;
        combinedScale.dataType = scaleWeight.dataType;
        combinedScale.shape = scaleWeight.shape;
        combinedScale.data = nullptr;
        combinedScale.data = (char*)malloc(combinedScale.byteCount);
        CHECK_ASSERT(combinedScale.data, "malloc mem fail!\n");
        std::string combinedScaleName = "combinedScale_" + inputs[1];
        nodeWeightsInfo[combinedScaleName] = combinedScale;

        combinedBias.byteCount = scaleWeight.byteCount;
        combinedBias.dataType = scaleWeight.dataType;
        combinedBias.shape = scaleWeight.shape;
        combinedBias.data = nullptr;
        combinedBias.data = (char*)malloc(combinedBias.byteCount);
        CHECK_ASSERT(combinedBias.data, "malloc mem fail!\n");
        std::string combinedBiasName = "combinedBias_" + inputs[2];
        nodeWeightsInfo[combinedBiasName] = combinedBias;

        auto nweight = scaleWeight.shape[0];
        for (size_t i = 0; i < nweight; ++i)
        {
            float scale = ((float*)(scaleWeight.data))[i];
            float bias = ((float*)(biasWeight.data))[i];
            float mean = ((float*)(meanWeight.data))[i];
            float variance = ((float*)(varWeight.data))[i];
            float* combinedScalePtr = (float*)(combinedScale.data) + i;
            float* combinedBiasPtr = (float*)(combinedBias.data) + i;
            *combinedScalePtr = scale / sqrtf(variance + epsilon);
            *combinedBiasPtr = bias - mean * (*combinedScalePtr);
        }

        auto shift = combinedBias.getTensorrtWeights();
        auto scale = combinedScale.getTensorrtWeights();
        nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};
        nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kCHANNEL;
        nvinfer1::IScaleLayer* batchNormalLayer = network->addScaleNd(*inputTensor, mode, shift, scale, power, 1);
        
        CHECK_ASSERT(batchNormalLayer, "create BatchNormalization node fail\n");
        return batchNormalLayer;
    }
}