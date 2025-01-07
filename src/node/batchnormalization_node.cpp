/********************************************
 * Filename: batchnormalization_node.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "NvInfer.h"
#include "parser/graph_parser.hpp"
#include "node/node_creator.hpp"
#include "node_info/batchnormalization_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createBatchNormalizationNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto batch_normal_node_info = (BatchNormalizationNodeInfo*)node_info;
        auto inputs = batch_normal_node_info->getInputs();
        float epsilon = batch_normal_node_info->getEpsilon();
        nvinfer1::ITensor* input_tensor = tensors[inputs[0]];
        auto scaleWeight = node_weight_info[inputs[1]];
        auto biasWeight = node_weight_info[inputs[2]];
        auto meanWeight = node_weight_info[inputs[3]];
        auto varWeight = node_weight_info[inputs[4]];
        auto scaleType = scaleWeight.dataType;
        auto biasType = biasWeight.dataType;
        auto meanType = meanWeight.dataType;
        auto varType = varWeight.dataType;
        CHECK_ASSERT(scaleType == biasType && meanType == varType && biasType == varType, "scale bias mean var must have same data type!\n");
        CHECK_ASSERT((OnnxDataType)scaleType == OnnxDataType::FLOAT, "scale bias mean var must be float!\n");
        nvinfer1::Dims dims = input_tensor->getDimensions();
        CHECK_ASSERT(dims.nbDims == 4 || dims.nbDims == 5, "input tensor dims must be 4 or 5!\n");

        WeightInfo combinedScale;
        WeightInfo combinedBias;
        combinedScale.byteCount = scaleWeight.byteCount;
        combinedScale.dataType = scaleWeight.dataType;
        combinedScale.shape = scaleWeight.shape;
        combinedScale.data = nullptr;
        combinedScale.data = (char*)malloc(combinedScale.byteCount);
        CHECK_ASSERT(combinedScale.data, "malloc mem fail!\n");
        std::string combinedScaleName = "combinedScale_" + inputs[1];
        node_weight_info[combinedScaleName] = combinedScale;

        combinedBias.byteCount = scaleWeight.byteCount;
        combinedBias.dataType = scaleWeight.dataType;
        combinedBias.shape = scaleWeight.shape;
        combinedBias.data = nullptr;
        combinedBias.data = (char*)malloc(combinedBias.byteCount);
        CHECK_ASSERT(combinedBias.data, "malloc mem fail!\n");
        std::string combinedBiasName = "combinedBias_" + inputs[2];
        node_weight_info[combinedBiasName] = combinedBias;

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
        nvinfer1::IScaleLayer* batchNormalLayer = network->addScaleNd(*input_tensor, mode, shift, scale, power, 1);
        
        CHECK_ASSERT(batchNormalLayer, "create BatchNormalization node fail\n");
        return batchNormalLayer;
    }

    class BatchNormalizationNodeCreator : public NodeCreator
    {
    public:
        virtual nvinfer1::ILayer* onCreate(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
            NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info) const override 
        {
            return createBatchNormalizationNode(network, tensors, node_info, node_weight_info);
        }
    };

    void registerBatchNormalizationNodeCreator()
    {
        insertNodeCreator("BatchNormalization", new BatchNormalizationNodeCreator);
    }

} // namespace TENSORRT_WRAPPER