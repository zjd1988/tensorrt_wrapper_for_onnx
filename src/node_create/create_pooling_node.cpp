#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "pooling_node_info.hpp"
#include "create_pooling_node.hpp"


namespace tensorrtInference
{
    void getKernelParams(PoolingNodeInfo* nodeInfo, nvinfer1::Dims* kernelSize, nvinfer1::Dims* strides, nvinfer1::Dims* begPadding, 
    nvinfer1::Dims* endPadding, nvinfer1::PaddingMode& paddingMode, bool& countExcludePadding, const bool poolingCeilMode)
    {
        kernelSize->nbDims = 0;
        strides->nbDims = 0;
        begPadding->nbDims = 0;
        endPadding->nbDims = 0;
        auto onnxKernelShape      = nodeInfo->getKernelShape();
        auto onnxPads             = nodeInfo->getPads();
        auto onnxStrides          = nodeInfo->getStrides();
        auto onnxAutoPad          = nodeInfo->getAutoPad();
        auto onnxCountIncludePad  = nodeInfo->getCountIncludePad();
        CHECK_ASSERT(onnxKernelShape.size() == 2, "current only support 2d pooling\n");

        kernelSize->nbDims = onnxKernelShape.size();
        for(int i = 0; i < onnxKernelShape.size(); i++)
        {
            kernelSize->d[i] = onnxKernelShape[i];
        }
        strides->nbDims = onnxStrides.size();
        for(int i = 0; i < onnxStrides.size(); i++)
        {
            strides->d[i] = onnxStrides[i];
        }
        countExcludePadding = (onnxCountIncludePad == 1) ? false : true;
        paddingMode = poolingCeilMode ? nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP : nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
        if (onnxAutoPad.compare("SAME_LOWER") != 0 && onnxAutoPad.compare("SAME_UPPER") != 0)
        {
            if (onnxPads.size() > 0)
            {
                begPadding->nbDims = kernelSize->nbDims;
                endPadding->nbDims = kernelSize->nbDims;
                int ndim = onnxPads.size() / 2;
                CHECK_ASSERT(ndim <= kernelSize->nbDims, "pads must be less than 2x kernel size\n");
                for (int i = 0; i < begPadding->nbDims; ++i)
                {
                    if (i < ndim)
                    {
                        begPadding->d[i] = onnxPads[i];
                        endPadding->d[i] = onnxPads[i + ndim];
                    }
                    else
                    {
                        begPadding->d[i] = 0;
                        endPadding->d[i] = 0;
                    }
                }
            }
        }
        else
        {
            CHECK_ASSERT(onnxPads.size() <= 0, "If auto_pad is SAME_LOWER or SAME_UPPER, input padding should be calculated \n'pads' attribute should not be specified\n");
            // Note: ONNX is always NCHW ordering
            if (onnxAutoPad.compare("SAME_LOWER") == 0)
            {
                paddingMode = nvinfer1::PaddingMode::kSAME_LOWER;
            }
            else if (onnxAutoPad.compare("SAME_UPPER") == 0)
            {
                paddingMode = nvinfer1::PaddingMode::kSAME_UPPER;
            }
            else
            {
                CHECK_ASSERT(0, "Unexpected auto_pad value: %s \n", onnxAutoPad.c_str());
            }
        }
    }
    nvinfer1::ILayer* createPoolingNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        PoolingNodeInfo *nodeConfigInfo = (PoolingNodeInfo *)nodeConfInfo;
        auto inputs = nodeConfigInfo->getInputs();
        nvinfer1::ITensor* inputTensor = tensors[inputs[0]];
        nvinfer1::IPoolingLayer* pooling = nullptr;
        nvinfer1::Dims kernelSize;
        nvinfer1::Dims strides;
        nvinfer1::Dims beginPadding;
        nvinfer1::Dims endPadding;
        nvinfer1::PaddingMode paddingMode;
        bool excludePadding = true;
        auto poolingCeilMode = nodeConfigInfo->getCeilMode();
        getKernelParams(nodeConfigInfo, &kernelSize, &strides, &beginPadding, &endPadding, paddingMode, excludePadding, poolingCeilMode);
        auto subNodeType     = nodeConfigInfo->getSubNodeType();
        if(subNodeType.compare("MaxPool") == 0 )
        {
            pooling = network->addPoolingNd(*inputTensor, nvinfer1::PoolingType::kMAX, kernelSize);
        }
        else if(subNodeType.compare("AveragePool") == 0)
        {
            pooling = network->addPoolingNd(*inputTensor, nvinfer1::PoolingType::kAVERAGE, kernelSize);
        }
        else
            LOG("current noly support max/average 2d pooling!\n");
        CHECK_ASSERT(pooling, "create pooling node fail\n");
        pooling->setAverageCountExcludesPadding(excludePadding);
        pooling->setPaddingMode(paddingMode);
        if(strides.nbDims)
            pooling->setStrideNd(strides);
        if(beginPadding.nbDims)
            pooling->setPrePadding(beginPadding);
        if(endPadding.nbDims)
            pooling->setPostPadding(endPadding);
        
        return pooling;
    }
}