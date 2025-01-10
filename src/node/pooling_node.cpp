/********************************************
 * Filename: pooling_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "NvInfer.h"
#include "parser/graph_parser.hpp"
#include "node/node_creator.hpp"
#include "node_info/pooling_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    void getKernelParams(PoolingNodeInfo* NodeInfo, nvinfer1::Dims* kernelSize, nvinfer1::Dims* strides, nvinfer1::Dims* begPadding, 
        nvinfer1::Dims* endPadding, nvinfer1::PaddingMode& paddingMode, bool& countExcludePadding, const bool poolingCeilMode)
    {
        kernelSize->nbDims = 0;
        strides->nbDims = 0;
        begPadding->nbDims = 0;
        endPadding->nbDims = 0;
        auto onnxKernelShape      = NodeInfo->getKernelShape();
        auto onnxPads             = NodeInfo->getPads();
        auto onnxStrides          = NodeInfo->getStrides();
        auto onnxAutoPad          = NodeInfo->getAutoPad();
        auto onnxCountIncludePad  = NodeInfo->getCountIncludePad();
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
        NodeInfo* node_info, std::map<std::string, WeightInfo>& weight_info)
    {
        auto pooling_node_info = (PoolingNodeInfo *)node_info;
        auto inputs = pooling_node_info->getInputs();
        nvinfer1::ITensor* input_tensor = tensors[inputs[0]];
        nvinfer1::IPoolingLayer* pooling = nullptr;
        nvinfer1::Dims kernel_size;
        nvinfer1::Dims strides;
        nvinfer1::Dims begin_padding;
        nvinfer1::Dims end_padding;
        nvinfer1::PaddingMode padding_mode;
        bool exclude_padding = true;
        auto pooling_ceil_mode = pooling_node_info->getCeilMode();
        getKernelParams(pooling_node_info, &kernel_size, &strides, &begin_padding, &end_padding, padding_mode, 
            exclude_padding, pooling_ceil_mode);
        auto subNodeType     = pooling_node_info->getNodeSubType();
        if(subNodeType.compare("MaxPool") == 0 )
        {
            pooling = network->addPoolingNd(*input_tensor, nvinfer1::PoolingType::kMAX, kernel_size);
        }
        else if(subNodeType.compare("AveragePool") == 0)
        {
            pooling = network->addPoolingNd(*input_tensor, nvinfer1::PoolingType::kAVERAGE, kernel_size);
        }
        else
            LOG("current noly support max/average 2d pooling!\n");
        CHECK_ASSERT(pooling, "create pooling node fail\n");
        pooling->setAverageCountExcludesPadding(exclude_padding);
        pooling->setPaddingMode(padding_mode);
        if(strides.nbDims)
            pooling->setStrideNd(strides);
        if(beginPadding.nbDims)
            pooling->setPrePadding(begin_padding);
        if(endPadding.nbDims)
            pooling->setPostPadding(end_padding);

        return pooling;
    }

    class PoolingNodeCreator : public NodeCreator
    {
    public:
        virtual nvinfer1::ILayer* onCreate(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
            NodeInfo* node_info, std::map<std::string, WeightInfo>& weight_info) const override 
        {
            return createPoolingNode(network, tensors, node_info, weight_info);
        }
    };

    void registerPoolingNodeCreator()
    {
        insertNodeCreator("Pooling", new PoolingNodeCreator);
    }

} // namespace TENSORRT_WRAPPER