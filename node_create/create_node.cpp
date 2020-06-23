#include "create_node.hpp"
#include "create_elementwise_node.hpp"
#include "create_padding_node.hpp"
#include "create_reduce_node.hpp"
#include "create_softmax_node.hpp"
#include "create_unary_node.hpp"
#include "create_shuffle_node.hpp"
#include "create_activation_node.hpp"
#include "create_conv2d_node.hpp"
#include "create_slice_node.hpp"
#include "create_identity_node.hpp"
#include "create_pooling_node.hpp"
#include "create_nonzero_node.hpp"
#include "create_shape_node.hpp"
#include "create_gather_node.hpp"
#include "create_unsqueeze_node.hpp"
#include "create_concatenation_node.hpp"
#include "create_gemm_node.hpp"

namespace tensorrtInference
{
    typedef nvinfer1::ILayer* (*func)(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo);
    
    static std::map<std::string, func> createNodeFuncMap;

    nvinfer1::ILayer* createNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
     tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        if(createNodeFuncMap.size() == 0)
        {
            createNodeFuncMap["ElementWise"]    = tensorrtInference::createElementWiseNode;
            createNodeFuncMap["Activation"]     = tensorrtInference::createActivationNode;
            createNodeFuncMap["Padding"]        = tensorrtInference::createPaddingNode;
            createNodeFuncMap["Reduce"]         = tensorrtInference::createReduceNode;
            createNodeFuncMap["Softmax"]        = tensorrtInference::createSoftmaxNode;
            createNodeFuncMap["Unary"]          = tensorrtInference::createUnaryNode;
            createNodeFuncMap["Shuffle"]        = tensorrtInference::createShuffleNode;
            createNodeFuncMap["Conv2d"]         = tensorrtInference::createConv2dNode;
            createNodeFuncMap["Slice"]          = tensorrtInference::createSliceNode;
            createNodeFuncMap["Identity"]       = tensorrtInference::createIdentityNode;
            createNodeFuncMap["Pooling"]        = tensorrtInference::createPoolingNode;
            createNodeFuncMap["NonZero"]        = tensorrtInference::createNonZeroNode;
            createNodeFuncMap["Shape"]          = tensorrtInference::createShapeNode;
            createNodeFuncMap["Gather"]         = tensorrtInference::createGatherNode;
            createNodeFuncMap["Unsqueeze"]      = tensorrtInference::createUnsqueezeNode;
            createNodeFuncMap["Concatenation"]  = tensorrtInference::createConcatenationNode;
            createNodeFuncMap["Gemm"]           = tensorrtInference::createGemmNode;
        }
        auto inputs = nodeConfInfo->getInputs();
        for(int i = 0; i < inputs.size(); i++)
        {
            if(tensors.count(inputs[i]) == 0 && nodeWeightsInfo.count(inputs[i]) == 0)
            {
                CHECK_ASSERT(0, "topo order wrong!\n");
            }
        }
        auto nodeType = nodeConfInfo->getNodeType();
        nvinfer1::ILayer* layer = nullptr;
        if(createNodeFuncMap.count(nodeType) != 0)
        {
            layer = createNodeFuncMap[nodeType](network, tensors, nodeConfInfo, nodeWeightsInfo);
        }
        else
            LOG("current not support node type (%s)\n", nodeType.c_str());
        
        return layer;
    }

} // tensorrtInference