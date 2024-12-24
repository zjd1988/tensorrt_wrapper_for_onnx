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
#include "create_resize_node.hpp"
#include "create_batchnormalization_node.hpp"

namespace TENSORRT_WRAPPER
{

    typedef nvinfer1::ILayer* (*func)(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info);
    
    static std::map<std::string, func> createNodeFuncMap;

    nvinfer1::ILayer* createNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
     NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        if(createNodeFuncMap.size() == 0)
        {
            createNodeFuncMap["ElementWise"]             = createElementWiseNode;
            createNodeFuncMap["Activation"]              = createActivationNode;
            createNodeFuncMap["Padding"]                 = createPaddingNode;
            createNodeFuncMap["Reduce"]                  = createReduceNode;
            createNodeFuncMap["Softmax"]                 = createSoftmaxNode;
            createNodeFuncMap["Unary"]                   = createUnaryNode;
            createNodeFuncMap["Shuffle"]                 = createShuffleNode;
            createNodeFuncMap["Conv2d"]                  = createConv2dNode;
            createNodeFuncMap["Slice"]                   = createSliceNode;
            createNodeFuncMap["Identity"]                = createIdentityNode;
            createNodeFuncMap["Pooling"]                 = createPoolingNode;
            createNodeFuncMap["NonZero"]                 = createNonZeroNode;
            createNodeFuncMap["Shape"]                   = createShapeNode;
            createNodeFuncMap["Gather"]                  = createGatherNode;
            createNodeFuncMap["Unsqueeze"]               = createUnsqueezeNode;
            createNodeFuncMap["Concatenation"]           = createConcatenationNode;
            createNodeFuncMap["Gemm"]                    = createGemmNode;
            createNodeFuncMap["Resize"]                  = createResizeNode;
            createNodeFuncMap["BatchNormalization"]      = createBatchNormalizationNode;
        }
        auto inputs = node_info->getInputs();
        for(int i = 0; i < inputs.size(); i++)
        {
            if(tensors.count(inputs[i]) == 0 && node_weight_info.count(inputs[i]) == 0)
            {
                CHECK_ASSERT(0, "topo order wrong!\n");
            }
        }
        auto nodeType = node_info->getNodeType();
        nvinfer1::ILayer* layer = nullptr;
        if(createNodeFuncMap.count(nodeType) != 0)
        {
            layer = createNodeFuncMap[nodeType](network, tensors, node_info, node_weight_info);
        }
        else
            LOG("current not support node type (%s)\n", nodeType.c_str());
        
        return layer;
    }

} // namespace TENSORRT_WRAPPER