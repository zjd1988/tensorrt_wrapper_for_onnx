/********************************************
 * Filename: create_gemm_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_create/create_node.hpp"
#include "node_create/create_identity_node.hpp"
#include "node_info/identity_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createIdentityNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        IdentityNodeInfo *nodeConfigInfo = (IdentityNodeInfo *)node_info;
        auto inputs = nodeConfigInfo->getInputs();
        nvinfer1::ITensor* inputTensor = tensors[inputs[0]];
        nvinfer1::IIdentityLayer* identity = network->addIdentity(*inputTensor);
        CHECK_ASSERT(identity, "create identity node fail\n");
        int type = getTensorrtDataType(OnnxDataType(nodeConfigInfo->getDataType()));
        CHECK_ASSERT(type != -1, "only support float/half!\n");
        identity->setOutputType(0, nvinfer1::DataType(type));
        return identity;
    }

} // namespace TENSORRT_WRAPPER