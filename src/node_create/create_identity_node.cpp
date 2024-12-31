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
        auto identity_node_info = (IdentityNodeInfo *)node_info;
        auto inputs = identity_node_info->getInputs();
        nvinfer1::ITensor* input_tensor = tensors[inputs[0]];
        nvinfer1::IIdentityLayer* identity = network->addIdentity(*input_tensor);
        CHECK_ASSERT(identity, "create identity node fail\n");
        int type = getTensorrtDataType(OnnxDataType(identity_node_info->getDataType()));
        CHECK_ASSERT(type != -1, "only support float/half!\n");
        identity->setOutputType(0, nvinfer1::DataType(type));
        return identity;
    }

} // namespace TENSORRT_WRAPPER