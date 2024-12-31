/********************************************
 * Filename: create_shuffle_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_create/create_node.hpp"
#include "node_create/create_shuffle_node.hpp"
#include "node_info/shuffle_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createShuffleNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto shuffle_node_info = (ShuffleNodeInfo*)node_info;
        auto inputs = shuffle_node_info->getInputs();
        nvinfer1::IShuffleLayer* shuffle = nullptr;
        nvinfer1::ITensor* input_tensor = tensors[inputs[0]];
        auto subType = shuffle_node_info->getNodeSubType();
        if(inputs.size() == 1 && subType.compare("Transpose") == 0)
        {
            auto perm = shuffle_node_info->getPerm();
            shuffle = network->addShuffle(*input_tensor);
            CHECK_ASSERT(shuffle, "create shuffle node fail\n");
            // CHECK_ASSERT(perm.size() == 4, "perm dims must equal to 4\n");
            nvinfer1::Dims dims = input_tensor->getDimensions();
            CHECK_ASSERT(perm.size() == dims.nbDims, "perm dims must equal to input tensors dim\n");
            nvinfer1::Permutation permutation;
            for(int i = 0; i < perm.size(); i++) {
                permutation.order[i] = perm[i];
            }
            shuffle->setFirstTranspose(permutation);
            nvinfer1::Dims new_dims;
            new_dims.nbDims = dims.nbDims;
            for(int i = 0; i < dims.nbDims; i++) {
                new_dims.d[i] = dims.d[perm[i]];
            }
            shuffle->setReshapeDimensions(new_dims);
            // shuffle->setFirstTranspose(nvinfer1::Permutation{perm[0], perm[1], perm[2], perm[3]});
            // if(dims.nbDims == 4)
            //     shuffle->setReshapeDimensions(nvinfer1::Dims4(dims.d[perm[0]], dims.d[perm[1]], dims.d[perm[2]], dims.d[perm[3]]));
            // else
            //     CHECK_ASSERT(0, "current only support 4 dims in transpose\n");
        }
        else if(inputs.size() == 2 && subType.compare("Reshape") == 0)
        {
            auto dims = parseIntArrayValue(node_weight_info[inputs[1]].dataType, node_weight_info[inputs[1]].data, 
                            node_weight_info[inputs[1]].byteCount, node_weight_info[inputs[1]].shape);
            shuffle = network->addShuffle(*input_tensor);
            CHECK_ASSERT(shuffle, "create shuffle node fail\n");

            nvinfer1::Dims new_dims;
            new_dims.nbDims = dims.size();
            for(int i = 0; i < dims.size(); i++)
            {
                new_dims.d[i] = dims[i];
            }
            shuffle->setReshapeDimensions(new_dims);
        }
        else if(subType.compare("Flatten") == 0)
        {
            auto tensor_dims = input_tensor->getDimensions();
            int axis = shuffle_node_info->getAxis();
            int new_axis = 0;
            new_axis = (axis < 0) ? (axis += tensor_dims.nbDims) : axis;
            CHECK_ASSERT(new_axis >= 0 && new_axis < tensor_dims.nbDims, "axis(%d) must be 0 < axis < %d\n", new_axis, tensor_dims.nbDims);
            int d0 = 1;
            int d1 = 1;
            std::vector<int> shape;
            d0 = std::accumulate(&(tensor_dims.d[0]), &(tensor_dims.d[new_axis]), 1, std::multiplies<int>());
            d1 = std::accumulate(&(tensor_dims.d[new_axis]), &(tensor_dims.d[tensor_dims.nbDims]), 1, std::multiplies<int>());
            shape.push_back(d0);
            shape.push_back(d1);
            nvinfer1::Dims new_dims = vectorToDims(shape);
            shuffle = network->addShuffle(*input_tensor);
            CHECK_ASSERT(shuffle, "create shuffle(flatten) node fail\n");
            shuffle->setReshapeDimensions(new_dims);
        }
        else
            LOG("unspported shuffle sub type: %s", subType.c_str());
        return shuffle;
    }

} // namespace TENSORRT_WRAPPER