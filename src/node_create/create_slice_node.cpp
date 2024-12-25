/********************************************
 * Filename: create_slice_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "node_create/create_node.hpp"
#include "node_create/create_slice_node.hpp"
#include "node_info/slice_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createSliceNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto inputs = node_info->getInputs();
        CHECK_ASSERT(inputs.size() >= 3 && inputs.size() <= 5, "conv2d inputs must greater equal than 3 and less equal than 5\n");
        nvinfer1::ISliceLayer* slice = nullptr;
        nvinfer1::ITensor* inputTensor = tensors[inputs[0]];
        nvinfer1::Dims dims = inputTensor->getDimensions();
        if(inputs.size() == 3)
        {
            auto starts = parseIntArrayValue(node_weight_info[inputs[1]].dataType, node_weight_info[inputs[1]].data, 
                    node_weight_info[inputs[1]].byteCount, node_weight_info[inputs[1]].shape);
            auto ends   = parseIntArrayValue(node_weight_info[inputs[2]].dataType, node_weight_info[inputs[2]].data, 
                    node_weight_info[inputs[2]].byteCount, node_weight_info[inputs[2]].shape);
            std::vector<int> axes = {1, 1, 1, 1};
            std::vector<int> steps = {1, 1, 1, 1};
            CHECK_ASSERT(starts.size() == ends.size(), "starts size must be equal to ends size!\n");
            CHECK_ASSERT(starts.size() == dims.nbDims, "starts size(%d) must be equal to input tensor dims(%d)!\n", starts.size(), dims.nbDims);

            for(int i = 0; i < starts.size(); i++)
            {
                int rank = starts.size();
                if(starts[i] < 0)
                    starts[i] += dims.d[i];
                if(ends[i] < 0)
                    ends[i] += dims.d[i];
            }
            std::vector<int> size = {1, 1, 1, 1};
            std::vector<int> stride = {1, 1, 1, 1};
            for(int i = 0; i < starts.size(); i++)
            {
                size[i] = ends[i] - starts[i];
            }
            slice = network->addSlice(*inputTensor, nvinfer1::Dims4{starts[0], starts[1], starts[2], starts[3]}, 
                    nvinfer1::Dims4{size[0], size[1], size[2], size[3]}, 
                    nvinfer1::Dims4{stride[0], stride[1], stride[2], stride[3]});
        }
        else if(inputs.size() == 4)
        {
            auto starts = parseIntArrayValue(node_weight_info[inputs[1]].dataType, node_weight_info[inputs[1]].data, 
                    node_weight_info[inputs[1]].byteCount, node_weight_info[inputs[1]].shape);
            auto ends   = parseIntArrayValue(node_weight_info[inputs[2]].dataType, node_weight_info[inputs[2]].data, 
                    node_weight_info[inputs[2]].byteCount, node_weight_info[inputs[2]].shape);
            auto axes   = parseIntArrayValue(node_weight_info[inputs[3]].dataType, node_weight_info[inputs[3]].data, 
                    node_weight_info[inputs[3]].byteCount, node_weight_info[inputs[3]].shape);
            std::vector<int> steps = {1, 1, 1, 1};
            CHECK_ASSERT(starts.size() == dims.nbDims, "starts size(%d) must be equal to input tensor dims(%d)!\n", starts.size(), dims.nbDims);
            CHECK_ASSERT(starts.size() == ends.size(), "starts size must be equal to ends size!\n");
            CHECK_ASSERT(starts.size() == axes.size(), "starts size must be equal to axes size!\n");
            for(int i = 0; i < starts.size(); i++)
            {
                if(starts[axes[i]] < 0)
                    starts[axes[i]] += dims.d[axes[i]];
                if(ends[axes[i]] < 0)
                    ends[axes[i]] += dims.d[axes[i]];
                if(starts[axes[i]] == 0x7fffffff)
                    starts[axes[i]] = dims.d[axes[i]];
                if(ends[axes[i]] == 0x7fffffff)
                    ends[axes[i]] = dims.d[axes[i]];
                CHECK_ASSERT(axes[i] < starts.size(), "axes value set error!\n");
            }
            std::vector<int> size = {1, 1, 1, 1};
            std::vector<int> stride = {1, 1, 1, 1};
            for(int i = 0; i < starts.size(); i++)
            {
                size[axes[i]] = ends[axes[i]] - starts[axes[i]];
            }
            slice = network->addSlice(*inputTensor, nvinfer1::Dims4{starts[0], starts[1], starts[2], starts[3]}, 
                    nvinfer1::Dims4{size[0], size[1], size[2], size[3]}, 
                    nvinfer1::Dims4{stride[0], stride[1], stride[2], stride[3]});
        }
        else
        {
            auto starts = parseIntArrayValue(node_weight_info[inputs[1]].dataType, node_weight_info[inputs[1]].data, 
                    node_weight_info[inputs[1]].byteCount, node_weight_info[inputs[1]].shape);
            auto ends   = parseIntArrayValue(node_weight_info[inputs[2]].dataType, node_weight_info[inputs[2]].data, 
                    node_weight_info[inputs[2]].byteCount, node_weight_info[inputs[2]].shape);
            auto axes   = parseIntArrayValue(node_weight_info[inputs[3]].dataType, node_weight_info[inputs[3]].data, 
                    node_weight_info[inputs[3]].byteCount, node_weight_info[inputs[3]].shape);
            auto steps = parseIntArrayValue(node_weight_info[inputs[4]].dataType, node_weight_info[inputs[4]].data, 
                    node_weight_info[inputs[4]].byteCount, node_weight_info[inputs[4]].shape);
            CHECK_ASSERT(starts.size() <= dims.nbDims, "starts size(%d) must be less than input tensor dims(%d)!\n", starts.size(), dims.nbDims);
            CHECK_ASSERT(starts.size() == ends.size(), "starts size must be equal to ends size!\n");
            CHECK_ASSERT(starts.size() == axes.size(), "starts size must be equal to axes size!\n");
            CHECK_ASSERT(starts.size() == steps.size(), "starts size must be equal to steps size!\n");
            for(int i = 0; i < starts.size(); i++)
            {
                if(starts[i] < 0)
                    starts[i] += dims.d[axes[i]];
                if(ends[i] < 0)
                    ends[i] += dims.d[axes[i]];
                if(starts[i] == 0x7fffffff)
                    starts[i] = dims.d[axes[i]];
                if(ends[i] == 0x7fffffff)
                    ends[i] = dims.d[axes[i]];
            }
            std::vector<int> size(dims.nbDims);
            std::vector<int> stride(dims.nbDims);
            std::vector<int> start(dims.nbDims);
            for(int i = 0; i < dims.nbDims; i++)
            {
                size[i] = dims.d[i];
                stride[i] = 1;
            }
            for(int i = 0; i < starts.size(); i++)
            {
                start[axes[i]] = starts[i];
                size[axes[i]] = (ends[i] - starts[i]) / steps[i];
                stride[axes[i]] = steps[i];
            }
            nvinfer1::Dims startDims = vectorToDims(start);
            nvinfer1::Dims sizeDims = vectorToDims(size);
            nvinfer1::Dims strideDims = vectorToDims(stride);
            slice = network->addSlice(*inputTensor, startDims, sizeDims, strideDims);
        }
        CHECK_ASSERT(slice, "create slice node fail\n");
        return slice;
    }

} // namespace TENSORRT_WRAPPER