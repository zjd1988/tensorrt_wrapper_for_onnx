/********************************************
 * Filename: resize_node.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "NvInfer.h"
#include "parser/graph_parser.hpp"
#include "node/node_creator.hpp"
#include "node/resize_node_info.hpp"

namespace TENSORRT_WRAPPER
{

    nvinfer1::ILayer* createResizeNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info)
    {
        auto resize_node_info = (ResizeNodeInfo*)node_info;
        auto inputs = resize_node_info->getInputs();
        std::string mode = resize_node_info->getMode();
        nvinfer1::ResizeMode resize_mode;
        if(mode.compare("nearest") == 0)
            resize_mode = nvinfer1::ResizeMode::kNEAREST;
        else if(mode.compare("linear") == 0)
            resize_mode = nvinfer1::ResizeMode::kLINEAR;
        else
            CHECK_ASSERT(0, "current only support nearest/linear resize mode\n");
        nvinfer1::ITensor* input_tensor = tensors[inputs[0]];
        nvinfer1::IResizeLayer* resize = network->addResize(*input_tensor);
        CHECK_ASSERT(resize, "create resize node fail\n");

        auto scale_weights = node_weight_info[inputs[1]];
        auto scales = parseFloatArrayValue(scale_weights.dataType, scale_weights.data, scale_weights.byteCount, scale_weights.shape);
        resize->setScales(scales.data(), scales.size());
        resize->setResizeMode(resize_mode);
        return resize;
    }

    class ResizeNodeCreator : public NodeCreator
    {
    public:
        virtual nvinfer1::ILayer* onCreate(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
            NodeInfo* node_info, std::map<std::string, WeightInfo>& node_weight_info) const override 
        {
            return createResizeNode(network, tensors, node_info, node_weight_info);
        }
    };

    void registerResizeNodeCreator()
    {
        insertNodeCreator("Resize", new ResizeNodeCreator);
    }

} // namespace TENSORRT_WRAPPER