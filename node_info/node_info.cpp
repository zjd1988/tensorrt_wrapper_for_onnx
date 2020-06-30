#include "node_info.hpp"
#include "conv2d_node_info.hpp"
#include "elementwise_node_info.hpp"
#include "activation_node_info.hpp"
#include "shuffle_node_info.hpp"
#include "padding_node_info.hpp"
#include "unary_node_info.hpp"
#include "softmax_node_info.hpp"
#include "reduce_node_info.hpp"
#include "pooling_node_info.hpp"
#include "slice_node_info.hpp"
#include "identity_node_info.hpp"
#include "nonzero_node_info.hpp"
#include "shape_node_info.hpp"
#include "gather_node_info.hpp"
#include "unsqueeze_node_info.hpp"
#include "concatenation_node_info.hpp"
#include "gemm_node_info.hpp"
#include "resize_node_info.hpp"
#include "batchnormalization_node_info.hpp"

#define PARSE_NODE_FUNC_DEF(nodeType)                                      \
nodeInfo* parse##nodeType##NodeInfo(std::string type, Json::Value& root)   \
{                                                                          \
    nodeInfo* node = new nodeType##NodeInfo();                             \
    if(node->parseNodeInfoFromJson(type, root))                            \
        return node;                                                       \
    else                                                                   \
        delete node;                                                       \
    return nullptr;                                                        \
}

#define PARSE_NODE_FUNC(nodeType) parse##nodeType##NodeInfo

namespace tensorrtInference
{
    //nodeInfo member function def
    nodeInfo::nodeInfo() {
        inputs.clear();
        outputs.clear();
    }
    nodeInfo::~nodeInfo() {
        inputs.clear();
        outputs.clear();
    }
    std::string nodeInfo::getNodeType() { return nodeType; }
    void nodeInfo::setNodeType(std::string type) { nodeType = type; }
    std::string nodeInfo::getSubNodeType() { return subNodeType; }
    void nodeInfo::setSubNodeType(std::string type) { subNodeType = type; }
    std::vector<std::string> nodeInfo::getOutputs() { return outputs; }
    std::vector<std::string> nodeInfo::getInputs() { return inputs; }
    void nodeInfo::addInput(std::string input) { inputs.push_back(input); }
    void nodeInfo::addOutput(std::string output) { outputs.push_back(output); }
    void nodeInfo::printNodeInfo() {
        LOG("################### NODE INFO ######################\n");
        LOG("currend node type is %s , sub node type is %s\n", nodeType.c_str(), subNodeType.c_str());
        auto input = getInputs();
        LOG("Input tensor size is %d\n", input.size());
        for(int i = 0; i < input.size(); i++) {
            LOG("----index %d tensor : %s\n", i, input[i].c_str());
        }
        auto output = getOutputs();
        LOG("Output tensor size is %d\n", output.size());
        for(int i = 0; i < output.size(); i++) {
            LOG("----index %d tensor : %s\n", i, output[i].c_str());
        }
    }


    PARSE_NODE_FUNC_DEF(Conv2d)
    PARSE_NODE_FUNC_DEF(ElementWise)
    PARSE_NODE_FUNC_DEF(Activation)
    PARSE_NODE_FUNC_DEF(Shuffle)
    PARSE_NODE_FUNC_DEF(Padding)
    PARSE_NODE_FUNC_DEF(Unary)
    PARSE_NODE_FUNC_DEF(Softmax)
    PARSE_NODE_FUNC_DEF(Reduce)
    PARSE_NODE_FUNC_DEF(Pooling)
    PARSE_NODE_FUNC_DEF(Slice)
    PARSE_NODE_FUNC_DEF(Identity)
    PARSE_NODE_FUNC_DEF(NonZero)
    PARSE_NODE_FUNC_DEF(Shape)
    PARSE_NODE_FUNC_DEF(Gather)
    PARSE_NODE_FUNC_DEF(Unsqueeze)
    PARSE_NODE_FUNC_DEF(Concatenation)
    PARSE_NODE_FUNC_DEF(Gemm)
    PARSE_NODE_FUNC_DEF(Resize)
    PARSE_NODE_FUNC_DEF(BatchNormalization)

    //nodeParseFunc member function def
    nodeParseFunc NodeParse::getNodeParseFunc(std::string onnxNodeType)
    {
        if(nodeParseFuncMap.size() == 0)
            registerNodeParseFunc();
        if(nodeParseFuncMap.count(onnxNodeType) != 0)
            return nodeParseFuncMap[onnxNodeType];
        else
            return (nodeParseFunc)nullptr;
    }
    void NodeParse::registerNodeParseFunc()
    {
        nodeParseFuncMap["Conv"]                      = PARSE_NODE_FUNC(Conv2d);
        nodeParseFuncMap["Add"]                       = PARSE_NODE_FUNC(ElementWise);
        nodeParseFuncMap["Sub"]                       = PARSE_NODE_FUNC(ElementWise);
        nodeParseFuncMap["Mul"]                       = PARSE_NODE_FUNC(ElementWise);
        nodeParseFuncMap["Div"]                       = PARSE_NODE_FUNC(ElementWise);
        nodeParseFuncMap["Max"]                       = PARSE_NODE_FUNC(ElementWise);
        nodeParseFuncMap["Equal"]                     = PARSE_NODE_FUNC(ElementWise);
        nodeParseFuncMap["Greater"]                   = PARSE_NODE_FUNC(ElementWise);
        nodeParseFuncMap["Clip"]                      = PARSE_NODE_FUNC(Activation);
        nodeParseFuncMap["Relu"]                      = PARSE_NODE_FUNC(Activation);
        nodeParseFuncMap["LeakyRelu"]                 = PARSE_NODE_FUNC(Activation);
        nodeParseFuncMap["Sigmoid"]                   = PARSE_NODE_FUNC(Activation);
        nodeParseFuncMap["Softplus"]                  = PARSE_NODE_FUNC(Activation);
        nodeParseFuncMap["Tanh"]                      = PARSE_NODE_FUNC(Activation);
        nodeParseFuncMap["Reshape"]                   = PARSE_NODE_FUNC(Shuffle);
        nodeParseFuncMap["Transpose"]                 = PARSE_NODE_FUNC(Shuffle);
        nodeParseFuncMap["Flatten"]                   = PARSE_NODE_FUNC(Shuffle);
        nodeParseFuncMap["Pad"]                       = PARSE_NODE_FUNC(Padding);
        nodeParseFuncMap["Sqrt"]                      = PARSE_NODE_FUNC(Unary);
        nodeParseFuncMap["Reciprocal"]                = PARSE_NODE_FUNC(Unary);
        nodeParseFuncMap["Abs"]                       = PARSE_NODE_FUNC(Unary);
        nodeParseFuncMap["Exp"]                       = PARSE_NODE_FUNC(Unary);
        nodeParseFuncMap["Softmax"]                   = PARSE_NODE_FUNC(Softmax);
        nodeParseFuncMap["ReduceSum"]                 = PARSE_NODE_FUNC(Reduce);
        nodeParseFuncMap["GlobalAveragePool"]         = PARSE_NODE_FUNC(Reduce);
        nodeParseFuncMap["MaxPool"]                   = PARSE_NODE_FUNC(Pooling);
        nodeParseFuncMap["AveragePool"]               = PARSE_NODE_FUNC(Pooling);
        nodeParseFuncMap["Slice"]                     = PARSE_NODE_FUNC(Slice);
        nodeParseFuncMap["Cast"]                      = PARSE_NODE_FUNC(Identity);
        nodeParseFuncMap["NonZero"]                   = PARSE_NODE_FUNC(NonZero);
        nodeParseFuncMap["Shape"]                     = PARSE_NODE_FUNC(Shape);
        nodeParseFuncMap["Gather"]                    = PARSE_NODE_FUNC(Gather);
        nodeParseFuncMap["Unsqueeze"]                 = PARSE_NODE_FUNC(Unsqueeze);
        nodeParseFuncMap["Concat"]                    = PARSE_NODE_FUNC(Concatenation);
        nodeParseFuncMap["Gemm"]                      = PARSE_NODE_FUNC(Gemm);
        nodeParseFuncMap["Resize"]                    = PARSE_NODE_FUNC(Resize);
        nodeParseFuncMap["BatchNormalization"]        = PARSE_NODE_FUNC(BatchNormalization);
    }
    NodeParse* NodeParse::instance = new NodeParse;


    nodeParseFunc getNodeParseFuncMap(std::string onnxNodeType)
    {
        auto instance = NodeParse::getInstance();
        if(instance != nullptr)
            return instance->getNodeParseFunc(onnxNodeType);
        else
            return (nodeParseFunc)nullptr;
    }

}