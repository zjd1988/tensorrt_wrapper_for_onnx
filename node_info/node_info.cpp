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

#define PARSE_NODE_FUNC(nodeType)                                          \
nodeInfo* parse##nodeType##NodeInfo(std::string type, Json::Value& root)   \
{                                                                          \
    nodeInfo* node = new nodeType##NodeInfo();                             \
    if(node->parseNodeInfoFromJson(type, root))                            \
        return node;                                                       \
    else                                                                   \
        delete node;                                                       \
    return nullptr;                                                        \
}



namespace tensorrtInference 
{
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
    // std::map<std::string, nodeParseFunc> nodeParseFuncMap;
    // static void registerNodeParseFunc(std::string nodeType, nodeParseFunc func)
    // {    
    //     if(nodeParseFuncMap.count(nodeType) == 0)
    //         nodeParseFuncMap[nodeType] = func;
    //     else
    //         std::cout << "already register" << nodeType << "parse func" << std::endl;
    // }

    class NodeParse
    {
    private:
        static NodeParse* instance;
        std::map<std::string, nodeParseFunc> nodeParseFuncMap;
        std::map<std::string, std::string> onnxNodeTypeToTensorrtNodeTypeMap;
        void registerNodeParseFunc();
        NodeParse()
        {
        }
    public:
        nodeParseFunc getNodeParseFunc(std::string nodeType);
        static NodeParse* getInstance() {
            return instance;
        }
    };
    nodeParseFunc NodeParse::getNodeParseFunc(std::string onnxNodeType)
    {
        std::string tensorrtNodeType;
        if(onnxNodeTypeToTensorrtNodeTypeMap.size() == 0)
            registerNodeParseFunc();
        tensorrtNodeType = onnxNodeTypeToTensorrtNodeTypeMap[onnxNodeType];
        if(nodeParseFuncMap.count(tensorrtNodeType) != 0)
            return nodeParseFuncMap[tensorrtNodeType];
        else
            (nodeParseFunc)nullptr;
    }
    void NodeParse::registerNodeParseFunc()
    {
        onnxNodeTypeToTensorrtNodeTypeMap["Conv"]             = "Conv2d";
        onnxNodeTypeToTensorrtNodeTypeMap["Add"]              = "ElementWise";
        onnxNodeTypeToTensorrtNodeTypeMap["Sub"]              = "ElementWise";
        onnxNodeTypeToTensorrtNodeTypeMap["Mul"]              = "ElementWise";
        onnxNodeTypeToTensorrtNodeTypeMap["Div"]              = "ElementWise";
        onnxNodeTypeToTensorrtNodeTypeMap["Max"]              = "ElementWise";
        onnxNodeTypeToTensorrtNodeTypeMap["Equal"]            = "ElementWise";
        onnxNodeTypeToTensorrtNodeTypeMap["Greater"]          = "ElementWise";
        onnxNodeTypeToTensorrtNodeTypeMap["Clip"]             = "Activation";
        onnxNodeTypeToTensorrtNodeTypeMap["Reshape"]          = "Shuffle";
        onnxNodeTypeToTensorrtNodeTypeMap["Transpose"]        = "Shuffle";
        onnxNodeTypeToTensorrtNodeTypeMap["Pad"]              = "Padding";
        onnxNodeTypeToTensorrtNodeTypeMap["Sqrt"]             = "Unary";
        onnxNodeTypeToTensorrtNodeTypeMap["Reciprocal"]       = "Unary";
        onnxNodeTypeToTensorrtNodeTypeMap["Abs"]              = "Unary";
        onnxNodeTypeToTensorrtNodeTypeMap["Softmax"]          = "Softmax";
        onnxNodeTypeToTensorrtNodeTypeMap["ReduceSum"]        = "Reduce";
        onnxNodeTypeToTensorrtNodeTypeMap["MaxPool"]          = "Pooling";
        onnxNodeTypeToTensorrtNodeTypeMap["AveragePool"]      = "Pooling";
        onnxNodeTypeToTensorrtNodeTypeMap["Slice"]            = "Slice";
        onnxNodeTypeToTensorrtNodeTypeMap["Cast"]             = "Identity";
        onnxNodeTypeToTensorrtNodeTypeMap["NonZero"]          = "NonZero";
        onnxNodeTypeToTensorrtNodeTypeMap["Shape"]            = "Shape";
        onnxNodeTypeToTensorrtNodeTypeMap["Gather"]           = "Gather";
        onnxNodeTypeToTensorrtNodeTypeMap["Unsqueeze"]        = "Unsqueeze";
    }
    NodeParse* NodeParse::instance = new NodeParse;

   
    PARSE_NODE_FUNC(Conv2d)
    PARSE_NODE_FUNC(ElementWise)
    PARSE_NODE_FUNC(Activation)
    PARSE_NODE_FUNC(Shuffle)
    PARSE_NODE_FUNC(Padding)
    PARSE_NODE_FUNC(Unary)
    PARSE_NODE_FUNC(Softmax)
    PARSE_NODE_FUNC(Reduce)
    PARSE_NODE_FUNC(Pooling)
    PARSE_NODE_FUNC(Slice)
    PARSE_NODE_FUNC(Identity)
    PARSE_NODE_FUNC(NonZero)
    PARSE_NODE_FUNC(Shape)
    PARSE_NODE_FUNC(Gather)
    PARSE_NODE_FUNC(Unsqueeze)


    nodeParseFunc getNodeParseFuncMap(std::string onnxNodeType)
    {
        std::string tensorrtNodeType;
        auto instance = NodeParse::getInstance();
        if(instance != nullptr)
            return instance->getNodeParseFunc(onnxNodeType);
        else
            return (nodeParseFunc)nullptr;
    }

}