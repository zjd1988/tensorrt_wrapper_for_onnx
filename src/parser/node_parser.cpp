/********************************************
 * Filename: node_parser.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "parser/node_parser.hpp"


#define PARSE_NODE_FUNC_DEF(NodeType)                                             \
static NodeInfo* parse##NodeType##NodeInfo(std::string type, Json::Value& root)   \
{                                                                                 \
    std::unique_ptr<NodeInfo> node_info(new NodeType##NodeInfo());                \
    if (nullptr == node_info.get())                                               \

    else                                                                          \
    {                                                                             \
        if (node_info->parseNodeInfoFromJson(type, root))                         \
            return node_info.release();                                           \
        else                                                                      \

    }                                                                             \
    return nullptr;                                                               \
}

#define PARSE_NODE_FUNC(NodeType) parse##NodeType##NodeInfo

namespace TENSORRT_WRAPPER
{

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

    void NodeParser::registerNodeParserFuncMap()
    {
        m_parser_func_map["Conv"]                      = PARSE_NODE_FUNC(Conv2d);
        m_parser_func_map["Add"]                       = PARSE_NODE_FUNC(ElementWise);
        m_parser_func_map["Sub"]                       = PARSE_NODE_FUNC(ElementWise);
        m_parser_func_map["Mul"]                       = PARSE_NODE_FUNC(ElementWise);
        m_parser_func_map["Div"]                       = PARSE_NODE_FUNC(ElementWise);
        m_parser_func_map["Max"]                       = PARSE_NODE_FUNC(ElementWise);
        m_parser_func_map["Equal"]                     = PARSE_NODE_FUNC(ElementWise);
        m_parser_func_map["Greater"]                   = PARSE_NODE_FUNC(ElementWise);
        m_parser_func_map["Clip"]                      = PARSE_NODE_FUNC(Activation);
        m_parser_func_map["Relu"]                      = PARSE_NODE_FUNC(Activation);
        m_parser_func_map["LeakyRelu"]                 = PARSE_NODE_FUNC(Activation);
        m_parser_func_map["Sigmoid"]                   = PARSE_NODE_FUNC(Activation);
        m_parser_func_map["Softplus"]                  = PARSE_NODE_FUNC(Activation);
        m_parser_func_map["Tanh"]                      = PARSE_NODE_FUNC(Activation);
        m_parser_func_map["Reshape"]                   = PARSE_NODE_FUNC(Shuffle);
        m_parser_func_map["Transpose"]                 = PARSE_NODE_FUNC(Shuffle);
        m_parser_func_map["Flatten"]                   = PARSE_NODE_FUNC(Shuffle);
        m_parser_func_map["Pad"]                       = PARSE_NODE_FUNC(Padding);
        m_parser_func_map["Sqrt"]                      = PARSE_NODE_FUNC(Unary);
        m_parser_func_map["Reciprocal"]                = PARSE_NODE_FUNC(Unary);
        m_parser_func_map["Abs"]                       = PARSE_NODE_FUNC(Unary);
        m_parser_func_map["Exp"]                       = PARSE_NODE_FUNC(Unary);
        m_parser_func_map["Softmax"]                   = PARSE_NODE_FUNC(Softmax);
        m_parser_func_map["ReduceSum"]                 = PARSE_NODE_FUNC(Reduce);
        m_parser_func_map["GlobalAveragePool"]         = PARSE_NODE_FUNC(Reduce);
        m_parser_func_map["MaxPool"]                   = PARSE_NODE_FUNC(Pooling);
        m_parser_func_map["AveragePool"]               = PARSE_NODE_FUNC(Pooling);
        m_parser_func_map["Slice"]                     = PARSE_NODE_FUNC(Slice);
        m_parser_func_map["Cast"]                      = PARSE_NODE_FUNC(Identity);
        m_parser_func_map["NonZero"]                   = PARSE_NODE_FUNC(NonZero);
        m_parser_func_map["Shape"]                     = PARSE_NODE_FUNC(Shape);
        m_parser_func_map["Gather"]                    = PARSE_NODE_FUNC(Gather);
        m_parser_func_map["Unsqueeze"]                 = PARSE_NODE_FUNC(Unsqueeze);
        m_parser_func_map["Concat"]                    = PARSE_NODE_FUNC(Concatenation);
        m_parser_func_map["Gemm"]                      = PARSE_NODE_FUNC(Gemm);
        m_parser_func_map["Resize"]                    = PARSE_NODE_FUNC(Resize);
        m_parser_func_map["BatchNormalization"]        = PARSE_NODE_FUNC(BatchNormalization);
    }

    NodeParserFunc getNodeParserFunc(const std::string node_type)
    {
        auto instance = NodeParser::getInstance();
        if(nullptr != instance)
        {
            auto func_map = instance->getNodeParserFuncMap();
            if(func_map.end() != func_map.find(node_type))
                return func_map[node_type];
            else
                return (NodeParserFunc)nullptr;
        }
        return (NodeParserFunc)nullptr;
    }

} // namespace TENSORRT_WRAPPER