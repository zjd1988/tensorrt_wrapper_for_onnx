/********************************************
 * Filename: node_parser.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "node/node_info.hpp"

namespace TENSORRT_WRAPPER
{

    typedef NodeInfo* (*NodeParserFunc)(std::string, Json::Value&);

    class NodeParser
    {
    public:
        const std::map<std::string, NodeParserFunc>& getNodeParserFuncMap(const std::string node_type)
        {
            return m_parser_func_map;
        }

        static NodeParser* getInstance()
        {
            static NodeParser instance;
            return &instance;
        }

    private:
        void registerNodeParserFuncMap();
        NodeParser()
        {
            registerNodeParserFuncMap();
        };

    private:
        std::map<std::string, NodeParserFunc>            m_parser_func_map;
    };

    NodeParserFunc getNodeParserFunc(const std::string node_type);

} // namespace TENSORRT_WRAPPER