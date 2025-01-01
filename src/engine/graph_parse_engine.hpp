/********************************************
 * Filename: graph_parse_engine.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <iostream>
#include <string>

namespace TENSORRT_WRAPPER
{

    class GraphParser;

    class GraphParseEngine
    {
    public:
        GraphParseEngine(const std::string json_file, const std::string weight_file);
        ~GraphParseEngine() = default;
        bool saveEngineFile(const std::string save_file);

    private:
        std::shared_ptr<GraphParser>        m_graph_parser;
    };

} // namespace TENSORRT_WRAPPER