/********************************************
 * Filename: graph_parse_engine.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/logger.hpp"
#include "parser/graph_parser.hpp"
#include "engine/graph_parse_engine.hpp"

namespace TENSORRT_WRAPPER
{

    GraphParseEngine::GraphParseEngine(const std::string json_file, const std::string weight_file)
    {
        m_graph_parser.reset(new GraphParser(json_file, weight_file));
        if (nullptr == m_graph_parser.get() || m_graph_parser.get()->getInitFlag())
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "init from json_file:{} and weight_file:{} fail", json_file, weight_file);
            m_graph_parser.reset();
        }
        return;
    }

    bool GraphParseEngine::saveEngineFile(const std::string save_file);
    {
        if (nullptr == m_graph_parser.get())
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "graph parser not init ok, please check log for more info");
            return false;
        }
        return true;
    }

} // namespace TENSORRT_WRAPPER