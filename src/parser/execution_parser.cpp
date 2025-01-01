/********************************************
 * Filename: execution_parse.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include <fstream>
#include "parser/execution_parser.hpp"
using namespace std;

namespace TENSORRT_WRAPPER
{

    ExecutionParser::ExecutionParser(CUDARuntime *cuda_runtime, std::string &json_file)
    {
        CHECK_ASSERT(nullptr != cuda_runtime, "cuda runtime is null!\n");
        m_cuda_runtime = runtime;
        std::ifstream json_fstream;
        json_fstream.open(json_file);
        if(!json_fstream.is_open())
        {
            std::cout << "open json file " << json_file << " fail!!!" << std::endl;
            return;
        }
        Json::Reader reader;
        Json::Value root;
        if (!reader.parse(json_fstream, root, false))
        {
            std::cout << "parse json file " << json_file << " fail!!!" << std::endl;
            json_fstream.close();
            return;
        }
        json_fstream.close();

        //extract topo node order
        {
            int size = root["topo_order"].size();
            for(int i = 0; i < size; i++)
            {
                std::string execution_name;
                execution_name = root["topo_order"][i].asString();
                m_topo_order.push_back(execution_name);
            }
        }

        //extract input tensor names
        {
            int input_size = root["input_tensor_names"].size();
            for(int i = 0; i < input_size; i++)
            {
                std::string tensor_name;
                tensor_name = root["input_tensor_names"][i].asString();
                inputTensorNames.push_back(tensor_name);
            }
        }

        //extract output tensor names
        {
            int output_size = root["output_tensor_names"].size();
            for(int i = 0; i < output_size; i++)
            {
                std::string tensor_name;
                tensor_name = root["output_tensor_names"][i].asString();
                m_output_names.push_back(tensor_name);
            } 
        }

        // extract execution execution info
        m_init_flag = extractExecutionInfo(root["execution_info"]);
        return;
    }

    bool ExecutionParser::extractExecutionInfo(Json::Value &root)
    {
        CUDARuntime *runtime = getCudaRuntime();
        for (int i = 0; i < m_topo_order.size(); i++) 
        {
            auto elem = m_topo_order[i];
            auto execution_type = root[elem]["type"].asString();
            auto parse_func = getConstructExecutionInfoFuncMap(execution_type);
            if(parse_func != nullptr)
            {
                auto curr_execution = parse_func(runtime, tensorsInfo, root[elem]);
                if(curr_execution == nullptr)
                    return false;
                curr_execution->init(root[elem]);
                m_execution_info_map[elem].reset(curr_execution);
            }
            else
            {
                LOG("current not support %s execution type\n", execution_type.c_str());
            }
        }
        return true;
    }

    void ExecutionParser::runInference()
    {
        size_t execution_size = m_topo_order.size();
        for(size_t i = 0; i < execution_size; i++)
        {
            auto name = m_topo_order[i];
            m_execution_info_map[name]->run();
        }
        return;
    }

    std::map<std::string, void*> ExecutionParser::getInferenceResult()
    {
        int size = m_output_names.size();
        std::map<std::string, void*> result;
        for( int i = 0; i < size; i++)
        {
            auto name = m_output_names[i];
            result[name] = tensorsInfo[name]->host<void>();
        }
        return result;
    }

} // namespace TENSORRT_WRAPPER