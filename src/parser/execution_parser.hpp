/********************************************
 * Filename: execution_parser.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "json/json.h"
#include "common/utils.hpp"
#include "execution/base_execution.hpp"

namespace TENSORRT_WRAPPER
{

    class ExecutionParser
    {
    public:
        ExecutionParser(CUDARuntime *cuda_runtime, std::string &json_file);
        ~ExecutionParser() = default;
        const std::vector<std::string>& getTopoOrder() { return m_topo_order; }
        const std::map<std::string, std::shared_ptr<Buffer>>& getTensorBuffer() { return m_tensor_buffer; }
        const std::map<std::string, std::shared_ptr<BaseExecution>>& getExecutionInfoMap() { return m_execution_info_map; }
        bool getInitFlag() { return m_init_flag; }
        void runInference();
        std::map<std::string, void*> getInferenceResult();

    private:
        bool extractExecutionInfo(Json::Value &root);
        CUDARuntime* getCudaRuntime() { return m_cuda_runtime; }

    private:
        std::vector<std::string>                                 m_topo_order;
        std::map<std::string, std::shared_ptr<BaseExecution>>    m_execution_map;
        std::map<std::string, std::shared_ptr<Buffer>>           m_tensor_buffer;
        std::vector<std::string>                                 m_input_names;
        std::vector<std::string>                                 m_output_names;
        CUDARuntime*                                             m_cuda_runtime;
        bool                                                     m_init_flag = false;
    };

} // namespace TENSORRT_WRAPPER
