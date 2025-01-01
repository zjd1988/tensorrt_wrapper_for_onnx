/********************************************
 * Filename: execution_factory.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "execution/base_execution.hpp"

namespace TENSORRT_WRAPPER
{

    /** execution factory */
    class ExecutionFactory
    {
    public:
        static BaseExecution* create(const std::string type, CUDARuntime *runtime, Json::Value& root);
    };

} // namespace TENSORRT_WRAPPER