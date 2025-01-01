/********************************************
 * Filename: execution_common.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once

namespace TENSORRT_WRAPPER
{

    // execution type enum
    typedef enum ExecutionType
    {
        INVALID_EXECUTION_TYPE                      = -1,
        TRT_ENGINE_EXECUTION_TYPE                   = 0,
        MAX_EXECUTION_TYPE,
    } ExecutionType;

    // string -> execution type / execution type -> string
    extern std::map<std::string, ExecutionType> gStringToExecutionType;
    extern std::map<ExecutionType, std::string> gExecutionTypeToString;

} // namespace TENSORRT_WRAPPER