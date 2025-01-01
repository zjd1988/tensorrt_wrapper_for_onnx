/********************************************
 * Filename: execution_common.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include <map>
#include <string>
#include "execution/execution_common.hpp"

namespace TENSORRT_WRAPPER
{

    // string to engine type map
    std::map<std::string, ExecutionType> gStringToExecutionType = {
        {"TrtEngine",                        TRT_ENGINE_EXECUTION_TYPE},
    };

    // engine type to string map
    std::map<ExecutionType, std::string> gExecutionTypeToString = {
        {TRT_ENGINE_EXECUTION_TYPE,          "TrtEngine"},
    };

} // namespace TENSORRT_WRAPPER