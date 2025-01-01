/********************************************
 * Filename: execution_register.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include <mutex>
#include "execution/base_execution.h"

namespace TENSORRT_WRAPPER
{

    extern registerTrtEngineExecutionCreator();

    static std::once_flag s_flag;
    void registerExecutionCreator()
    {
        std::call_once(s_flag, [&]() {
            registerTrtEngineExecutionCreator();
            logRegisteredExecutionCreator();
        });
    }

} // namespace TENSORRT_WRAPPER