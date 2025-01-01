/********************************************
 * Filename: execution_factory.cpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#include "common/logger.hpp"
#include "execution/execution_factory.hpp"

namespace TENSORRT_WRAPPER
{

    BaseExecution* ExecutionFactory::create(const std::string type, CUDARuntime *runtime, Json::Value& root)
    {
        if (gEngineTypeToString)
        auto creator = getExecutionCreator(type);
        if (nullptr == creator)
        {
            logRegisteredExecutionCreator();
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "have no engine creator for type: {}", int(type));
            return nullptr;
        }
        auto engine = creator->onCreate(runtime, root);
        if (nullptr == engine)
        {
            TRT_WRAPPER_LOG(TRT_WRAPPER_LOG_LEVEL_ERROR, "create {} engine failed, creator return nullptr", 
                gEngineTypeToString[type]);
        }
        return engine;
    }

} // namespace TENSORRT_WRAPPER