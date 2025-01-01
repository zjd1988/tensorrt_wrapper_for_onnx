/********************************************
 * Filename: execution_engine.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <iostream>
#include <string>

namespace TENSORRT_WRAPPER
{

    class ExecutionEngine
    {
    public:
        ExecutionEngine(const std::string engine_file, int gpu_id = 0);
        ~ExecutionEngine();
    
    private:
        
    };

} // namespace TENSORRT_WRAPPER