#ifndef __CONVERT_EXECUTION_HPP__
#define __CONVERT_EXECUTION_HPP__
#include "execution.hpp"

namespace tensorrtInference
{
    class ConvertExecution : public Execution
    {
    public:
        ConvertExecution(CUDARuntime *runtime, std::string subType);
        ~ConvertExecution();
        bool init(std::vector<Buffer*> inputBuffers) override;
        void run(bool sync = false) override;
    private:
        bool needMemCpy = false;
    };
} // namespace tensorrtInference 

#endif