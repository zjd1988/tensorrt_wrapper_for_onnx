#ifndef __FORMAT_CONVERT_EXECUTION_HPP__
#define __FORMAT_CONVERT_EXECUTION_HPP__
#include "execution.hpp"

namespace tensorrtInference
{
    class FormatConvertExecution : public Execution
    {
    public:
        FormatConvertExecution(CUDARuntime *runtime, std::string subType);
        ~FormatConvertExecution();
        bool init(std::vector<Buffer*> inputBuffers) override;
        void run(bool sync = false) override;
        void callFormatConvertExecutionKernel(Buffer* src, Buffer* dst, std::string &convertType, CUDARuntime *runtime);
    private:
        bool needMemCpy = false;
    };
} // namespace tensorrtInference 

#endif