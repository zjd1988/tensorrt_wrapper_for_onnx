#ifndef __DATA_CONVERT_EXECUTION_HPP__
#define __DATA_CONVERT_EXECUTION_HPP__
#include "execution.hpp"

namespace tensorrtInference
{
    class DataConvertExecution : public Execution
    {
    public:
        DataConvertExecution(CUDARuntime *runtime, std::string subType);
        ~DataConvertExecution();
        bool init(std::vector<Buffer*> inputBuffers) override;
        void run(bool sync = false) override;
    private:
        bool needMemCpy = false;
    };
} // namespace tensorrtInference 

#endif