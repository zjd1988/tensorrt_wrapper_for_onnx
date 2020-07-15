#ifndef __TRANSSPOSE_EXECUTION_HPP__
#define __TRANSSPOSE_EXECUTION_HPP__
#include "execution.hpp"
#define TRANSPOSE_MAX_DIMENSION 4

namespace tensorrtInference
{
    class TransposeExecution : public Execution
    {
    public:
        TransposeExecution(CUDARuntime *runtime, std::string subType);
        ~TransposeExecution();
        bool init(std::vector<Buffer*> inputBuffers) override;
        void run(bool sync = false) override;
    private:
        bool needMemCpy = false;
    };
} // namespace tensorrtInference 

#endif