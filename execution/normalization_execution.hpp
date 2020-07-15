#ifndef __NORMALIZATION_EXECUTION_HPP__
#define __NORMALIZATION_EXECUTION_HPP__
#include "execution.hpp"

namespace tensorrtInference
{
    class NormalizationExecution : public Execution
    {
    public:
        NormalizationExecution(CUDARuntime *runtime, std::string subType);
        ~NormalizationExecution();
        bool init(std::vector<Buffer*> inputBuffers) override;
        void run(bool sync = false) override;
    private:
        bool needMemCpy = false;
    };
} // namespace tensorrtInference 

#endif