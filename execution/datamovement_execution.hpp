#ifndef __DATAMOVEMENT_EXECUTION_HPP__
#define __DATAMOVEMENT_EXECUTION_HPP__
#include "execution.hpp"

namespace tensorrtInference
{
    class DataMovementExecution : public Execution
    {
    public:
        DataMovementExecution(CUDARuntime *runtime, std::string subType);
        ~DataMovementExecution();
        bool init(std::vector<Buffer*> inputBuffers) override;
        void run(bool sync = false) override;
    private:
        bool needMemCpy = false;
    };
} // namespace tensorrtInference 

#endif