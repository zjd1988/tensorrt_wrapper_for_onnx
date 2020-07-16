#ifndef __RESHAPE_EXECUTION_INFO_HPP__
#define __RESHAPE_EXECUTION_INFO_HPP__
#include "execution_info.hpp"

namespace tensorrtInference
{
    class ReshapeExecutionInfo : public ExecutionInfo
    {
    public:
        ReshapeExecutionInfo(CUDARuntime *runtime, 
            std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root);
        ~ReshapeExecutionInfo();
        bool init(Json::Value& root) override;
        void run() override;
    private:
        std::vector<int> newShape;
        int totalElementSize;
        Buffer* srcTensor;
        Buffer* dstTensor;
    };
} // namespace tensorrtInference 

#endif