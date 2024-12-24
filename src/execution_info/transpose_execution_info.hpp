#ifndef __TRANSSPOSE_EXECUTION_INFO_HPP__
#define __TRANSSPOSE_EXECUTION_INFO_HPP__
#include "execution_info.hpp"
#define TRANSPOSE_MAX_DIMENSION 4

namespace TENSORRT_WRAPPER
{
    class TransposeExecutionInfo : public ExecutionInfo
    {
    public:
        TransposeExecutionInfo(CUDARuntime *runtime, 
            std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root);
        ~TransposeExecutionInfo();
        bool init(Json::Value& root) override;
        void run() override;
    private:
        int blockSize;
        int gridSize;
        int totalElementSize;
        int shapeSize;
        std::shared_ptr<Buffer> inputShape;
        std::shared_ptr<Buffer> inputAxis;
        Buffer* srcTensor;
        Buffer* dstTensor;
    };
} // namespace TENSORRT_WRAPPER

#endif