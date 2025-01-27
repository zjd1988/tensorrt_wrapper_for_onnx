#ifndef __NORMALIZATION_EXECUTION_INFO_HPP__
#define __NORMALIZATION_EXECUTION_INFO_HPP__
#include "execution_info.hpp"

namespace tensorrtInference
{
    class NormalizationExecutionInfo : public ExecutionInfo
    {
    public:
        NormalizationExecutionInfo(CUDARuntime *runtime, 
            std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root);
        ~NormalizationExecutionInfo();
        bool init(Json::Value& root) override;
        void run() override;
    private:
        float alpha;
        float beta;
        float bias;
        int blockSize;
        int gridSize;
        int totalElementSize;
        Buffer* srcTensor;
        Buffer* dstTensor;        
    };
} // namespace tensorrtInference 

#endif