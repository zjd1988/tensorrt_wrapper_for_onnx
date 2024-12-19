#ifndef __DATAFORMAT_CONVERT_EXECUTION_HPP__
#define __DATAFORMAT_CONVERT_EXECUTION_HPP__
#include "execution_info.hpp"

namespace tensorrtInference
{
    class DataFormatConvertExecutionInfo : public ExecutionInfo
    {
    public:
        DataFormatConvertExecutionInfo(CUDARuntime *runtime, 
            std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root);
        ~DataFormatConvertExecutionInfo();
        bool init(Json::Value& root) override;
        void run() override;
    private:
        std::string convertType;
        int blockSize;
        int gridSize;
        int totalElementSize;        
        Buffer* srcTensor;
        Buffer* dstTensor;
    };
} // namespace tensorrtInference 

#endif