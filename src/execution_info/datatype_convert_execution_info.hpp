#ifndef __DATATYPE_CONVERT_EXECUTION_INFO_HPP__
#define __DATATYPE_CONVERT_EXECUTION_INFO_HPP__
#include "execution_info.hpp"

namespace tensorrtInference
{
    class DataTypeConvertExecutionInfo : public ExecutionInfo
    {
    public:
        DataTypeConvertExecutionInfo(CUDARuntime *runtime, 
            std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root);
        ~DataTypeConvertExecutionInfo();
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