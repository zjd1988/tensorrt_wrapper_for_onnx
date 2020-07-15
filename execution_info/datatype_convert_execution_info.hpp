#ifndef __DATATYPE_CONVERT_EXECUTION_INFO_HPP__
#define __DATATYPE_CONVERT_EXECUTION_INFO_HPP__
#include "execution_info.hpp"

namespace tensorrtInference
{
    class DataTypeConvertExecutionInfo : public ExecutionInfo
    {
    public:
        DataTypeConvertExecutionInfo(CUDARuntime *runtime, Json::Value& root);
        ~DataTypeConvertExecutionInfo();
        bool init(std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root) override;
        void run(bool sync = false) override;
    private:
        bool needMemCpy = false;
        std::string convertType;
    };
} // namespace tensorrtInference 

#endif