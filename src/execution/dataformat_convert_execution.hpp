/********************************************
 * Filename: dataformat_convert_execution.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include "execution_info.hpp"

namespace TENSORRT_WRAPPER
{

    class DataFormatConvertExecution : public BaseExecution
    {
    public:
        DataFormatConvertExecutionInfo(CUDARuntime *runtime, Json::Value& root);
        ~DataFormatConvertExecutionInfo() = default;
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

} // namespace TENSORRT_WRAPPER