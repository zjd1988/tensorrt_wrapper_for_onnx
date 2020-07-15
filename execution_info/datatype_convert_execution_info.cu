#include "datatype_convert_execution_info.hpp"

namespace tensorrtInference
{
    template <typename SRC_T, typename DST_T>
    __global__ void DataTypeConvertExecutionKernel(const SRC_T* src, DST_T* dst, const int size)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < size){
            dst[index] = (DST_T)src[index];
        }
    }

    DataTypeConvertExecutionInfo::DataTypeConvertExecutionInfo(CUDARuntime *runtime,
        std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root) : ExecutionInfo(runtime, tensorsInfo, root)
    {
        convertType = "";
        blockSize = 0;
        gridSize = 0;
        totalElementSize = 0;
        srcTensor = nullptr;
        dstTensor = nullptr;
    }
    
    DataTypeConvertExecutionInfo::~DataTypeConvertExecutionInfo()
    {
    }

    bool DataTypeConvertExecutionInfo::init(Json::Value& root)
    {
        convertType = root["attr"]["convert_type"].asString();
        if(convertType.compare("ConvertUint8ToFloat32") != 0 && convertType.compare("ConvertUint8ToFloat16") != 0)
            LOG("current not support %s\n", convertType.c_str());

        auto runtime = getCudaRuntime();
        auto srcTensorNames = getInputTensorNames();
        auto dstTensorNames = getOutputTensorNames();
        CHECK_ASSERT(srcTensorNames.size() == dstTensorNames.size(), "input tensor size should be equal to output!\n");
        CHECK_ASSERT(srcTensorNames.size() == 1, "input tensor size should be equal to 1!\n");
        auto tensorsInfo = getTensorsInfo();
        srcTensor = tensorsInfo[srcTensorNames[0]];
        dstTensor = tensorsInfo[dstTensorNames[0]];
        totalElementSize = srcTensor->getElementCount();
        blockSize = runtime->threads_num();
        gridSize = DIVUP(totalElementSize, blockSize);
        return true;
    }

    void DataTypeConvertExecutionInfo::run()
    {
        auto runtime = getCudaRuntime();
        auto stream = runtime->stream();
        beforeRun();
        if(convertType.compare("ConvertUint8ToFloat32") == 0)
        {
            DataTypeConvertExecutionKernel<<<gridSize, blockSize, 0, stream>>>(srcTensor->device<const unsigned char>(),
                dstTensor->device<float>(), totalElementSize);
        }
        else if(convertType.compare("ConvertUint8ToFloat16") == 0)
        {
            DataTypeConvertExecutionKernel<<<gridSize, blockSize, 0, stream>>>(srcTensor->device<const unsigned char>(),
                dstTensor->device<float>(), totalElementSize);
        }
        else
        {
            CHECK_ASSERT(false, "current not support (%s) \n", convertType.c_str());
        }
        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch data type convert kernel(%s) fail: %s\n", convertType.c_str(),
            cudaGetErrorString(cudastatus));
        afterRun();
        return;
    }
}