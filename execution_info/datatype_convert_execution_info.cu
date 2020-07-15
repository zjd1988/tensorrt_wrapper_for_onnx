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

    DataTypeConvertExecutionInfo::DataTypeConvertExecutionInfo(CUDARuntime *runtime, Json::Value& root) : ExecutionInfo(runtime, root)
    {
    }
    
    DataTypeConvertExecutionInfo::~DataTypeConvertExecutionInfo()
    {
    }

    bool DataTypeConvertExecutionInfo::init(std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root)
    {
        convertType = root["attr"]["convert_type"];
        if(convertType.compare("ConvertUint8ToFloat32") == 0 && convertType.compare("ConvertUint8ToFloat16") == 0)
            LOG("current not support %s\n", convertType.c_str());
        return true;
    }

    void DataTypeConvertExecutionInfo::run(bool sync)
    {
        auto inputTensorNames = getInputTensorNames();
        auto outputTensorNames = getOutputTensorNames();
        auto runtime = getCudaRuntime();
        auto stream = runtime->stream();
        auto srcTensor = tensors[inputTensorNames[0]];
        auto dstTensor = tensors[outputTensorNames[0]];
        const int size = srcTensor->getElementCount();

        if(needMemCpy)
            runtime->copyToDevice(srcTensor.get(), srcTensor.get());

        int blockSize = runtime->threads_num();
        int gridSize = size / blockSize + 1;
        if(convertType.compare("ConvertUint8ToFloat32") == 0)
        {
            DataTypeConvertExecutionKernel<<<gridSize, blockSize, 0, stream>>>(srcTensor->device<const unsigned char>(),
                dstTensor->device<float>(), size);
        }
        else if(convertType.compare("ConvertUint8ToFloat16") == 0)
        {
            DataTypeConvertExecutionKernel<<<gridSize, blockSize, 0, stream>>>(srcTensor->device<const unsigned char>(),
                dstTensor->device<float>(), size);
        }
        else
        {
            CHECK_ASSERT(false, "current not support (%s) \n", getSubExecutionType().c_str());
        }
        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch data type convert kernel(%s) fail: %s\n", convertType.c_str(),
            cudaGetErrorString(cudastatus));
        if(sync)
            runtime->onWaitFinish();
        return;
    }
}