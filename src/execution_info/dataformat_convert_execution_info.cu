#include "dataformat_convert_execution_info.hpp"

namespace TENSORRT_WRAPPER
{
    template <typename T>
    __global__ void BGR2RGBExecutionKernel(const T* src, T* dst, const int size, const int channel)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < size){
            int mod = channel - 1;
            int new_pos = index / channel * channel + mod - index % channel;
            dst[new_pos] = src[index];
        }
    }

    DataFormatConvertExecutionInfo::DataFormatConvertExecutionInfo(CUDARuntime *runtime,
        std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root) : ExecutionInfo(runtime, tensorsInfo, root)
    {
        convertType = "";
        blockSize = 0;
        gridSize = 0;
        totalElementSize = 0;
        srcTensor = nullptr;
        dstTensor = nullptr;        
    }
    
    DataFormatConvertExecutionInfo::~DataFormatConvertExecutionInfo()
    {
    }

    bool DataFormatConvertExecutionInfo::init(Json::Value& root)
    {
        convertType = root["attr"]["convert_type"].asString();
        if(convertType.compare("BGR2RGB") != 0)
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
        recycleBuffers();
        return true;
    }

    void DataFormatConvertExecutionInfo::run()
    {
        auto runtime = getCudaRuntime();
        auto stream = runtime->stream();
        beforeRun();

        if(convertType.compare("BGR2RGB") == 0)
        {
            BGR2RGBExecutionKernel<<<gridSize, blockSize, 0, stream>>>(srcTensor->device<const unsigned char>(),
                dstTensor->device<unsigned char>(), totalElementSize, 3);
        }
        else
        {
            CHECK_ASSERT(false, "current not support (%s) \n", convertType.c_str());
        }
        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch data format convert kernel(%s) fail: %s\n", convertType.c_str(),
            cudaGetErrorString(cudastatus));

        afterRun();
        return;
    }
}