#include "normalization_execution_info.hpp"

namespace TENSORRT_WRAPPER
{
    __global__ void NormalizationExecutionKernel(unsigned char* src, float* dst, const int size,
         const float alpha, const float beta, const float bias)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < size){
            dst[index] = (float)(src[index] - alpha) / beta + bias;
        }
    }

    NormalizationExecutionInfo::NormalizationExecutionInfo(CUDARuntime *runtime,
        std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root) : ExecutionInfo(runtime, tensorsInfo, root)
    {
    }
    
    NormalizationExecutionInfo::~NormalizationExecutionInfo()
    {
    }

    bool NormalizationExecutionInfo::init(Json::Value& root)
    {
        alpha = root["attr"]["alpha"].asFloat();
        beta  = root["attr"]["beta"].asFloat();
        bias  = root["attr"]["bias"].asFloat();

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

    void NormalizationExecutionInfo::run()
    {

        auto runtime = getCudaRuntime();
        auto stream = runtime->stream();
        beforeRun();
        NormalizationExecutionKernel<<<gridSize, blockSize, 0, stream>>>(srcTensor->device<unsigned char>(),
                dstTensor->device<float>(), totalElementSize, alpha, beta, bias);
        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch normalization kernel fail: %s\n", cudaGetErrorString(cudastatus));
        // {
        //     printBuffer<float>(dstTensor, 0, 10);
        // }
        afterRun();
        return;
    }
}