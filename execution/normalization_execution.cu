#include "normalization_execution.hpp"

namespace tensorrtInference
{
    __global__ void NormalizationExecutionKernel(unsigned char* src, float* dst, const int size,
         const float alpha, const float beta, const float bias)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < size){
            dst[index] = (float)(src[index] - alpha) / beta + bias;
        }
    }

    NormalizationExecution::NormalizationExecution(CUDARuntime *runtime, std::string executionType) : Execution(runtime, executionType)
    {
        setExecutionType("Normalization");
        setSubExecutionType(executionType);
    }
    
    NormalizationExecution::~NormalizationExecution()
    {
    }

    bool NormalizationExecution::init(std::vector<Buffer*> inputBuffers)
    {
        CHECK_ASSERT(inputBuffers.size() == 1, "input buffer vector size must be 1\n");
        auto shape = inputBuffers[0]->getShape();
        auto outBuffer = new Buffer(shape, OnnxDataType::FLOAT);
        CHECK_ASSERT(outBuffer != nullptr, "new Buffer fail\n");
        addOutput(outBuffer);
        addInput(inputBuffers[0]);
        auto runtime = getCudaRuntime();
        if(inputBuffers[0]->device<void>() == nullptr)
        {
            runtime->onAcquireBuffer(inputBuffers[0], StorageType::STATIC);
            needMemCpy = true;
        }
        runtime->onAcquireBuffer(outBuffer, StorageType::DYNAMIC);
        recycleBuffers();
        return true;
    }

    void NormalizationExecution::run(bool sync)
    {
        auto inputBuffers = getInputs();
        auto outputBuffers = getOutputs();
        auto runtime = getCudaRuntime();
        auto subType = getSubExecutionType();
        int size = inputBuffers[0]->getSize();
        if(needMemCpy)
            runtime->copyToDevice(inputBuffers[0], inputBuffers[0]);

        // int blockSize = 256;
        int blockSize = runtime->threads_num();
        int gridSize = size / blockSize + 1;
        auto stream = runtime->stream();
        float alpha = 0.0f;
        float beta  = 1.0f;
        float bias = 0.0f;
        if(subType.compare("Scale_0_1") == 0) {
            beta = 255.0f;
        }
        else if(subType.compare("Scale_n1_1") == 0){
            beta = 127.0f;
            bias = -1.0f;
        }
        else
            CHECK_ASSERT(false, "current only support scale to [0,1] or [-1, 1]!\n");

        NormalizationExecutionKernel<<<gridSize, blockSize, 0, stream>>>(inputBuffers[0]->device<unsigned char>(),
                outputBuffers[0]->device<float>(), size, alpha, beta, bias);

        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "run convert kernel(%s) fail: %s\n", getSubExecutionType().c_str(),
            cudaGetErrorString(cudastatus));
        // {
        //     printBuffer<unsigned char>(inputBuffers[0], 0, 10);
        //     cudastatus = cudaGetLastError();
        //     CHECK_ASSERT(cudastatus == cudaSuccess, "launch debug print kernel fail: %s\n",cudaGetErrorString(cudastatus));            
        // }
        if(sync)
            runtime->onWaitFinish();
        return;
    }
}