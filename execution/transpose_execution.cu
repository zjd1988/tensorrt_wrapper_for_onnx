#include "transpose_execution.hpp"

namespace tensorrtInference
{

    template <typename T>
    __global__ void TransposeNHWCToNCHWKernel(const int size, const int channel, const int stride, const T* input, T* output)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < size)
        {
            int hw_index = (index / channel) % stride;
            int batch_index = index / (channel * stride);
            int stride_num = index % 3;
            int new_pos = batch_index * channel * stride + stride * stride_num + hw_index;
            output[new_pos] = input[index];
        }
        return;
    }
    template <typename T>
    __global__ void TransposeExecutionCommonKernel(const int size, const T* input, const int* input_shape, const int* input_axis,
                              const int shape_size, T* output) {
      int pos_size;
      int temp_pos;
      int newpos;
      int newpos_size;
      int pos_array[TRANSPOSE_MAX_DIMENSION];
    
      // for example 4-D: pos = posArray[0] * input_shape[1] * input_shape[2] * input_shape[3] +
      //                        posArray[1] * input_shape[2] * input_shape[3] +
      //                        posArray[2] * input_shape[3] +
      //                        posArray[3]
      for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
        temp_pos = pos;
        pos_size = size / input_shape[0];
        pos_array[0] = temp_pos / pos_size;
        for (int i = 1; i < shape_size; i++) {
          temp_pos -= pos_array[i - 1] * pos_size;
          pos_size = pos_size / input_shape[i];
          pos_array[i] = temp_pos / pos_size;
        }
    
        newpos = pos_array[input_axis[shape_size - 1]];
        newpos_size = 1;
        for (int j = shape_size - 2; j >= 0; j--) {
          newpos_size *= input_shape[input_axis[j + 1]];
          newpos += pos_array[input_axis[j]] * newpos_size;
        }
    
        output[newpos] = input[pos];
      }
      return;
    }

    void callTransposeExecutionKernel(Buffer* src, Buffer* dst, std::string &convertType, CUDARuntime *runtime)
    {
        int size = src->getElementCount();
        // int blockSize = 256;
        int blockSize = runtime->threads_num();
        int gridSize = size / blockSize + 1;
        auto stream = runtime->stream();
        auto shape = src->getShape();
        if(convertType.compare("NHWC2NCHW") == 0) {
            auto dataType = src->getDataType();
            CHECK_ASSERT(shape.size() == 4, "NHWC_TO_NCHW need input buffer dims(%d) != 4!\n", shape.size());
            int channle = shape[1];
            int stride = shape[2] * shape[3];
            if(dataType == OnnxDataType::FLOAT)
            {
                TransposeNHWCToNCHWKernel<float><<<gridSize, blockSize, 0, stream>>>(size, channle, stride,
                    src->device<float>(), dst->device<float>());
            }
            else if(dataType == OnnxDataType::UINT8)
            {
                TransposeNHWCToNCHWKernel<unsigned char><<<gridSize, blockSize, 0, stream>>>(size, channle, stride,
                    src->device<unsigned char>(), dst->device<unsigned char>());
            }
            else
                CHECK_ASSERT(false, "current only support float/uint8 NHWC2NCHW!\n");

        }
        else if(convertType.compare("NCHW2NHWC") == 0) {

        }
        else
            CHECK_ASSERT(false, "current not support %s!\n", convertType.c_str());

        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "run convert kernel(%s) fail: %s\n", convertType.c_str(),
            cudaGetErrorString(cudastatus));
    }

    TransposeExecution::TransposeExecution(CUDARuntime *runtime, std::string executionType) : Execution(runtime, executionType)
    {
        setExecutionType("Transpose");
        setSubExecutionType(executionType);
    }
    
    TransposeExecution::~TransposeExecution()
    {
    }

    bool TransposeExecution::init(std::vector<Buffer*> inputBuffers)
    {
        CHECK_ASSERT(inputBuffers.size() == 1, "input buffer vector size must be 1\n");
        auto shape = inputBuffers[0]->getShape();
        auto dataType = inputBuffers[0]->getDataType();
        auto outBuffer = new Buffer(shape, dataType);
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

    void TransposeExecution::run(bool sync)
    {
        auto inputBuffers = getInputs();
        auto outputBuffers = getOutputs();
        auto runtime = getCudaRuntime();
        auto subType = getSubExecutionType();
        std::vector<int> shape = inputBuffers[0]->getShape();
        if(needMemCpy)
            runtime->copyToDevice(inputBuffers[0], inputBuffers[0]);

        callTransposeExecutionKernel(inputBuffers[0], outputBuffers[0], subType, runtime);
        if(sync)
            runtime->onWaitFinish();
        return;
    }
}