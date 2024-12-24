#include "transpose_execution_info.hpp"

namespace TENSORRT_WRAPPER
{

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

    TransposeExecutionInfo::TransposeExecutionInfo(CUDARuntime *runtime,
        std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root) : ExecutionInfo(runtime, tensorsInfo, root)
    {
        blockSize = 0;
        gridSize = 0;
        totalElementSize = 0;
        shapeSize = 0;
        srcTensor = nullptr;
        dstTensor = nullptr;
        inputShape.reset();
        inputAxis.reset();
    }
    
    TransposeExecutionInfo::~TransposeExecutionInfo()
    {
    }

    bool TransposeExecutionInfo::init(Json::Value& root)
    {
        int permSize = root["attr"]["perm"].size();
        std::vector<int> perm;
        for(int i = 0; i < permSize; i++)
        {
            int dim = root["attr"]["perm"][i].asInt();
            perm.push_back(dim);
        }
        shapeSize = perm.size();
        
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

        auto shape = srcTensor->getShape();
        CHECK_ASSERT(perm.size() == shape.size(), "input tensor shape size should equal to perm size!\n");
        std::shared_ptr<Buffer> shapeBuffer(mallocBuffer(shape.size(), OnnxDataType::INT32, true, false));
        inputShape.reset(mallocBuffer(shape.size(), OnnxDataType::INT32, false, true, StorageType::STATIC));
        memcpy(shapeBuffer->host<void>(), &shape[0], sizeof(int) * shape.size());
        runtime->copyToDevice(shapeBuffer.get(), inputShape.get());

        std::shared_ptr<Buffer> axisBuffer(mallocBuffer(perm.size(), OnnxDataType::INT32, true, false));
        inputAxis.reset(mallocBuffer(perm.size(), OnnxDataType::INT32, false, true, StorageType::STATIC));
        memcpy(axisBuffer->host<void>(), &perm[0], sizeof(int) * perm.size());
        runtime->copyToDevice(axisBuffer.get(), inputAxis.get());
        recycleBuffers();
        return true;
    }

    void TransposeExecutionInfo::run()
    {
        auto runtime = getCudaRuntime();
        auto stream = runtime->stream();
        beforeRun();

        if(srcTensor->getDataType() == OnnxDataType::FLOAT)
        {
            TransposeExecutionCommonKernel<<<gridSize, blockSize, 0, stream>>>(totalElementSize, srcTensor->device<float>(), 
                inputShape->device<int>(), inputAxis->device<int>(), shapeSize, dstTensor->device<float>());
            
        }
        else if(srcTensor->getDataType() == OnnxDataType::UINT8)
        {
            TransposeExecutionCommonKernel<<<gridSize, blockSize, 0, stream>>>(totalElementSize, srcTensor->device<unsigned char>(), 
                inputShape->device<int>(), inputAxis->device<int>(), shapeSize, dstTensor->device<unsigned char>());
        }
        else
            CHECK_ASSERT(false, "only support float/uint8!\n");
        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch transpose kernel fail: %s\n", cudaGetErrorString(cudastatus));
        // {
        //     printBuffer<unsigned char>(dstTensor, 0, 10);
        // }
        afterRun();
        return;
    }
}