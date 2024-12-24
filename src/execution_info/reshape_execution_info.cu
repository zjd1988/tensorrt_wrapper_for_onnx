#include "reshape_execution_info.hpp"

namespace TENSORRT_WRAPPER
{

    ReshapeExecutionInfo::ReshapeExecutionInfo(CUDARuntime *runtime,
        std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root) : ExecutionInfo(runtime, tensorsInfo, root)
    {
        newShape.clear();
        totalElementSize = 0;
        srcTensor = nullptr;
        dstTensor = nullptr;
    }
    
    ReshapeExecutionInfo::~ReshapeExecutionInfo()
    {
    }

    bool ReshapeExecutionInfo::init(Json::Value& root)
    {
        int size = root["attr"]["shape"].size();
        int count = 1;
        for(int i = 0; i < size; i++)
        {
            int dim = root["attr"]["shape"][i].asInt();
            newShape.push_back(dim);
            count *= dim;
        }

        auto runtime = getCudaRuntime();
        auto srcTensorNames = getInputTensorNames();
        auto dstTensorNames = getOutputTensorNames();
        CHECK_ASSERT(srcTensorNames.size() == dstTensorNames.size(), "input tensor size should be equal to output!\n");
        CHECK_ASSERT(srcTensorNames.size() == 1, "input tensor size should be equal to 1!\n");
        auto tensorsInfo = getTensorsInfo();
        srcTensor = tensorsInfo[srcTensorNames[0]];
        dstTensor = tensorsInfo[dstTensorNames[0]];
        totalElementSize = srcTensor->getElementCount();
        CHECK_ASSERT(count == totalElementSize, "src tensor elemet count should equal to dst tensor!\n");
        recycleBuffers();
        return true;
    }

    void ReshapeExecutionInfo::run()
    {
        auto runtime = getCudaRuntime();
        auto stream = runtime->stream();
        beforeRun();
        runtime->copyFromDeviceToDevice(srcTensor, dstTensor);
        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch reshape kernel fail: %s\n", cudaGetErrorString(cudastatus));
        // {
        //     printBuffer<unsigned char>(dstTensor, 0, 10);
        // }
        afterRun();
        return;
    }
}