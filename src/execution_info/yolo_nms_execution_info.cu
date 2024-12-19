#include "yolo_nms_execution_info.hpp"
#include <cub/cub.cuh>
#define NMS_THRESH 0.3
#define IOU_THRESH 0.6
#define MAX_KEEP_NUM 20
#define STRIDE 32

#define DIVUP(m,n) (((m)+(n)-1) / (n))
static int const threadsPerBlock = sizeof(unsigned long long) * 8;
namespace tensorrtInference
{

    template <typename scalar_t>
    __device__ inline scalar_t devIoU(scalar_t const * const a, scalar_t const * const b, float max_prob, const float conf_thresh)
    {
        scalar_t left = max(a[0], b[0]), right = min(a[2], b[2]);
        scalar_t top = max(a[1], b[1]), bottom = min(a[3], b[3]);
        scalar_t width = max(right - left, 0.f), height = max(bottom - top, 0.f);
        scalar_t interS = width * height;
        scalar_t Sa = (a[2] - a[0]) * (a[3] - a[1]);
        scalar_t Sb = (b[2] - b[0]) * (b[3] - b[1]);
        return (max_prob > conf_thresh) ? (interS / (Sa + Sb - interS)) : (scalar_t)1;
    }

    template <typename scalar_t>
    __global__ void nms_kernel(const int n_boxes, const scalar_t nms_overlap_thresh,
                            const scalar_t *dev_boxes, const int *idx, 
                            const float* max_prob, int64_t *dev_mask, const float conf_thresh) {
        const int row_start = blockIdx.y;
        const int col_start = blockIdx.x;

        const int row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
        const int col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

        __shared__ scalar_t block_boxes[threadsPerBlock * 4];
        if (threadIdx.x < col_size) {
            block_boxes[threadIdx.x * 4 + 0] =
                dev_boxes[idx[(threadsPerBlock * col_start + threadIdx.x)] * 4 + 0];
            block_boxes[threadIdx.x * 4 + 1] =
                dev_boxes[idx[(threadsPerBlock * col_start + threadIdx.x)] * 4 + 1];
            block_boxes[threadIdx.x * 4 + 2] =
                dev_boxes[idx[(threadsPerBlock * col_start + threadIdx.x)] * 4 + 2];
            block_boxes[threadIdx.x * 4 + 3] =
                dev_boxes[idx[(threadsPerBlock * col_start + threadIdx.x)] * 4 + 3];
        }
        __syncthreads();

        if (threadIdx.x < row_size) {
            const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
            const scalar_t *cur_box = dev_boxes + idx[cur_box_idx] * 4;
            const int prob_flag_start = threadsPerBlock * col_start;
            int i = 0;
            unsigned long long t = 0;
            int start = 0;
            if (row_start == col_start) {
                start = threadIdx.x + 1;
            }
            for (i = start; i < col_size; i++) {
                if (devIoU(cur_box, block_boxes + i * 4, max_prob[prob_flag_start + i], conf_thresh) > nms_overlap_thresh) {
                    t |= 1ULL << i;
                }
            }
            const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
            dev_mask[cur_box_idx * col_blocks + col_start] = t;
        }
    }

    __global__ void processMask(int boxsesNum, int threadsPerBlock, unsigned long long* mask, unsigned long long* remv, 
        int* keep_buffer, const int* sort_index)
    {
        int col_blocks = DIVUP(boxsesNum, threadsPerBlock);
        // std::vector<unsigned long long> remv(col_blocks);
        // memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);
        int num_to_keep = 0;
        int *keep_out = keep_buffer + 1;
        for (int i = 0; i < boxsesNum; i++)
        {
            int nblock = i / threadsPerBlock;
            int inblock = i % threadsPerBlock;
        
            if (!(remv[nblock] & (1ULL << inblock)))
            {
                keep_out[num_to_keep++] = sort_index[i];
                // printf("keep box index %d\n", sort_index[i]);
                unsigned long long* p = mask + i * col_blocks;
                for (int j = nblock; j < col_blocks; j++)
                {
                    remv[j] |= p[j];
                }
            }
        }
        keep_buffer[0] = num_to_keep;
        return;
    }

    __global__ void getMaxPorb(const int size, const float* class_prob, const int class_num, float* max_prob, int* idx,
         int *class_idx, const int conf_thresh)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < size)
        {
            // printf("run here %d!\n", index);
            float temp_max_prob = 0.0f;
            const float *start = class_prob + index * class_num;
            int class_index = -1;
            for(int i = 0; i < class_num; i++)
            {
                float curr_prob = start[i];
                if(temp_max_prob <= curr_prob)
                {
                    class_index = i;
                    temp_max_prob = curr_prob;
                }
            }
            max_prob[index] = 0.0f;
            if(temp_max_prob >= conf_thresh)
            {
                // atomicAdd(detecNum, 1);
                max_prob[index] = temp_max_prob;
                // printf("run here %d!\n", index);
            }
            idx[index] = index;
            class_idx[index] = class_index;
        }
    }

    __global__ void processBoxes(int size, const float* src, float* dst,const int stridex, const int stridey)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < size)
        {
            float4* src_boxes = (float4*)src + index;
            float4* dst_boxes = (float4*)dst + index;
            float4 boxes = *src_boxes;
            float4 new_boxes = {0};
            new_boxes.x = boxes.x - boxes.z * stridex / 2;
            new_boxes.y = boxes.y - boxes.w * stridey / 2;
            new_boxes.z = boxes.x + boxes.z * stridex / 2;
            new_boxes.w = boxes.y + boxes.w * stridey / 2;
            *dst_boxes = new_boxes;
        }
    }

    void YoloNMSExecutionInfo::callYoloNMSExecutionKernel()
    {
        auto srcTensorNames = getInputTensorNames();
        auto dstTensorNames = getOutputTensorNames();
        auto allBuffers = getTensorsInfo();
        auto classesPorbBuffer = allBuffers[srcTensorNames[0]];
        auto boxesBuffer = allBuffers[srcTensorNames[1]];
        auto keepBuffer = allBuffers[dstTensorNames[0]];
        auto boxesOutBuffer = allBuffers[dstTensorNames[1]];
        auto classIdxBuffer = allBuffers[dstTensorNames[2]];
        
        int boxesNumber = boxesNum;
        int classesNumber = classesNum;
        cudaError_t cudastatus = cudaSuccess;
        int size = boxesNumber;
        int blockSize = threadsPerBlock;
        int gridSize = DIVUP(size, threadsPerBlock);
        auto runtime = getCudaRuntime();
        auto stream = runtime->stream();
        
        // {
        //     printBuffer<float>(boxesBuffer, 0, 10);
        //     printBuffer<float>(classesPorbBuffer, 0, 10);
        // }

        getMaxPorb<<<gridSize, blockSize, 0, stream>>>(size, classesPorbBuffer->device<float>(), classesNumber,
            probBuffer->device<float>(), idxBuffer->device<int>(), classIdxBuffer->device<int>(), confThresh);
        cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch getMaxPorb kernel fail: %s\n",cudaGetErrorString(cudastatus));

        // {
        //     printBuffer<float>(probBuffer.get(), 300, 310);
        //     printBuffer<int>(classIdxBuffer, 300, 310);
        // }
        
        int stridex = imgWidth / STRIDE;
        int stridey = imgHeight / STRIDE;
        processBoxes<<<gridSize, blockSize, 0, stream>>>(size, boxesBuffer->device<float>(), boxesOutBuffer->device<float>(),
            stridex, stridey);
        cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch processBoxes kernel fail: %s\n",cudaGetErrorString(cudastatus));
        
        const int* valueIn = idxBuffer->device<const int>();
        int* valueOut = sortIdxBuffer->device<int>();
        const float* keyIn = probBuffer->device<const float>();
        float* keyOut = sortProbBuffer->device<float>();
        void *tempBuffer = cubBuffer->device<void>();
        size_t tempBufferSize = cubBufferSize;
        cudaMemsetAsync(cubBuffer->device<void>(), 0, cubBuffer->getSize(), stream);
        cub::DeviceRadixSort::SortPairsDescending(tempBuffer, tempBufferSize, keyIn, keyOut, valueIn, valueOut, boxesNumber,
            0, sizeof(unsigned int) * 8, stream);
          
        // {
        //     printBuffer<float>(sortProbBuffer.get(), 0, 10);
        //     printBuffer<int>(sortIdxBuffer.get(), 0, 10);
        // }

        dim3 blocks(DIVUP(boxesNumber, threadsPerBlock), DIVUP(boxesNumber, threadsPerBlock));
        dim3 threads(threadsPerBlock);
        float* boxesBufferPtr = boxesOutBuffer->device<float>();
        int64_t* maskBufferPtr = maskBuffer->device<int64_t>();
        cudaMemsetAsync(maskBuffer->device<void>(), 0, maskBuffer->getSize(), stream);
        nms_kernel<float><<<blocks, threads, 0, stream>>>(boxesNumber, iouThresh, boxesBufferPtr, valueOut, keyOut, maskBufferPtr, confThresh);
        cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "run yolo nms kernel fail: %s\n",cudaGetErrorString(cudastatus));

        cudaMemsetAsync(maskRemoveBuffer->device<void>(), 0, maskRemoveBuffer->getSize(), stream);
        processMask<<<1, 1, 0, stream>>>(boxesNumber, threadsPerBlock, maskBuffer->device<unsigned long long>(), 
            maskRemoveBuffer->device<unsigned long long>(), keepBuffer->device<int>(), sortIdxBuffer->device<int>());
        
        // {
        //     printBuffer<int>(keepBuffer, 0, 5);
        // }
    }

    YoloNMSExecutionInfo::YoloNMSExecutionInfo(CUDARuntime *runtime,
        std::map<std::string, std::shared_ptr<Buffer>> &tensorsInfo, Json::Value& root) : ExecutionInfo(runtime, tensorsInfo, root)
    {
        sortIdxBuffer.reset();
        sortProbBuffer.reset();
        idxBuffer.reset();
        probBuffer.reset();
        maskBuffer.reset();
        maskRemoveBuffer.reset();
        cubBuffer.reset();

        boxesNum = 0;
        classesNum = 0;
        cubBufferSize = 0;
        imgHeight = 512;
        imgWidth = 384;
        confThresh = 0.3;
        iouThresh = 0.6;
    }
    
    YoloNMSExecutionInfo::~YoloNMSExecutionInfo()
    {
    }


    void YoloNMSExecutionInfo::recycleBuffers()
    {
        auto runtime = getCudaRuntime();
        ExecutionInfo::recycleBuffers();
        runtime->onReleaseBuffer(idxBuffer.get(), StorageType::DYNAMIC);
        runtime->onReleaseBuffer(sortIdxBuffer.get(), StorageType::DYNAMIC);
        runtime->onReleaseBuffer(probBuffer.get(), StorageType::DYNAMIC);
        runtime->onReleaseBuffer(sortProbBuffer.get(), StorageType::DYNAMIC);
        runtime->onReleaseBuffer(maskBuffer.get(), StorageType::DYNAMIC);
        runtime->onReleaseBuffer(cubBuffer.get(), StorageType::DYNAMIC);
        runtime->onReleaseBuffer(maskRemoveBuffer.get(), StorageType::DYNAMIC);
    }

    bool YoloNMSExecutionInfo::init(Json::Value& root)
    {
        imgHeight = root["attr"]["img_height"].asInt();
        imgWidth = root["attr"]["img_width"].asInt();
        confThresh = root["attr"]["conf_thresh"].asFloat();
        iouThresh = root["attr"]["iou_thresh"].asFloat();
        auto srcTensorNames = getInputTensorNames();
        auto dstTensorNames = getOutputTensorNames();
        auto inputBuffers = getTensorsInfo();
        auto runtime = getCudaRuntime();
        CHECK_ASSERT(srcTensorNames.size() == 2, "input buffer size must be 2(classes and boxes)\n");
        auto shape1 = inputBuffers[srcTensorNames[0]]->getShape();
        auto shape2 = inputBuffers[srcTensorNames[1]]->getShape();
        CHECK_ASSERT(shape1.size() == shape2.size() && shape1.size() == 2, "classes and boxes must have 2 dimensions!\n");
        auto dataType1 = inputBuffers[srcTensorNames[0]]->getDataType();
        auto dataType2 = inputBuffers[srcTensorNames[1]]->getDataType();
        CHECK_ASSERT(dataType1 == dataType2 && dataType1 == OnnxDataType::FLOAT, "classes and boxes must be float!\n");

        boxesNum = shape1[0];
        classesNum = shape1[1];

        idxBuffer.reset(mallocBuffer(boxesNum, OnnxDataType::INT32, false, true));
        sortIdxBuffer.reset(mallocBuffer(boxesNum, OnnxDataType::INT32, false, true));
        probBuffer.reset(mallocBuffer(boxesNum, OnnxDataType::FLOAT, false, true));
        sortProbBuffer.reset(mallocBuffer(boxesNum, OnnxDataType::FLOAT, false, true));
        maskBuffer.reset(mallocBuffer(boxesNum * (DIVUP(boxesNum, 64)), OnnxDataType::INT64, true, true));
        maskRemoveBuffer.reset(mallocBuffer(DIVUP(boxesNum, threadsPerBlock), OnnxDataType::INT64, false, true));

        // Determine temporary device storage requirements
        void     *tempStorage = NULL;
        size_t   tempStorageBytes = 0;
        int* valueIn = idxBuffer->device<int>();
        int* valueOut = sortIdxBuffer->device<int>();
        float* keyIn = probBuffer->device<float>();
        float* keyOut = sortProbBuffer->device<float>();
        cub::DeviceRadixSort::SortPairsDescending(tempStorage, tempStorageBytes,
            keyIn, keyOut, valueIn, valueOut, boxesNum, 0, sizeof(unsigned int) * 8);
        CHECK_ASSERT(tempStorageBytes > 0, "SortPairs temp mem calculate fail!\n");

        cubBuffer.reset(mallocBuffer(tempStorageBytes, OnnxDataType::UINT8, false, true));
        cubBufferSize = tempStorageBytes;

        recycleBuffers();
        return true;
    }

    void YoloNMSExecutionInfo::run()
    {
        beforeRun();
        callYoloNMSExecutionKernel();
        afterRun();
        return;
    }
}