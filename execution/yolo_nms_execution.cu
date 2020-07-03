#include "yolo_nms_execution.hpp"
#include <cub/cub.cuh>
#define NMS_THRESH 0.3
#define IOU_THRESH 0.6
#define MAX_KEEP_NUM 20
#define STRIDE 32

#define DIVUP(m,n) (((m)+(n)-1) / (n))
int const threadsPerBlock = sizeof(unsigned long long) * 8;
namespace tensorrtInference
{

    template <typename scalar_t>
    __device__ inline scalar_t devIoU(scalar_t const * const a, scalar_t const * const b, float max_prob)
    {
        scalar_t left = max(a[0], b[0]), right = min(a[2], b[2]);
        scalar_t top = max(a[1], b[1]), bottom = min(a[3], b[3]);
        scalar_t width = max(right - left, 0.f), height = max(bottom - top, 0.f);
        scalar_t interS = width * height;
        scalar_t Sa = (a[2] - a[0]) * (a[3] - a[1]);
        scalar_t Sb = (b[2] - b[0]) * (b[3] - b[1]);
        // return interS / (Sa + Sb - interS) * flag;
        return (max_prob > NMS_THRESH) ? (interS / (Sa + Sb - interS)) : (scalar_t)1;
    }

    template <typename scalar_t>
    __global__ void nms_kernel(const int n_boxes, const scalar_t nms_overlap_thresh,
                            const scalar_t *dev_boxes, const int *idx, const float* max_prob, int64_t *dev_mask) {
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
                if (devIoU(cur_box, block_boxes + i * 4, max_prob[prob_flag_start + i]) > nms_overlap_thresh) {
                    t |= 1ULL << i;
                }
            }
            const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
            dev_mask[cur_box_idx * col_blocks + col_start] = t;
        }
    }

    __global__ void processMask(int boxsesNum, int threadsPerBlock, unsigned long long* mask, unsigned long long* remv, int* keep_buffer, const int* sort_index)
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
         int *class_idx)
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
            if(temp_max_prob >= NMS_THRESH)
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

    Buffer* YoloNMSExecution::mallocBuffer(int size, OnnxDataType dataType, bool mallocHost, bool mallocDevice, StorageType type)
    {
        auto runtime = getCudaRuntime();
        Buffer* buffer = nullptr;
        if(mallocHost)
            buffer = new Buffer(size, dataType, true);
        else
            buffer = new Buffer(size, dataType);
        CHECK_ASSERT(buffer != nullptr, "new Buffer fail\n");
        if(mallocDevice)
            runtime->onAcquireBuffer(buffer, type);
        return buffer;
    }

    void YoloNMSExecution::callYoloNMSExecutionKernel()
    {
        auto inputBuffers = getInputs();
        auto outputBuffers = getOutputs();
        auto runtime = getCudaRuntime();
        auto sortIdxBuffer = getSortIdx();
        auto idxBuffer = getIdx();
        auto sortProbBuffer = getSortProb();
        auto probBuffer = getProb();
        auto maskBuffer = getMask();
        auto classIdxBuffer = getClassIdx();
        int inputIndex = getInputBoxesIndex();
        auto boxesBuffer = inputBuffers[inputIndex];
        auto classesPorbBuffer = inputBuffers[(inputIndex + 1) % 2];
        auto cubBuffer = getCubBuffer();
        auto boxesOutBuffer = getBoxesOut();
        auto maskRemoveBuffer = getMaskRemove();
        
        int boxesNumber = getBoxesNum();
        int classesNumber = getClassesNum();
        cudaError_t cudastatus = cudaSuccess;    

        int size = boxesNumber;
        int blockSize = threadsPerBlock;
        int gridSize = DIVUP(size, threadsPerBlock);
        auto stream = runtime->stream();
        
        // {
        //     copyToDebugBuffer(classesPorbBuffer);
        //     auto debugBuffer = getDebugBuffer();
        //     // int *debugData = debugBuffer->host<int>();
        //     float *debugData = debugBuffer->host<float>();
        //     int start = 0;
        //     int end = start + 10;
        //     for(int i = start; i < end; i++)
        //     {
        //         // printf("%d \n", debugData[i]);
        //         printf("%e \n", debugData[i]);
        //     }
        // }

        getMaxPorb<<<gridSize, blockSize, 0, stream>>>(size, classesPorbBuffer->device<float>(), classesNumber,
            probBuffer->device<float>(), idxBuffer->device<int>(), classIdxBuffer->device<int>());
        cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch getMaxPorb kernel fail: %s\n",cudaGetErrorString(cudastatus));

        int stridex = getImgWidth() / STRIDE;
        int stridey = getImgHeight() / STRIDE;
        processBoxes<<<gridSize, blockSize, 0, stream>>>(size, boxesBuffer->device<float>(), boxesOutBuffer->device<float>(),
            stridex, stridey);
        cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch processBoxes kernel fail: %s\n",cudaGetErrorString(cudastatus));
        
        const int* valueIn = idxBuffer->device<const int>();
        int* valueOut = sortIdxBuffer->device<int>();
        const float* keyIn = probBuffer->device<const float>();
        float* keyOut = sortProbBuffer->device<float>();
        void *tempBuffer = cubBuffer->device<void>();
        size_t tempBufferSize = getCubBufferSize();
        cudaMemsetAsync(cubBuffer->device<void>(), 0, cubBuffer->getSize(), stream);
        cub::DeviceRadixSort::SortPairsDescending(tempBuffer, tempBufferSize, keyIn, keyOut, valueIn, valueOut, boxesNumber,
            0, sizeof(unsigned int) * 8, stream);       

        dim3 blocks(DIVUP(boxesNumber, threadsPerBlock), DIVUP(boxesNumber, threadsPerBlock));
        dim3 threads(threadsPerBlock);
        float* boxesBufferPtr = boxesOutBuffer->device<float>();
        int64_t* maskBufferPtr = maskBuffer->device<int64_t>();
        cudaMemsetAsync(maskBuffer->device<void>(), 0, maskBuffer->getSize(), stream);
        nms_kernel<float><<<blocks, threads, 0, stream>>>(boxesNumber, IOU_THRESH, boxesBufferPtr, valueOut, keyOut, maskBufferPtr);
        cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "run yolo nms kernel fail: %s\n",cudaGetErrorString(cudastatus));

        // runtime->copyFromDevice(maskBuffer, maskBuffer);
        cudaMemsetAsync(maskRemoveBuffer->device<void>(), 0, maskRemoveBuffer->getSize(), stream);
        auto keepBuffer = outputBuffers[0];
        processMask<<<1, 1, 0, stream>>>(boxesNumber, threadsPerBlock, maskBuffer->device<unsigned long long>(), 
            maskRemoveBuffer->device<unsigned long long>(), keepBuffer->device<int>(), sortIdxBuffer->device<int>());
        
    }

    YoloNMSExecution::YoloNMSExecution(CUDARuntime *runtime, std::string executionType) : Execution(runtime, executionType)
    {
        setExecutionType("YoloNMS");
        setSubExecutionType(executionType);
        classIdx = nullptr;
        sortIdx = nullptr;
        sortProb = nullptr;
        idx = nullptr;
        prob = nullptr;
        mask = nullptr;
        keep = nullptr;
        cubBuffer = nullptr;
        boxesOut = nullptr;
        // detectBoxesNum = nullptr;

        boxesNum = 0;
        classesNum = 0;
        cubBufferSize = 0;
        inputBoxesIndex = 0;
        imgHeight = 512;
        imgWidth = 384;
    }
    
    YoloNMSExecution::~YoloNMSExecution()
    {
        delete idx;
        delete sortIdx;
        delete prob;
        delete sortProb;
        delete mask;
        delete cubBuffer;
        delete maskRemove;
    }

    bool YoloNMSExecution::init(std::vector<Buffer*> inputBuffers)
    {      
        auto runtime = getCudaRuntime();
        CHECK_ASSERT(inputBuffers.size() == 2, "input buffer size must be 2(classes and boxes)\n");
        auto shape1 = inputBuffers[0]->getShape();
        auto shape2 = inputBuffers[1]->getShape();
        CHECK_ASSERT(shape1.size() == shape2.size() && shape1.size() == 2, "classed and boxes must have 2 dimensions!\n");
        auto dataType1 = inputBuffers[0]->getDataType();
        auto dataType2 = inputBuffers[1]->getDataType();
        CHECK_ASSERT(dataType1 == dataType2 && dataType1 == OnnxDataType::FLOAT, "classed and boxes must be float!\n");

        int boxesNumber = shape1[0];
        int classesNumber = shape1[1] == 4 ? shape2[1] : shape1[1];
        setBoxesNum(boxesNumber);
        setClassesNum(classesNumber);
        
        auto idxBuffer = mallocBuffer(boxesNumber, OnnxDataType::INT32, false, true);
        setIdx(idxBuffer);

        auto sortIdxBuffer = mallocBuffer(boxesNumber, OnnxDataType::INT32, false, true);
        setSortIdx(sortIdxBuffer);

        auto probBuffer = mallocBuffer(boxesNumber, OnnxDataType::FLOAT, false, true);
        setProb(probBuffer);

        auto sortProbBuffer = mallocBuffer(boxesNumber, OnnxDataType::FLOAT, false, true);
        setSortProb(sortProbBuffer);

        auto maskBuffer = mallocBuffer(boxesNumber * (DIVUP(boxesNumber, 64)), OnnxDataType::INT64, true, true);
        setMask(maskBuffer);

        auto maskRemoveBuffer = mallocBuffer(DIVUP(boxesNumber, threadsPerBlock), OnnxDataType::INT64, false, true);
        setMaskRemove(maskRemoveBuffer);

        auto keepBuffer = mallocBuffer(boxesNumber + 1, OnnxDataType::INT32, false, true, StorageType::STATIC);
        setKeep(keepBuffer);
        addOutput(keepBuffer);

        int index = classesNumber == shape1[1] ? 1 : 0;
        setInputBoxesIndex(index);
        auto boxesOutBuffer = new Buffer(inputBuffers[index]->getShape(), OnnxDataType::FLOAT);
        CHECK_ASSERT(boxesOutBuffer != nullptr, "new Buffer fail\n");
        runtime->onAcquireBuffer(boxesOutBuffer, StorageType::STATIC);
        setBoxesOut(boxesOutBuffer);
        addOutput(boxesOutBuffer);

        auto classIdxBuffer = mallocBuffer(boxesNumber, OnnxDataType::INT32, false, true, StorageType::STATIC);
        setClassIdx(classIdxBuffer);
        addOutput(classIdxBuffer);

        // Determine temporary device storage requirements
        void     *tempStorage = NULL;
        size_t   tempStorageBytes = 0;
        int* valueIn = idxBuffer->device<int>();
        int* valueOut = sortIdxBuffer->device<int>();
        float* keyIn = probBuffer->device<float>();
        float* keyOut = sortProbBuffer->device<float>();
        cub::DeviceRadixSort::SortPairsDescending(tempStorage, tempStorageBytes,
            keyIn, keyOut, valueIn, valueOut, boxesNumber, 0, sizeof(unsigned int) * 8);
        CHECK_ASSERT(tempStorageBytes > 0, "SortPairs temp mem calculate fail!\n");

        auto cubTempBuffer = mallocBuffer(tempStorageBytes, OnnxDataType::UINT8, false, true);
        setCubBuffer(cubTempBuffer);
        setCubBufferSize(tempStorageBytes);
        
        addInput(inputBuffers[0]);
        addInput(inputBuffers[1]);

        runtime->onReleaseBuffer(idxBuffer, StorageType::DYNAMIC);
        runtime->onReleaseBuffer(sortIdxBuffer, StorageType::DYNAMIC);
        runtime->onReleaseBuffer(probBuffer, StorageType::DYNAMIC);
        runtime->onReleaseBuffer(sortProbBuffer, StorageType::DYNAMIC);
        runtime->onReleaseBuffer(maskBuffer, StorageType::DYNAMIC);
        runtime->onReleaseBuffer(cubTempBuffer, StorageType::DYNAMIC);
        runtime->onReleaseBuffer(maskRemoveBuffer, StorageType::DYNAMIC);
                
        return true;
    }

    void YoloNMSExecution::run(bool sync)
    {
        auto runtime = getCudaRuntime();

        callYoloNMSExecutionKernel();
        if(sync)
            runtime->onWaitFinish();
        return;
    }
}