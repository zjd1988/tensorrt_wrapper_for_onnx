#include "buffer.hpp"


#define MEMORY_ALIGN_DEFAULT 64
namespace tensorrtInference {

    static inline void **alignPointer(void **ptr, size_t alignment) {
        return (void **)((intptr_t)((unsigned char *)ptr + alignment - 1) & -alignment);
    }
    void *bufferMemoryAllocAlign(size_t size, size_t alignment) {
        CHECK_ASSERT(size > 0, "malloc size must larger than 0!\n");

        void **origin = (void **)malloc(size + sizeof(void *) + alignment);
        CHECK_ASSERT(origin != NULL, "malloc host mem fail!\n");
        if (!origin) {
            return NULL;
        }

        void **aligned = alignPointer(origin + 1, alignment);
        aligned[-1]    = origin;
        return aligned;
    }

    void *bufferMemoryCallocAlign(size_t size, size_t alignment) {
        CHECK_ASSERT(size > 0, "calloc size must larger than 0!\n");

        void **origin = (void **)calloc(size + sizeof(void *) + alignment, 1);
        CHECK_ASSERT(origin != NULL, "calloc host mem fail!\n");
        if (!origin) {
            return NULL;
        }
        void **aligned = alignPointer(origin + 1, alignment);
        aligned[-1]    = origin;
        return aligned;

    }

    void bufferMemoryFreeAlign(void *aligned) {
        if (aligned) {
            void *origin = ((void **)aligned)[-1];
            free(origin);
        }
    }

    Buffer::Buffer(std::vector<int> shape, OnnxDataType dataType, bool mallocFlag)
    {
        bufferShape.clear();
        hostPtr = nullptr;
        devicePtr = nullptr;
        this->dataType = dataType;
        bufferShape = shape;
        int size = getSize();
        if(mallocFlag)
        {
            hostPtr = bufferMemoryAllocAlign(size, MEMORY_ALIGN_DEFAULT);
            ownHost = true;            
        }
    }

    Buffer::Buffer(int size, OnnxDataType dataType, bool mallocFlag)
    {
        bufferShape.clear();
        hostPtr = nullptr;
        devicePtr = nullptr;
        this->dataType = dataType;
        bufferShape.push_back(size);
        if(mallocFlag)
        {
            hostPtr = bufferMemoryAllocAlign(getSize(), MEMORY_ALIGN_DEFAULT);
            ownHost = true;            
        }
    }    

    Buffer* Buffer::create(std::vector<int> shape, OnnxDataType dataType, void* userData)
    {
        bool ownData = userData == nullptr;
        auto result = new Buffer(shape, dataType, ownData);
        if (nullptr != userData) {
            result->setHost(userData);
        }
        return result;
    }
    
    Buffer::~Buffer() {
        bufferShape.clear();
        if(ownHost && hostPtr != nullptr)
        {
            bufferMemoryFreeAlign(hostPtr);
            hostPtr = nullptr;
        }
    }

    std::vector<int> Buffer::getShape(){
        return bufferShape;
    }
    OnnxDataType Buffer::getDataType(){
        return dataType;
    }

    int Buffer::getSize(){
        int eleSize = onnxDataTypeEleCount[int(dataType)];
        int count = eleSize;
        if(bufferShape.size() <= 0)
            return 0;
        for(int i = 0; i < bufferShape.size(); i++){
            count *= bufferShape[i];
        }
        return count;
    }

    int Buffer::getElementCount(){
        int count = 1;
        if(bufferShape.size() <= 0)
            return 0;
        for(int i = 0; i < bufferShape.size(); i++){
            count *= bufferShape[i];
        }
        return count;
    }

    void Buffer::setDevice(void* ptr) {
        devicePtr = ptr;
    }
    void Buffer::setHost(void* ptr) {
        hostPtr = ptr;
    }
}