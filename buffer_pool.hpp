#ifndef __BUFFER_POOL_HPP__
#define __BUFFER_POOL_HPP__

#include <map>
#include <memory>
#include <vector>
#include "utils.hpp"

namespace tensorrtInference {

class BufferPool {
public:
    BufferPool() {
    }
    ~BufferPool() { clear(); }
    void* alloc(int size, bool seperate = false);
    void recycle(void* buffer, bool release = false);
    void clear();

    class BufferNode {
    public:
        int size;
        void* buffer;
        BufferNode(int size) {
            buffer = nullptr;
            this->size = 0;
            buffer = bufferMalloc(size);
            this->size = size;
        }
        ~BufferNode() {
            if(buffer != nullptr) {
                bufferFree(buffer);
                this->size = size;
            }
        }
        void *bufferMalloc(size_t size_in_bytes)
        {
            void *ptr;
            CUDA_CHECK(cudaMalloc(&ptr, size_in_bytes));
            return ptr;
        }

        void bufferFree(void *ptr)
        {
            CUDA_CHECK(cudaFree(ptr));
        }
    };
private:
    std::map<void*, std::shared_ptr<BufferNode>> mAllBuffer;
    std::multimap<int, std::shared_ptr<BufferNode>> mFreeList;
};

} // namespace tensorrtInference

#endif //__BUFFER_POOL_HPP__
