/********************************************
 * Filename: buffer_pool.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <map>
#include <memory>
#include <vector>
#include "common/utils.hpp"

namespace TENSORRT_WRAPPER
{

    class BufferPool
    {
    public:
        BufferPool() {}
        ~BufferPool() { clear(); }
        void* alloc(int size, bool seperate = false);
        void recycle(void* buffer, bool release = false);
        void clear();

        class BufferNode
        {
        public:
            int size;
            void* buffer;
            BufferNode(int size)
            {
                buffer = nullptr;
                this->size = 0;
                buffer = bufferMalloc(size);
                this->size = size;
            }

            ~BufferNode()
            {
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

} // namespace TENSORRT_WRAPPER
