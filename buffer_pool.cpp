#include "buffer_pool.hpp"

namespace tensorrtInference {

void* BufferPool::alloc(int size, bool seperate) {
    if (!seperate) {
        auto iter = mFreeList.lower_bound(size);
        if (iter != mFreeList.end()) {
            auto buffer = iter->second->buffer;
            mFreeList.erase(iter);
            return buffer;
        }
    }
    std::shared_ptr<BufferNode> node(new BufferNode(size));
    mAllBuffer.insert(std::make_pair(node->buffer, node));
    return node->buffer;
}

void BufferPool::recycle(void* buffer, bool release) {
    auto iter = mAllBuffer.find(buffer);
    if (iter == mAllBuffer.end()) {
        CHECK_ASSERT(false, "Error for recycle buffer\n");
        return;
    }
    if (release) {
        mAllBuffer.erase(iter);
        return;
    }
    mFreeList.insert(std::make_pair(iter->second->size, iter->second));
}

void BufferPool::clear() {
    mFreeList.clear();
    mAllBuffer.clear();
}

} // namespace tensorrtInference
