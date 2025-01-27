#ifndef __BUFFER_HPP__
#define __BUFFER_HPP__

#include <iostream>
#include <vector>
#include "utils.hpp"
using namespace std;

namespace tensorrtInference {

    /** backend buffer storage type */
    enum StorageType {
        /**
         use NOT reusable memory.
            - allocates memory when `onAcquireBuffer` is called.
            - releases memory when `onReleaseBuffer` is called or when the backend is deleted.
            - do NOTHING when `onClearBuffer` is called.
            */
        STATIC,
        /**
         use reusable memory.
            - allocates or reuses memory when `onAcquireBuffer` is called. prefers reusing.
            - collects memory for reuse when `onReleaseBuffer` is called.
            - releases memory when `onClearBuffer` is called or when the backend is deleted.
            */
        DYNAMIC,
        /**
         * */
        UNDEFINED_STORAGE_TYPE,
    };
    class Buffer
    {
    public:
        Buffer(std::vector<int> shape, OnnxDataType dataType, bool mallocFlag = false);
        Buffer(int size, OnnxDataType dataType, bool mallocFlag = false);
        ~Buffer();
        Buffer(const Buffer&)  = delete;
        Buffer(const Buffer&&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer& operator=(const Buffer&&) = delete;
        static Buffer* create(std::vector<int> shape, OnnxDataType dataType, void* userData);
        std::vector<int> getShape();
        OnnxDataType getDataType();
        int getSize();
        int getElementCount();
        void setDevice(void* ptr);
        void setHost(void* ptr);
        template <typename T>
        T* host() { return (T*)hostPtr;}
        template <typename T>
        T* device() { return (T*)devicePtr;}
        void setStorageType(StorageType type) {storageType = type;}
        StorageType getStorageType() { return storageType; }

    private:
        OnnxDataType dataType;
        void* hostPtr;
        void* devicePtr;
        std::vector<int> bufferShape;
        bool ownHost = false;
        StorageType storageType;
    };
} // namespace tensorrtInference
#endif