#ifndef __EXCUTION_HPP__
#define __EXCUTION_HPP__
#include <vector>
#include "cuda_runtime.hpp"
#include "buffer.hpp"
#include "utils.hpp"


namespace tensorrtInference {

    class Execution {
    public:
        Execution(CUDARuntime *runtime, std::string subType);
        virtual ~Execution();
        virtual bool init(std::vector<Buffer*> inputBuffers) = 0;
        virtual void run(bool sync = false) = 0;
        void printExecutionInfo();
        std::string getExecutionType();
        std::string getSubExecutionType();
        std::vector<Buffer*> getInputs();
        std::vector<Buffer*> getOutputs();
        CUDARuntime* getCudaRuntime();
        void addInput(Buffer* buffer);
        void addOutput(Buffer* buffer);
        void setExecutionType(std::string type);
        void setSubExecutionType(std::string subType);
        template<typename T>
        void printBuffer(Buffer* buffer, int start, int end)
        {
            CHECK_ASSERT(buffer != nullptr, "buffer must not be none!\n");
            CHECK_ASSERT(start >= 0, "start index must greater than 1!\n");
            auto runtime = getCudaRuntime();
            runtime->copyFromDevice(buffer, debugBuffer);
            auto debugData = debugBuffer->host<T>();
            int count = debugBuffer->getElementCount();
            int printStart = (start > count) ? 0 : start;
            int printEnd   = ((end - start) > count) ? (start + count) : end;
            std::cout << "buffer data is :" << std::endl;
            if(onnxDataTypeEleCount[buffer->getDataType()] != 1)
            {
                for(int i = printStart; i < printEnd; i++)
                {
                    std::cout << debugData[i] << " " << std::endl;
                }
            }
            else
            {
                for(int i = printStart; i < printEnd; i++)
                {
                    printf("%d \n", debugData[i]);
                }
            }
        }
        void recycleBuffers();
        Buffer* mallocBuffer(int size, OnnxDataType dataType, bool mallocHost, bool mallocDevice, 
            StorageType type = StorageType::DYNAMIC);
        Buffer* mallocBuffer(std::vector<int> shape, OnnxDataType dataType, bool mallocHost, 
            bool mallocDevice, StorageType type = StorageType::DYNAMIC);
    private:
        CUDARuntime* cudaRuntime;
        std::string executionType;
        std::string subExecutionType;
        std::vector<Buffer*> inputs;
        std::vector<Buffer*> outputs;
        Buffer* debugBuffer;
    };

    typedef Execution* (*constructExecutionFunc)(CUDARuntime *runtime, std::string subType);

    class ConstructExecution
    {
    private:
        static ConstructExecution* instance;
        void registerConstructExecutionFunc();
        std::map<std::string, constructExecutionFunc> constructExecutionFuncMap;
        ConstructExecution()
        {
        }
    public:
        constructExecutionFunc getConstructExecutionFunc(std::string executionType);
        static ConstructExecution* getInstance() {
            return instance;
        }
    };
    
    extern constructExecutionFunc getConstructExecutionFuncMap(std::string executionType);

}


#endif