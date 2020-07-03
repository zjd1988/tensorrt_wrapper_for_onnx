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
        void copyToDebugBuffer(Buffer* srcBuffer);
        Buffer* getDebugBuffer();
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