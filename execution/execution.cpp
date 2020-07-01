#include <iostream>
#include <map>
#include "execution.hpp"
#include "utils.hpp"
#include "convert_execution.hpp"
#include "datamovement_execution.hpp"
using namespace std;

#define CONSTUCT_EXECUTION_FUNC_DEF(type)                                          \
Execution* construct##type##Execution(CUDARuntime *runtime, std::string subType)   \
{                                                                                  \
    Execution* execution = new type##Execution(runtime, subType);                  \
    return execution;                                                              \
}

#define CONSTUCT_EXECUTION_FUNC(type) construct##type##Execution

namespace tensorrtInference {

    Execution::Execution(CUDARuntime *runtime, std::string executionType)
    {
        inputs.clear();
        outputs.clear();
        cudaRuntime = runtime;
    }
    Execution::~Execution()
    {
        for(int i = 0; i < outputs.size(); i++){
            delete outputs[i];
            outputs[i] = nullptr;
        }
        inputs.clear();
        outputs.clear();
        cudaRuntime = nullptr;
        executionType = "";
    }    
    std::vector<Buffer*> Execution::getInputs() {return inputs;}
    std::vector<Buffer*> Execution::getOutputs() {return outputs;}
    std::string Execution::getExecutionType() {return executionType;}
    std::string Execution::getSubExecutionType() {return subExecutionType;}
    CUDARuntime* Execution::getCudaRuntime() {return cudaRuntime;}
    void Execution::setExecutionType(std::string type) {executionType = type;}
    void Execution::setSubExecutionType(std::string subType) {subExecutionType = subType;}
    void Execution::addInput(Buffer* buffer) {inputs.push_back(buffer);}
    void Execution::addOutput(Buffer* buffer) {outputs.push_back(buffer);}
    void Execution::printExecutionInfo() {
        LOG("################### Execution INFO ######################\n");
        LOG("currend execution type is %s\n", executionType.c_str());
        auto input = getInputs();
        LOG("Input tensor size is %d\n", input.size());
        auto output = getOutputs();
        LOG("Output tensor size is %d\n", output.size());
    }    

    CONSTUCT_EXECUTION_FUNC_DEF(Convert)
    CONSTUCT_EXECUTION_FUNC_DEF(DataMovement)

    constructExecutionFunc ConstructExecution::getConstructExecutionFunc(std::string executionType)
    {
        if(constructExecutionFuncMap.size() == 0)
            registerConstructExecutionFunc();
        if(constructExecutionFuncMap.count(executionType) != 0)
            return constructExecutionFuncMap[executionType];
        else
            return (constructExecutionFunc)nullptr;
    }

    void ConstructExecution::registerConstructExecutionFunc()
    {
        constructExecutionFuncMap["ConvertToFloat"]            = CONSTUCT_EXECUTION_FUNC(Convert);
        constructExecutionFuncMap["CopyFromDevice"]            = CONSTUCT_EXECUTION_FUNC(DataMovement);
    }

    ConstructExecution* ConstructExecution::instance = new ConstructExecution;

    constructExecutionFunc getConstructExecutionFuncMap(std::string executionType)
    {
        auto instance = ConstructExecution::getInstance();
        if(instance != nullptr)
            return instance->getConstructExecutionFunc(executionType);
        else
            return (constructExecutionFunc)nullptr;
    }
}