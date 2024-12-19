#ifndef __EXECUTION_PARSE_HPP__
#define __EXECUTION_PARSE_HPP__
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "utils.hpp"
#include "json/json.h"
#include "execution_info.hpp"


namespace tensorrtInference
{
    class executionParse {
    public:
        executionParse(CUDARuntime *cudaRuntime, std::string &jsonFile);
        ~executionParse();
        const std::vector<std::string>& getTopoNodeOrder();
        const std::map<std::string, std::shared_ptr<Buffer>>& getTensorsInfo();
        const std::map<std::string, std::shared_ptr<ExecutionInfo>>& getExecutionInfoMap();
        bool getInitFlag() {return initFlag;}
        void runInference();
        std::map<std::string, void*> getInferenceResult();
    private:
        bool extractExecutionInfo(Json::Value &root);
        CUDARuntime* getCudaRuntime() {return cudaRuntime;}
        std::vector<std::string> topoExecutionInfoOrder;
        std::map<std::string, std::shared_ptr<ExecutionInfo>> executionInfoMap;
        std::map<std::string, std::shared_ptr<Buffer>> tensorsInfo;
        std::vector<std::string> inputTensorNames;
        std::vector<std::string> outputTensorNames;
        CUDARuntime *cudaRuntime;
        bool initFlag = false;
    };
} //tensorrtInference

#endif //__EXECUTION_PARSE_HPP__