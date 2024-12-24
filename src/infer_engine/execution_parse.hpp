/********************************************
 * Filename: execution_parse.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "json/json.h"
#include "common/utils.hpp"
#include "execution_info/execution_info.hpp"

namespace TENSORRT_WRAPPER
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

} // namespace TENSORRT_WRAPPER
