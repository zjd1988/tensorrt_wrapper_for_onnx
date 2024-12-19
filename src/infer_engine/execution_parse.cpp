#include "execution_parse.hpp"
using namespace std;

namespace tensorrtInference
{
    executionParse::executionParse(CUDARuntime *runtime, std::string &jsonFile)
    {
        CHECK_ASSERT(runtime != nullptr, "cuda runtime is null!\n");
        cudaRuntime = runtime;
        ifstream jsonStream;
        jsonStream.open(jsonFile);
        if(!jsonStream.is_open())
        {
            std::cout << "open json file " << jsonFile << " fail!!!" << std::endl;
            return;
        }
        Json::Reader reader;
        Json::Value root;
        if (!reader.parse(jsonStream, root, false))
        {
            std::cout << "parse json file " << jsonFile << " fail!!!" << std::endl;
            jsonStream.close();
            return;
        }
        //extract topo node order
        {
            int size = root["topo_order"].size();
            for(int i = 0; i < size; i++)
            {
                std::string executionName;
                executionName = root["topo_order"][i].asString();
                topoExecutionInfoOrder.push_back(executionName);
            }
        }
        //extract input tensor names
        {
            int size = root["input_tensor_names"].size();
            for(int i = 0; i < size; i++)
            {
                std::string tensorName;
                tensorName = root["input_tensor_names"][i].asString();
                inputTensorNames.push_back(tensorName);
            }
        }
        //extract output tensor names
        {
            int size = root["output_tensor_names"].size();
            for(int i = 0; i < size; i++)
            {
                std::string tensorName;
                tensorName = root["output_tensor_names"][i].asString();
                outputTensorNames.push_back(tensorName);
            } 
        }

        // extract execution execution info
        initFlag = extractExecutionInfo(root["execution_info"]);
        jsonStream.close();
        return;
    }
    executionParse::~executionParse()
    {

    }

    bool executionParse::extractExecutionInfo(Json::Value &root)
    {
        CUDARuntime *runtime = getCudaRuntime();
        for (int i = 0; i < topoExecutionInfoOrder.size(); i++) 
        {
            auto elem = topoExecutionInfoOrder[i];
            auto execution_type = root[elem]["type"].asString();
            auto parseNodeInfoFromJsonFunc = getConstructExecutionInfoFuncMap(execution_type);
            if(parseNodeInfoFromJsonFunc != nullptr)
            {
                auto curr_execution = parseNodeInfoFromJsonFunc(runtime, tensorsInfo, root[elem]);
                if(curr_execution == nullptr)
                    return false;
                curr_execution->init(root[elem]);
                executionInfoMap[elem].reset(curr_execution);
            }
            else
            {
                LOG("current not support %s execution type\n", execution_type.c_str());
            }
        }
        return true;
    }

    const std::vector<std::string>& executionParse::getTopoNodeOrder()
    {
        return topoExecutionInfoOrder;
    }
    const std::map<std::string, std::shared_ptr<Buffer>>& executionParse::getTensorsInfo()
    {
        return tensorsInfo;
    }
    const std::map<std::string, std::shared_ptr<ExecutionInfo>>& executionParse::getExecutionInfoMap()
    {
        return executionInfoMap;
    }
    void executionParse::runInference()
    {
        int executionSize = topoExecutionInfoOrder.size();
        for(int i = 0; i < executionSize; i++)
        {
            auto name = topoExecutionInfoOrder[i];
            executionInfoMap[name]->run();
        }
    }
    std::map<std::string, void*> executionParse::getInferenceResult()
    {
        int size = outputTensorNames.size();
        std::map<std::string, void*> result;
        for( int i = 0; i < size; i++)
        {
            auto name = outputTensorNames[i];
            result[name] = tensorsInfo[name]->host<void>();
        }
        return result;
    }
}