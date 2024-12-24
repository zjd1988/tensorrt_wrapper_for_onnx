/********************************************
 * Filename: node_info.hpp
 * Created by zjd1988 on 2024/12/19
 * Description:
 ********************************************/
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "json/json.h"
#include "utils.hpp"
using namespace std;

namespace TENSORRT_WRAPPER
{

    class NodeInfo
    {
    public:
        NodeInfo();
        ~NodeInfo();
        void setNodeType(std::string type);
        std::string getNodeType();
        void setNodeSubType(std::string type);
        std::string getNodeSubType();
        std::vector<std::string> getInputs();
        std::vector<std::string> getOutputs();
        void addInput(std::string input);
        void addOutput(std::string output);
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) = 0;
        void printNodeInfo();

    private:
        std::string nodeType;
        std::string subNodeType;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
    };

    typedef NodeInfo* (*nodeParseFunc)(std::string, Json::Value&);

    class NodeParse
    {
    private:
        static NodeParse* instance;
        void registerNodeParseFunc();
        std::map<std::string, nodeParseFunc> nodeParseFuncMap;
        std::map<std::string, std::string> onnxNodeTypeToTensorrtNodeTypeMap;
        NodeParse()
        {
        }
    public:
        nodeParseFunc getNodeParseFunc(std::string nodeType);
        static NodeParse* getInstance() {
            return instance;
        }
    };
    
    extern nodeParseFunc getNodeParseFuncMap(std::string onnxNodeType);
}

#endif 