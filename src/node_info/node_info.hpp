#ifndef __NODE_INFO_HPP__
#define __NODE_INFO_HPP__

#include <iostream>
#include <string>
#include <vector>
#include "json/json.h"
#include "utils.hpp"
using namespace std;

namespace tensorrtInference
{
    class nodeInfo
    {
    public:
        nodeInfo();
        ~nodeInfo();
        void setNodeType(std::string type);
        std::string getNodeType();
        void setSubNodeType(std::string type);
        std::string getSubNodeType();
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

    typedef nodeInfo* (*nodeParseFunc)(std::string, Json::Value&);

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