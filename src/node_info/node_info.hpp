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
#include "common/utils.hpp"
using namespace std;

namespace TENSORRT_WRAPPER
{

    class NodeInfo
    {
    public:
        NodeInfo() {
            inputs.clear();
            outputs.clear();
        };
        ~NodeInfo() = default;
        void setNodeType(std::string type) { m_type = type; }
        std::string getNodeType() { return m_type; }
        void setNodeSubType(std::string type) { m_sub_type = type; }
        std::string getNodeSubType() { return m_sub_type; }
        std::vector<std::string> getInputs() { return m_inputs; }
        std::vector<std::string> getOutputs() { return m_outputs; }
        void addInput(std::string input) { m_inputs.push_back(input); }
        void addOutput(std::string output) { m_outputs.push_back(output); }
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) = 0;
        void printNodeInfo() {
            LOG("################### NODE INFO ######################\n");
            LOG("currend node type is %s , sub node type is %s\n", m_type.c_str(), m_sub_type.c_str());
            auto input = getInputs();
            LOG("Input tensor size is %d\n", input.size());
            for(int i = 0; i < input.size(); i++) {
                LOG("----index %d tensor : %s\n", i, input[i].c_str());
            }
            auto output = getOutputs();
            LOG("Output tensor size is %d\n", output.size());
            for(int i = 0; i < output.size(); i++) {
                LOG("----index %d tensor : %s\n", i, output[i].c_str());
            }
        }

    private:
        std::string                    m_type;
        std::string                    m_sub_type;
        std::vector<std::string>       m_inputs;
        std::vector<std::string>       m_outputs;
    };

} // namespace TENSORRT_WRAPPER